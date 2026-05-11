use arrow::record_batch::RecordBatch;

use crate::error::Result;
use crate::pipeline::{Pipeline, StitchedBuffer};

/// Point-in-time view of a [`Buffer`], used by checkpoint to reconstruct the resume position.
#[derive(Debug, Default)]
pub struct BufferSnapshot {
    pub offset: usize,
    /// Number of refills that have occurred; used to derive the shuffle seed for the next fill.
    pub seed_offset: usize,
    /// Rows stitched from the previous fill into the current one (see `Buffer::refill`).
    pub tail_size: usize,
}

/// Wraps the `Pipeline` receiver and serves row slices of a requested size.
/// Unconsumed rows from the previous buffer are stitched to the front of the next by the
/// `buffer_stitcher` thread; this struct owns the channels used for that handshake.
#[derive(Debug)]
pub struct Buffer {
    pipeline: Pipeline,
    data: Option<RecordBatch>,
    offset: usize,
    /// Incremented after each refill; the nth buffer is shuffled with `SmallRng(seed + seed_offset_at_that_refill)`.
    seed_offset: usize,
    /// Applied once on the first refill to skip into a partially-consumed buffer (checkpoint resume).
    resume_offset: usize,
    /// Rows prepended from the previous fill; snapshotted so checkpoint can correct its cursor.
    tail_size: usize,
}

impl Buffer {
    pub fn new(pipeline: Pipeline, seed_offset: usize, resume_offset: usize) -> Self {
        let _ = pipeline.tail_tx.send(None); // ensure the tail channel is initialized for the first refill
        Self {
            pipeline,
            data: None,
            offset: 0,
            seed_offset,
            resume_offset,
            tail_size: 0,
        }
    }

    fn need_refill(&self, num_rows: usize) -> bool {
        self.data
            .as_ref()
            .is_none_or(|batch| self.offset + num_rows > batch.num_rows())
    }

    /// Fetches the next stitched buffer from the pipeline, blocking until it is ready.
    fn refill(&mut self) -> Result<bool> {
        // Send the current tail BEFORE blocking on ready_rx — the stitcher already has the next
        // buffer pre-fetched and only needs the tail to complete the stitch.
        if let Some(data) = &self.data {
            let tail = (self.offset < data.num_rows())
                .then(|| data.slice(self.offset, data.num_rows() - self.offset));
            let _ = self.pipeline.tail_tx.send(tail);
        }

        match self.pipeline.ready_rx.recv() {
            Ok(Ok(StitchedBuffer { data, tail_size })) => {
                self.tail_size = tail_size;
                self.data = Some(data);
                self.offset = self.resume_offset;
                self.resume_offset = 0;
                self.seed_offset += 1;
                Ok(true)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(false), // channel closed = exhausted
        }
    }

    /// Returns the next `num_rows` rows, triggering a refill if needed. Returns `None` when the channel is closed.
    pub fn take(&mut self, num_rows: usize) -> Result<Option<RecordBatch>> {
        if self.need_refill(num_rows) && !self.refill()? {
            return Ok(None);
        }

        let data = self.data.as_ref().unwrap();
        let length = num_rows.min(data.num_rows() - self.offset);
        let batch = data.slice(self.offset, length);
        self.offset += length;
        Ok(Some(batch))
    }

    /// Captures the current offset, seed state, and tail size for use in a [`BufferSnapshot`].
    pub fn snapshot(&self) -> BufferSnapshot {
        BufferSnapshot {
            offset: self.offset,
            seed_offset: self.seed_offset,
            tail_size: self.tail_size,
        }
    }
}
