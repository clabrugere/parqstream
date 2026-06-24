use arrow::compute::concat_batches;
use arrow::record_batch::RecordBatch;
use crossbeam_channel::Receiver;

use crate::error::Result;
use crate::position::BufferPosition;

/// Point-in-time view of a [`Buffer`], used by checkpoint to reconstruct the resume position.
#[derive(Debug, Default)]
pub struct BufferSnapshot {
    pub position: BufferPosition,
    /// Rows stitched from the previous fill into the current one (see `Buffer::refill`).
    pub tail_size: usize,
}

/// Wraps the [`Pipeline`] receiver and serves row slices of a requested size.
/// Unconsumed rows from the previous buffer are stitched to the front of the next buffer.
#[derive(Debug)]
pub struct Buffer {
    rx: Receiver<Result<RecordBatch>>,
    data: Option<RecordBatch>,
    offset: usize,
    /// Incremented after each refill; the nth buffer is shuffled with `SmallRng(seed + refill_count_at_that_refill)`.
    refill_count: usize,
    /// Applied once on the first refill to skip into a partially-consumed buffer (checkpoint resume).
    resume_offset: usize,
    /// Rows prepended from the previous fill; snapshotted so checkpoint can correct its cursor.
    tail_size: usize,
}

impl Buffer {
    pub fn new(rx: Receiver<Result<RecordBatch>>, position: BufferPosition) -> Self {
        Self {
            rx,
            data: None,
            offset: 0,
            refill_count: position.refill_count,
            resume_offset: position.offset,
            tail_size: 0,
        }
    }

    fn need_refill(&self, num_rows: usize) -> bool {
        self.data
            .as_ref()
            .is_none_or(|batch| self.offset + num_rows > batch.num_rows())
    }

    /// Refill the buffer by receiving the next batch from the channel and concatenating any unconsumed rows from the previous batch.
    /// Returns `Ok(false)` if the channel is closed.
    fn refill(&mut self) -> Result<bool> {
        let tail = self.data.as_ref().and_then(|data| {
            (self.offset < data.num_rows())
                .then(|| data.slice(self.offset, data.num_rows() - self.offset))
        });
        let tail_size = tail.as_ref().map_or(0, RecordBatch::num_rows);

        match self.rx.recv() {
            Ok(Ok(next)) => {
                self.data = Some(match tail {
                    Some(tail) => concat_batches(&next.schema(), [&tail, &next])?,
                    None => next,
                });
                self.tail_size = tail_size;
                self.offset = self.resume_offset;
                self.resume_offset = 0;
                self.refill_count += 1;
                Ok(true)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(false),
        }
    }

    /// Returns the next `num_rows` rows, triggering a refill if needed. Returns `None` when the channel is closed.
    pub fn take(&mut self, num_rows: usize) -> Result<Option<RecordBatch>> {
        if self.need_refill(num_rows) && !self.refill()? {
            return Ok(None);
        }

        let data = self
            .data
            .as_ref()
            .expect("data is Some after a successful refill");
        let length = num_rows.min(data.num_rows() - self.offset);
        let batch = data.slice(self.offset, length);
        self.offset += length;
        Ok(Some(batch))
    }

    /// Captures the current position (offset and refill count) and tail size for use in a [`BufferSnapshot`].
    pub fn snapshot(&self) -> BufferSnapshot {
        let position = BufferPosition {
            offset: self.offset,
            refill_count: self.refill_count,
        };
        BufferSnapshot {
            position,
            tail_size: self.tail_size,
        }
    }
}
