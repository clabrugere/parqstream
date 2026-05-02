use std::result::Result as StdResult;

use arrow::array::UInt32Array;
use arrow::compute::{concat_batches, take};
use arrow::record_batch::RecordBatch;
use crossbeam_channel::Receiver;
use pyo3::Python;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::error::Result;

/// Returns a copy of `buffer` with all rows randomly permuted using `rng`.
pub fn shuffle_buffer(buffer: &RecordBatch, rng: &mut impl Rng) -> Result<RecordBatch> {
    let n = u32::try_from(buffer.num_rows())?;
    let mut indices = (0..n).collect::<Vec<_>>();
    indices.shuffle(rng);

    let idx_arr = UInt32Array::from(indices);
    let columns = buffer
        .columns()
        .iter()
        .map(|col| take(col, &idx_arr, None))
        .collect::<StdResult<_, _>>()?;

    Ok(RecordBatch::try_new(buffer.schema(), columns)?)
}

/// Point-in-time view of a [`Buffer`], used by checkpoint to reconstruct the resume position.
#[derive(Debug, Default)]
pub struct BufferSnapshot {
    pub offset: usize,
    /// Number of refills that have occurred; used to derive the shuffle seed for the next fill.
    pub seed_offset: usize,
    /// Rows stitched from the previous fill into the current one (see `Buffer::refill`).
    pub tail_size: usize,
}

/// Wraps the prefetch receiver and serves row slices of a requested size. Each fill is optionally
/// shuffled; unconsumed rows from the previous fill are stitched to the front of the next.
#[derive(Debug)]
pub struct Buffer {
    prefetch_rx: Receiver<Result<RecordBatch>>,
    shuffle: bool,
    data: Option<RecordBatch>,
    offset: usize,
    seed: u64,
    /// Incremented after each refill; the nth fill is shuffled with `SmallRng(seed + seed_offset_at_that_fill)`.
    seed_offset: usize,
    /// Applied once on the first refill to skip into a partially-consumed buffer (checkpoint resume).
    resume_offset: usize,
    /// Rows prepended from the previous fill; snapshotted so checkpoint can correct its cursor.
    tail_size: usize,
}

impl Buffer {
    pub fn new(
        prefetch_rx: Receiver<Result<RecordBatch>>,
        shuffle: bool,
        seed: u64,
        seed_offset: usize,
        resume_offset: usize,
    ) -> Self {
        Self {
            prefetch_rx,
            shuffle,
            data: None,
            offset: 0,
            seed,
            seed_offset,
            resume_offset,
            tail_size: 0,
        }
    }

    fn maybe_shuffle(&self, batch: RecordBatch) -> Result<RecordBatch> {
        if self.shuffle {
            let mut rng = SmallRng::seed_from_u64(self.seed + self.seed_offset as u64);
            shuffle_buffer(&batch, &mut rng)
        } else {
            Ok(batch)
        }
    }

    fn need_refill(&self, num_rows: usize) -> bool {
        self.data
            .as_ref()
            .is_none_or(|batch| self.offset + num_rows > batch.num_rows())
    }

    /// Fetches the next fill, prepending any unconsumed tail from the previous fill. The tail keeps
    /// its previous shuffle order; only the fresh data is shuffled with the new seed.
    fn refill(&mut self, py: Python<'_>) -> Result<bool> {
        let remaining_rows = self.data.as_ref().and_then(|data| {
            (self.offset < data.num_rows())
                .then(|| data.slice(self.offset, data.num_rows() - self.offset))
        });

        match py.detach(|| self.prefetch_rx.recv()) {
            Ok(Ok(next)) => {
                let next = self.maybe_shuffle(next)?;
                // Record tail size before stitching so checkpoint.rs can locate the fresh-data boundary.
                self.tail_size = remaining_rows.as_ref().map_or(0, RecordBatch::num_rows);
                self.data = Some(match remaining_rows {
                    Some(remaining_rows) => {
                        concat_batches(&next.schema(), &[remaining_rows, next])?
                    }
                    None => next,
                });
                // resume_offset is non-zero only on the first refill after a checkpoint restore;
                // it skips past the tail portion that cannot be reconstructed on resume.
                self.offset = self.resume_offset;
                self.resume_offset = 0;
                self.seed_offset += 1;
                Ok(true)
            }
            Ok(Err(e)) => Err(e), // upstream error
            Err(_) => Ok(false),  // channel closed, no more batches
        }
    }

    /// Returns the next `num_rows` rows, triggering a refill if needed. Returns `None` when the channel is closed.
    pub fn take(&mut self, num_rows: usize, py: Python<'_>) -> Result<Option<RecordBatch>> {
        if self.need_refill(num_rows) && !self.refill(py)? {
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
