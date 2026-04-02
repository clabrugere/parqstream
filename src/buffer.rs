use std::result::Result as StdResult;

use arrow::array::UInt32Array;
use arrow::compute::{concat_batches, take};
use arrow::record_batch::RecordBatch;
use crossbeam_channel::Receiver;
use pyo3::Python;
use rand::prelude::*;

use crate::error::Result;

/// Shuffle of all rows in `buffer`.
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

pub struct Buffer {
    rx: Receiver<Result<RecordBatch>>,
    shuffle: bool,
    data: Option<RecordBatch>,
    offset: usize,
    rng: SmallRng,
}

impl Buffer {
    pub fn new(rx: Receiver<Result<RecordBatch>>, shuffle: bool, seed: Option<u64>) -> Self {
        let rng = seed.map_or_else(rand::make_rng, SmallRng::seed_from_u64);
        Self {
            rx,
            shuffle,
            data: None,
            offset: 0,
            rng,
        }
    }

    fn need_refill(&self, num_rows: usize) -> bool {
        self.data
            .as_ref()
            .is_none_or(|batch| self.offset + num_rows > batch.num_rows())
    }

    // fetch the next buffer from the channel, stitching any remaining rows from the current buffer
    fn refill(&mut self, py: Python<'_>) -> Result<bool> {
        let remaining_rows = self.data.as_ref().and_then(|data| {
            (self.offset < data.num_rows())
                .then(|| data.slice(self.offset, data.num_rows() - self.offset))
        });

        match py.detach(|| self.rx.recv()) {
            Ok(Ok(next)) => {
                let next = if self.shuffle {
                    shuffle_buffer(&next, &mut self.rng)?
                } else {
                    next
                };
                self.data = Some(match remaining_rows {
                    Some(remaining_rows) => {
                        concat_batches(&next.schema(), &[remaining_rows, next])?
                    }
                    None => next,
                });
                self.offset = 0;
                Ok(true)
            }
            Ok(Err(e)) => Err(e), // upstream error
            Err(_) => Ok(false),  // channel closed, no more batches
        }
    }

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
}
