use std::result::Result as StdResult;

use arrow::array::UInt32Array;
use arrow::compute::take;
use arrow::record_batch::RecordBatch;
use crossbeam_channel::Receiver;
use pyo3::Python;
use rand::prelude::*;

use crate::error::Result;

/// Shuffle of all rows in `buffer`.
pub fn shuffle_buffer(buffer: &RecordBatch) -> Result<RecordBatch> {
    let mut rng = rand::rng();
    let n: u32 = u32::try_from(buffer.num_rows())?;
    let mut indices = (0..n).collect::<Vec<_>>();
    indices.shuffle(&mut rng);

    let idx_arr = UInt32Array::from(indices);
    let columns: Vec<_> = buffer
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
}

impl Buffer {
    pub fn new(rx: Receiver<Result<RecordBatch>>, shuffle: bool) -> Self {
        Self {
            rx,
            shuffle,
            data: None,
            offset: 0,
        }
    }

    fn need_refill(&self, num_rows: usize) -> bool {
        self.data
            .as_ref()
            .is_none_or(|batch| self.offset + num_rows > batch.num_rows())
    }

    pub fn take(&mut self, num_rows: usize, py: Python<'_>) -> Result<Option<RecordBatch>> {
        if self.need_refill(num_rows) {
            match py.detach(|| self.rx.recv()) {
                Ok(Ok(buffer)) => {
                    self.data = if self.shuffle {
                        Some(shuffle_buffer(&buffer)?)
                    } else {
                        Some(buffer)
                    };
                    self.offset = 0;
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => return Ok(None), // channel closed, no more batches
            }
        }

        let length = num_rows.min(self.data.as_ref().unwrap().num_rows() - self.offset);
        let batch = self.data.as_ref().unwrap().slice(self.offset, length);
        self.offset += batch.num_rows();
        Ok(Some(batch))
    }
}
