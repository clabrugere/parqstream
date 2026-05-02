use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use crossbeam_channel::{Receiver, Sender};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::checkpoint::CheckpointCursor;
use crate::dataset::Dataset;
use crate::error::Result;

/// Tracks the dispatcher's position within the infinite epoch/row-group stream.
#[derive(Debug)]
pub struct StreamCursor {
    pub epoch: usize,
    pub row_group: usize,
    pub row_in_group: usize,
}

impl StreamCursor {
    /// Advances past `num_rows`, rolling over to the next row group when the current one is exhausted.
    pub fn advance(&mut self, num_rows: usize, row_group_length: usize) {
        self.row_in_group += num_rows;
        if self.row_in_group >= row_group_length {
            self.row_group += 1;
            self.row_in_group = 0;
        }
    }
    /// Resets to the start of the next epoch.
    pub fn new_epoch(&mut self) {
        self.epoch += 1;
        self.row_group = 0;
        self.row_in_group = 0;
    }
}

impl From<&CheckpointCursor> for StreamCursor {
    fn from(cursor: &CheckpointCursor) -> Self {
        Self {
            epoch: cursor.stream_epoch,
            row_group: cursor.row_group,
            row_in_group: cursor.row_in_group,
        }
    }
}

/// A read task: a contiguous slice of `num_rows` rows at `start_row` within a single row group.
#[derive(Debug)]
pub struct Chunk {
    pub row_group_idx: usize,
    pub start_row: usize,
    pub num_rows: usize,
}

/// Emits read tasks to `chunk_tx` indefinitely, cycling through row groups epoch by epoch and
/// shuffling each epoch's visit order if enabled.
pub fn chunk_dispatcher(
    chunk_tx: &Sender<Chunk>,
    row_group_lengths: &[usize],
    chunk_size: usize,
    shuffle: bool,
    seed: u64,
    mut cursor: StreamCursor,
) {
    let num_groups = row_group_lengths.len();
    // Row group visit order for the current epoch. Shuffle seed = seed + epoch so
    // each epoch gets an independent, deterministic order (mirrors locate_row_in_order in checkpoint.rs).
    let mut order = (0..num_groups).collect::<Vec<_>>();
    if shuffle {
        order.shuffle(&mut SmallRng::seed_from_u64(seed + cursor.epoch as u64));
    }

    loop {
        if cursor.row_group >= num_groups {
            cursor.new_epoch();
            if shuffle {
                // Reset to identity before shuffling: re-shuffling a non-identity slice would
                // make epoch N's order depend on epoch N-1's, breaking checkpoint resume.
                order = (0..num_groups).collect();
                order.shuffle(&mut SmallRng::seed_from_u64(seed + cursor.epoch as u64));
            }
        }
        let row_group_idx = order[cursor.row_group];
        let row_group_length = row_group_lengths[row_group_idx];
        let num_rows = chunk_size.min(row_group_length - cursor.row_in_group);

        if chunk_tx
            .send(Chunk {
                row_group_idx,
                start_row: cursor.row_in_group,
                num_rows,
            })
            .is_err()
        {
            return; // consumer dropped
        }
        cursor.advance(num_rows, row_group_length);
    }
}

/// Reads each `Chunk` from disk and forwards the resulting `RecordBatch` to `batch_tx`.
pub fn chunk_reader(
    chunk_rx: &Receiver<Chunk>,
    batch_tx: &Sender<Result<RecordBatch>>,
    dataset: &Dataset,
) {
    for Chunk {
        row_group_idx,
        start_row,
        num_rows,
    } in chunk_rx
    {
        let meta = &dataset.row_group_index[row_group_idx];
        match dataset.read_row_group_range(meta.file_idx, meta.row_group_idx, start_row, num_rows) {
            Ok(chunk_data) => {
                if batch_tx.send(Ok(chunk_data)).is_err() {
                    return; // consumer dropped
                }
            }
            Err(e) => {
                let _ = batch_tx.send(Err(e));
                return; // upstream error
            }
        }
    }
}

/// Accumulates batches from `batch_rx` until `buffer_size` rows are gathered, concatenates them
/// into one fill, and sends it to `prefetch_tx`.
pub fn chunk_collector(
    batch_rx: &Receiver<Result<RecordBatch>>,
    prefetch_tx: &Sender<Result<RecordBatch>>,
    schema: &SchemaRef,
    buffer_size: usize,
) {
    loop {
        let mut parts: Vec<RecordBatch> = Vec::new();
        let mut rows = 0;
        while rows < buffer_size {
            match batch_rx.recv() {
                Ok(Ok(chunk_data)) => {
                    rows += chunk_data.num_rows();
                    parts.push(chunk_data);
                }
                Ok(Err(e)) => {
                    let _ = prefetch_tx.send(Err(e));
                    return; // upstream error
                }
                Err(_) => return, // consumer dropped
            }
        }
        let buffer = match concat_batches(schema, &parts) {
            Ok(buffer) => buffer,
            Err(e) => {
                let _ = prefetch_tx.send(Err(e.into()));
                return; // error concatenating batches
            }
        };
        if prefetch_tx.send(Ok(buffer)).is_err() {
            return; // consumer dropped
        }
    }
}
