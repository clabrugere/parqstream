use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use crossbeam_channel::{Receiver, Sender};

use crate::checkpoint::CheckpointCursor;
use crate::dataloader::ShuffleConfig;
use crate::dataset::Dataset;
use crate::distributed::DistributedConfig;
use crate::error::Result;

/// Tracks the dispatcher's position within the infinite epoch/row-group stream.
#[derive(Debug)]
pub struct StreamCursor {
    pub epoch: usize,
    pub row_group_pos: usize, // position within the rank-local epoch order
    pub row_in_group: usize,
}

impl StreamCursor {
    /// Advances past `num_rows`, rolling over to the next row group when the current one is exhausted.
    pub fn advance(&mut self, num_rows: usize, row_group_length: usize) {
        self.row_in_group += num_rows;
        if self.row_in_group >= row_group_length {
            self.row_group_pos += 1;
            self.row_in_group = 0;
        }
    }
    /// Resets to the start of the next epoch.
    pub fn new_epoch(&mut self) {
        self.epoch += 1;
        self.row_group_pos = 0;
        self.row_in_group = 0;
    }
}

impl From<&CheckpointCursor> for StreamCursor {
    fn from(cursor: &CheckpointCursor) -> Self {
        Self {
            epoch: cursor.stream_epoch,
            row_group_pos: cursor.row_group_pos,
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

/// Emits read tasks to `chunk_tx` indefinitely, cycling through row groups epoch by epoch,
/// shuffling each epoch's visit order if enabled, and retaining only this rank's strided slice.
pub fn chunk_dispatcher(
    chunk_tx: &Sender<Chunk>,
    row_group_lengths: &[usize],
    chunk_size: usize,
    mut cursor: StreamCursor,
    shuffle_config: ShuffleConfig,
    dist_config: DistributedConfig,
) {
    let num_global = row_group_lengths.len();
    let mut order = dist_config.epoch_order(shuffle_config, cursor.epoch, num_global);
    // rank_groups is invariant across epochs: same dataset and same dist_config always produce the same count.
    let rank_groups = order.len();

    loop {
        if cursor.row_group_pos >= rank_groups {
            cursor.new_epoch();
            order = dist_config.epoch_order(shuffle_config, cursor.epoch, num_global);
        }
        let row_group_idx = order[cursor.row_group_pos];
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
        let row_group = &dataset.row_group_index[row_group_idx];
        match dataset.read_row_group_range(row_group, start_row, num_rows) {
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
