use std::result::Result as StdResult;

use arrow::array::{RecordBatch, UInt32Array};
use arrow::compute::{concat_batches, take};
use arrow::datatypes::SchemaRef;
use crossbeam_channel::{Receiver, Sender};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::checkpoint::CheckpointCursor;
use crate::dataloader::ShuffleConfig;
use crate::dataset::Dataset;
use crate::distributed::DistributedConfig;
use crate::error::Result;
use crate::ring_buffer::RingBuffer;

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
pub struct ReadTask {
    pub id: usize, // monotonic, assigned at dispatch
    pub row_group_idx: usize,
    pub start_row: usize,
    pub num_rows: usize,
}

/// Emits read tasks to `task_tx` indefinitely, cycling through row groups epoch by epoch,
/// shuffling each epoch's visit order if enabled, and retaining only this rank's strided slice.
pub fn task_dispatcher(
    task_tx: &Sender<ReadTask>,
    row_group_lengths: &[usize],
    chunk_size: usize,
    mut cursor: StreamCursor,
    shuffle_config: ShuffleConfig,
    dist_config: DistributedConfig,
) {
    let mut task_id = 0;
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

        if task_tx
            .send(ReadTask {
                id: task_id,
                row_group_idx,
                start_row: cursor.row_in_group,
                num_rows,
            })
            .is_err()
        {
            return; // consumer dropped
        }
        task_id += 1;
        cursor.advance(num_rows, row_group_length);
    }
}

/// Reads chunks defined by a `ReadTask` from disk and forwards the resulting `RecordBatch` to `batch_tx`.
pub fn chunk_reader(
    task_rx: &Receiver<ReadTask>,
    batch_tx: &Sender<Result<(usize, RecordBatch)>>,
    dataset: &Dataset,
) {
    for ReadTask {
        id: task_id,
        row_group_idx,
        start_row,
        num_rows,
    } in task_rx
    {
        let row_group = &dataset.row_group_index[row_group_idx];
        match dataset.read_row_group_range(row_group, start_row, num_rows) {
            Ok(chunk_data) => {
                if batch_tx.send(Ok((task_id, chunk_data))).is_err() {
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

/// Reorders out-of-order batches from `batch_rx` and sends them in order to `ordered_tx` using a ring buffer.
/// Each batch is placed at slot `id % capacity` and consecutive runs are drained after every insert.
/// `capacity` must be at least as large as the total number of batches that can be in-flight simultaneously
/// (channels + workers) to avoid collisions.
pub fn reorder_batch(
    batch_rx: &Receiver<Result<(usize, RecordBatch)>>,
    ordered_tx: &Sender<Result<RecordBatch>>,
    capacity: usize,
) {
    let mut ring_buffer = RingBuffer::new(capacity);
    loop {
        match batch_rx.recv() {
            Ok(Ok((id, batch))) => {
                ring_buffer.insert(id, batch);
                for batch in ring_buffer.drain() {
                    if ordered_tx.send(Ok(batch)).is_err() {
                        return; // consumer dropped
                    }
                }
            }
            Ok(Err(e)) => {
                let _ = ordered_tx.send(Err(e));
                return; // upstream error
            }
            Err(_) => return, // consumer dropped
        }
    }
}

/// Returns a copy of `buffer` with all rows randomly permuted using `rng`.
fn shuffle_buffer(buffer: &RecordBatch, rng: &mut impl Rng) -> Result<RecordBatch> {
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

/// Accumulates batches from `ordered_rx` until `buffer_size` rows are gathered, concatenates them into one fill,
/// optionally shuffles it, and sends it to `buffer_tx`.
pub fn buffer_builder(
    ordered_rx: &Receiver<Result<RecordBatch>>,
    buffer_tx: &Sender<Result<RecordBatch>>,
    schema: &SchemaRef,
    buffer_size: usize,
    shuffle_config: ShuffleConfig,
    seed_offset: usize,
) {
    let mut seed_offset = seed_offset as u64;
    loop {
        let mut parts: Vec<RecordBatch> = Vec::new();
        let mut rows = 0;
        // accumulate batches until we have enough to fill a buffer
        while rows < buffer_size {
            match ordered_rx.recv() {
                Ok(Ok(chunk_data)) => {
                    rows += chunk_data.num_rows();
                    parts.push(chunk_data);
                }
                Ok(Err(e)) => {
                    let _ = buffer_tx.send(Err(e));
                    return; // upstream error
                }
                Err(_) => return, // consumer dropped
            }
        }
        // build the buffer batch by concatenating the accumulated parts and optionally shuffle
        let buffer = match concat_batches(schema, &parts) {
            Ok(buffer) => {
                if shuffle_config.shuffle {
                    let mut rng = SmallRng::seed_from_u64(shuffle_config.seed + seed_offset);
                    shuffle_buffer(&buffer, &mut rng)
                } else {
                    Ok(buffer)
                }
            }
            Err(e) => {
                let _ = buffer_tx.send(Err(e.into()));
                return; // error concatenating batches
            }
        };
        // send to consumer and advance the seed for the next fill; if the consumer has dropped, stop producing
        match buffer_tx.send(buffer) {
            Ok(()) => seed_offset += 1, // only advance the seed if the buffer was successfully sent
            Err(_) => return,           // consumer dropped
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_possible_wrap)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use crossbeam_channel::{bounded, Receiver};

    use super::reorder_batch;
    use crate::error::{Error, Result};

    /// Build a one-row batch whose single `val` column contains `value`.
    /// Each test stores the dispatch id as the column value so arrival order is verifiable.
    fn make_batch(value: i32) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![value]))]).unwrap()
    }

    fn collect_values(ordered_rx: Receiver<Result<RecordBatch>>) -> Vec<i32> {
        ordered_rx
            .into_iter()
            .map(|r| {
                r.unwrap()
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .value(0)
            })
            .collect()
    }

    #[test]
    fn test_reorder_restores_dispatch_order() {
        let (batch_tx, batch_rx) = bounded(16);
        let (ordered_tx, ordered_rx) = bounded(16);

        let n = 5;
        for id in (0..n).rev() {
            batch_tx.send(Ok((id, make_batch(id as i32)))).unwrap();
        }
        drop(batch_tx);

        thread::spawn(move || reorder_batch(&batch_rx, &ordered_tx, 8));

        assert_eq!(
            collect_values(ordered_rx),
            (0..n as i32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_reorder_flushes_buffered_successors_on_gap_fill() {
        let (batch_tx, batch_rx) = bounded(16);
        let (ordered_tx, ordered_rx) = bounded(16);

        for id in [1, 2, 3, 0] {
            batch_tx.send(Ok((id, make_batch(id as i32)))).unwrap();
        }
        drop(batch_tx);

        thread::spawn(move || reorder_batch(&batch_rx, &ordered_tx, 8));

        assert_eq!(collect_values(ordered_rx), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_reorder_propagates_error() {
        let (batch_tx, batch_rx) = bounded(16);
        let (ordered_tx, ordered_rx) = bounded(16);

        batch_tx.send(Err(Error::IterationNotStarted)).unwrap();
        drop(batch_tx);

        thread::spawn(move || reorder_batch(&batch_rx, &ordered_tx, 8));

        assert!(ordered_rx.recv().unwrap().is_err());
    }
}
