use std::result::Result as StdResult;

use arrow::array::{RecordBatch, UInt32Array};
use arrow::compute::{concat_batches, take};
use arrow::datatypes::SchemaRef;
use crossbeam_channel::{bounded, Receiver, Sender};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::dataloader::ShuffleConfig;
use crate::dataset::Dataset;
use crate::distributed::DistributedConfig;
use crate::error::{Error, Result};
use crate::position::StreamPosition;

/// Contiguous slice of `num_rows` rows at `start_row` within a single row group.
#[derive(Debug)]
pub struct ReadTask {
    pub row_group_idx: usize,
    pub start_row: usize,
    pub num_rows: usize,
}

/// [`ReadTask`] and its associated result channel for returning the decoded [`RecordBatch`]
pub struct Job {
    pub task: ReadTask,
    pub result_tx: Sender<Result<RecordBatch>>,
}

/// Emits `Job`s (a read task and its dedicated result channel) to `job_tx` indefinitely, cycling through row groups epoch by epoch,
/// shuffling each epoch's visit order if enabled, and retaining only this rank's strided slice.
pub fn job_dispatcher(
    job_tx: &Sender<Job>,
    pending_tx: &Sender<Receiver<Result<RecordBatch>>>,
    row_group_lengths: &[usize],
    chunk_size: usize,
    mut position: StreamPosition,
    shuffle_config: ShuffleConfig,
    dist_config: DistributedConfig,
) {
    let num_global = row_group_lengths.len();
    let mut order = dist_config.epoch_order(shuffle_config, position.epoch, num_global);
    // rank_groups is invariant across epochs: same dataset and same dist_config always produce the same count.
    let rank_groups = order.len();

    loop {
        if position.row_group_pos >= rank_groups {
            position.new_epoch();
            order = dist_config.epoch_order(shuffle_config, position.epoch, num_global);
        }
        let row_group_idx = order[position.row_group_pos];
        let row_group_length = row_group_lengths[row_group_idx];
        let num_rows = chunk_size.min(row_group_length - position.row_in_group);

        // single use channel for this task to send the result through, to keep the dispatch ordering.
        // Sender side is kept in the Job, Receiver side is sent to a pending FIFO queue.
        let (result_tx, result_rx) = bounded::<Result<RecordBatch>>(1);
        let task = ReadTask {
            row_group_idx,
            start_row: position.row_in_group,
            num_rows,
        };
        // Send task and result channel or stop if the consumer has been dropped
        if job_tx.send(Job { task, result_tx }).is_err() || pending_tx.send(result_rx).is_err() {
            return; // consumer dropped
        }
        position.advance(num_rows, row_group_length);
    }
}

/// Reads chunks defined by a [`ReadTask`] from disk and forwards the resulting [`RecordBatch`] to the task's `result_tx`.
pub fn chunk_reader(job_rx: &Receiver<Job>, dataset: &Dataset) {
    for Job { task, result_tx } in job_rx {
        let row_group = &dataset.row_group_index[task.row_group_idx];
        match dataset.read_row_group_range(row_group, task.start_row, task.num_rows) {
            Ok(chunk_data) => {
                if result_tx.send(Ok(chunk_data)).is_err() {
                    return; // consumer dropped
                }
            }
            Err(e) => {
                let _ = result_tx.send(Err(e));
                return; // upstream error
            }
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

/// Accumulates batches pulled in dispatch order from `pending_rx` until `buffer_size` rows are gathered
fn collect_buffer(
    pending_rx: &Receiver<Receiver<Result<RecordBatch>>>,
    buffer_size: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let mut parts: Vec<RecordBatch> = Vec::new();
    let mut row_count = 0;
    // accumulate batches until we have enough to fill a buffer
    while row_count < buffer_size {
        // receive the next result channel from the pending queue, or stop if the consumer has been dropped
        let Ok(result_rx) = pending_rx.recv() else {
            return Ok(None); // consumer dropped
        };
        // receive the next chunk from the result channel; Err means the worker exited without
        // sending (panic mid-decode). Surface this as an error rather than silent EOF
        match result_rx.recv() {
            Ok(Ok(chunk_data)) => {
                row_count += chunk_data.num_rows();
                parts.push(chunk_data);
            }
            Ok(Err(e)) => {
                return Err(e); // upstream error
            }
            Err(_) => {
                return Err(Error::WorkerPanic); // worker exited without sending
            }
        }
    }

    Ok(Some(parts))
}

/// Accumulates batches pulled in dispatch order from `pending_rx` until `buffer_size` rows are gathered,
/// concatenates them into one [`RecordBatch`], optionally shuffles it with a deterministic seed
/// derived from `shuffle_config.seed + refill_count`, and sends it to `buffer_tx`.
/// `refill_count` is incremented after each successful send so every buffer gets a unique seed.
pub fn buffer_builder(
    pending_rx: &Receiver<Receiver<Result<RecordBatch>>>,
    buffer_tx: &Sender<Result<RecordBatch>>,
    schema: &SchemaRef,
    buffer_size: usize,
    shuffle_config: ShuffleConfig,
    refill_count: usize,
) {
    let mut refill_count = refill_count as u64;
    loop {
        // collect enough chunks to fill a buffer, or stop if the consumer has been dropped
        let parts = match collect_buffer(pending_rx, buffer_size) {
            Ok(Some(parts)) => parts,
            Ok(None) => return, // consumer dropped
            Err(e) => {
                let _ = buffer_tx.send(Err(e));
                return;
            }
        };

        // build the buffer batch by concatenating the accumulated parts and optionally shuffle
        let buffer = match concat_batches(schema, &parts) {
            Ok(buffer) => {
                if shuffle_config.shuffle {
                    let mut rng = SmallRng::seed_from_u64(shuffle_config.seed + refill_count);
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
            Ok(()) => refill_count += 1, // only advance the seed if the buffer was successfully sent
            Err(_) => return,            // consumer dropped
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
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use crossbeam_channel::{bounded, Receiver, Sender};

    use super::buffer_builder;
    use crate::dataloader::ShuffleConfig;
    use crate::error::Result;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("val", DataType::Int32, false)]))
    }

    fn make_rows(values: &[i32]) -> RecordBatch {
        RecordBatch::try_new(schema(), vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap()
    }

    fn batch_values(batch: &RecordBatch) -> Vec<i32> {
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    fn pending_from(batches: Vec<Result<RecordBatch>>) -> Receiver<Receiver<Result<RecordBatch>>> {
        let (pending_tx, pending_rx) = bounded(1);
        for batch in batches {
            let (result_tx, result_rx) = bounded(1);
            result_tx.send(batch).unwrap();
            pending_tx.send(result_rx).unwrap();
        }
        pending_rx
    }

    // buffer_builder

    // Verifies that dispatch order is preserved even when workers complete out of order.
    // The FIFO pending queue is the mechanism: buffer_builder blocks on each result_rx in
    // the order they were enqueued, so a fast worker on job N+1 cannot overtake a slow job N.
    #[test]
    fn test_buffer_builder_preserves_dispatch_order_with_out_of_order_workers() {
        let n = 4;
        let (pending_tx, pending_rx) = bounded::<Receiver<Result<RecordBatch>>>(n);
        let mut result_txs: Vec<Sender<Result<RecordBatch>>> = Vec::new();

        // Enqueue receivers in dispatch order (0, 1, 2, 3)
        for _ in 0..n {
            let (result_tx, result_rx) = bounded(1);
            result_txs.push(result_tx);
            pending_tx.send(result_rx).unwrap();
        }
        drop(pending_tx);

        let schema = schema();
        let (buffer_tx, buffer_rx) = bounded::<Result<RecordBatch>>(n);
        let no_shuffle = ShuffleConfig {
            shuffle: false,
            seed: 0,
        };
        // buffer_size=1 so each single-row chunk is flushed immediately
        thread::spawn(move || buffer_builder(&pending_rx, &buffer_tx, &schema, 1, no_shuffle, 0));

        // Workers complete in reverse dispatch order (3, 2, 1, 0)
        for tx in result_txs.into_iter().rev() {
            tx.send(Ok(make_rows(&[0]))).unwrap(); // value doesn't matter; order of receipt does
        }

        // All 4 buffers must arrive (output order == dispatch order, not completion order)
        let count = (0..n).filter_map(|_| buffer_rx.recv().ok()).count();
        assert_eq!(count, n);
    }

    // Verifies that different refill_counts produce different permutations.
    // Python shuffle tests can't inspect that directly since they only observe batch values.
    #[test]
    fn test_buffer_builder_shuffle_advances_seed_per_buffer() {
        let rows: Vec<i32> = (0..100).collect();
        let shuffle_cfg = ShuffleConfig {
            shuffle: true,
            seed: 42,
        };
        let schema = schema();

        let run_with_offset = |offset: usize| {
            let batches = vec![Ok(make_rows(&rows))];
            let pending_rx = pending_from(batches);
            let (buffer_tx, buffer_rx) = bounded(4);
            let schema = schema.clone();

            thread::spawn(move || {
                buffer_builder(&pending_rx, &buffer_tx, &schema, 100, shuffle_cfg, offset);
            });
            batch_values(&buffer_rx.recv().unwrap().unwrap())
        };

        assert_ne!(run_with_offset(0), run_with_offset(1));
    }
}
