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

/// Tracks the position within the infinite epoch/row-group stream.
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

/// Channels connecting the `buffer_stitcher` thread to the main-thread [`Buffer`]. Returned by `spawn_pipeline`.
#[derive(Debug)]
pub struct Pipeline {
    pub ready_rx: Receiver<Result<StitchedBuffer>>,
    pub tail_tx: Sender<Option<RecordBatch>>,
}

/// A buffer delivered by `buffer_stitcher`: the assembled and shuffled `RecordBatch` with the tail from the previous
///  buffer prepended, plus `tail_size` (number of prepended rows) for checkpoint accounting.
#[derive(Debug)]
pub struct StitchedBuffer {
    pub data: RecordBatch,
    pub tail_size: usize,
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

/// Reads chunks defined by a [`ReadTask`] from disk and forwards the resulting [`RecordBatch`] to `batch_tx`.
pub fn chunk_reader(
    task_rx: &Receiver<ReadTask>,
    batch_tx: &Sender<Result<(usize, RecordBatch)>>,
    dataset: &Dataset,
) {
    for ReadTask {
        id,
        row_group_idx,
        start_row,
        num_rows,
    } in task_rx
    {
        let row_group = &dataset.row_group_index[row_group_idx];
        match dataset.read_row_group_range(row_group, start_row, num_rows) {
            Ok(chunk_data) => {
                if batch_tx.send(Ok((id, chunk_data))).is_err() {
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

/// Reorders out-of-order batches from `batch_rx` and sends them in order to `ordered_tx` using a [`RingBuffer`].
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

/// Accumulates ordered batches from `ordered_rx` until `buffer_size` rows are gathered,
/// concatenates them into one [`RecordBatch`], optionally shuffles it with a deterministic seed
/// derived from `shuffle_config.seed + seed_offset`, and sends it to `buffer_tx`.
/// `seed_offset` is incremented after each successful send so every buffer gets a unique seed.
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

/// Prepends the unconsumed tail of the previous buffer to each new buffer, then forwards the
/// stitched result to `ready_tx` for the main-thread [`Buffer`] to consume.
///
/// Protocol (per iteration):
/// 1. Pre-fetch the next assembled buffer from `buffer_rx`. Overlaps with the main thread consuming the previous buffer.
/// 2. Wait for the tail signal on `tail_rx`. [`Buffer::new`] sends `None` upon creation so the first buffer
///    is delivered with no tail. Subsequent signals carry the unconsumed rows.
/// 3. Prepend the tail (if any), record `tail_size`, and send a [`StitchedBuffer`] to `ready_tx`.
pub fn buffer_stitcher(
    buffer_rx: &Receiver<Result<RecordBatch>>,
    tail_rx: &Receiver<Option<RecordBatch>>,
    ready_tx: &Sender<Result<StitchedBuffer>>,
    schema: &SchemaRef,
) {
    loop {
        // Pre-fetch eagerly. By the time the main thread exhausts the current buffer and sends the tail
        let next = match buffer_rx.recv() {
            Ok(Ok(next)) => next,
            Ok(Err(e)) => {
                let _ = ready_tx.send(Err(e));
                return; // upstream error
            }
            Err(_) => return, // channel closed
        };
        // Wait for the tail from [`Buffer`] (None on the first call, sent by [`Buffer::new`]).
        let Ok(tail) = tail_rx.recv() else {
            return; // consumer dropped
        };

        let tail_size = tail.as_ref().map_or(0, RecordBatch::num_rows);
        let data = match tail {
            Some(tail) => match concat_batches(schema, [&tail, &next]) {
                Ok(data) => data,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.into()));
                    return;
                }
            },
            None => next,
        };

        if ready_tx
            .send(Ok(StitchedBuffer { data, tail_size }))
            .is_err()
        {
            return; // consumer dropped
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
    use crossbeam_channel::bounded;

    use super::{buffer_builder, buffer_stitcher, reorder_batch};
    use crate::dataloader::ShuffleConfig;
    use crate::error::Error;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("val", DataType::Int32, false)]))
    }

    fn make_batch(value: i32) -> RecordBatch {
        make_rows(&[value])
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

    // ── reorder_batch ────────────────────────────────────────────────────────

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

        let values: Vec<i32> = ordered_rx
            .into_iter()
            .map(|r| batch_values(&r.unwrap())[0])
            .collect();
        assert_eq!(values, (0..n as i32).collect::<Vec<_>>());
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

        let values: Vec<i32> = ordered_rx
            .into_iter()
            .map(|r| batch_values(&r.unwrap())[0])
            .collect();
        assert_eq!(values, vec![0, 1, 2, 3]);
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

    // ── buffer_builder ───────────────────────────────────────────────────────

    // Verifies that different seed_offsets produce different permutations.
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
            let (ordered_tx, ordered_rx) = bounded(128);
            let (buffer_tx, buffer_rx) = bounded(4);
            let schema = schema.clone();
            ordered_tx.send(Ok(make_rows(&rows))).unwrap();
            drop(ordered_tx);
            thread::spawn(move || {
                buffer_builder(&ordered_rx, &buffer_tx, &schema, 100, shuffle_cfg, offset);
            });
            batch_values(&buffer_rx.recv().unwrap().unwrap())
        };

        assert_ne!(run_with_offset(0), run_with_offset(1));
    }

    // ── buffer_stitcher ──────────────────────────────────────────────────────

    // Verifies the tail_size contract and row prepending that checkpoint.rs relies on.
    // Python tests observe batch values but cannot inspect tail_size directly.
    #[test]
    fn test_stitcher_prepends_tail_to_next_buffer() {
        let (buffer_tx, buffer_rx) = bounded(4);
        let (tail_tx, tail_rx) = bounded(4);
        let (ready_tx, ready_rx) = bounded(4);
        let schema = schema();

        buffer_tx.send(Ok(make_rows(&[1, 2, 3]))).unwrap();
        buffer_tx.send(Ok(make_rows(&[4, 5, 6]))).unwrap();
        drop(buffer_tx);

        thread::spawn(move || buffer_stitcher(&buffer_rx, &tail_rx, &ready_tx, &schema));

        // Bootstrap: first buffer has no tail.
        tail_tx.send(None).unwrap();
        let s0 = ready_rx.recv().unwrap().unwrap();
        assert_eq!(s0.tail_size, 0);
        assert_eq!(batch_values(&s0.data), vec![1, 2, 3]);

        // Simulate Buffer consuming 2 of 3 rows; row [3] remains as the tail.
        let tail = s0.data.slice(2, 1);
        tail_tx.send(Some(tail)).unwrap();
        let s1 = ready_rx.recv().unwrap().unwrap();
        assert_eq!(s1.tail_size, 1);
        assert_eq!(batch_values(&s1.data), vec![3, 4, 5, 6]);
    }

    // Error propagation through the stitcher cannot be triggered from the Python test suite.
    #[test]
    fn test_stitcher_propagates_error_from_buffer_rx() {
        let (buffer_tx, buffer_rx) = bounded(4);
        let (_tail_tx, tail_rx) = bounded(4);
        let (ready_tx, ready_rx) = bounded(4);
        let schema = schema();

        // The stitcher receives from buffer_rx first; it forwards the error and returns
        // before ever touching tail_rx, so no bootstrap signal is needed here.
        buffer_tx.send(Err(Error::IterationNotStarted)).unwrap();
        drop(buffer_tx);

        thread::spawn(move || buffer_stitcher(&buffer_rx, &tail_rx, &ready_tx, &schema));

        assert!(ready_rx.recv().unwrap().is_err());
    }
}
