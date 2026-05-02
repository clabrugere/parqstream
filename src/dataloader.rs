use std::sync::Arc;
use std::thread;

use arrow::record_batch::RecordBatch;
use crossbeam_channel::{bounded, Receiver};
use pyo3::prelude::*;

use crate::batch::Batch;
use crate::buffer::Buffer;
use crate::checkpoint::{Checkpoint, Cursor};
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::pipeline::{chunk_collector, chunk_dispatcher, chunk_reader, Chunk, EpochCursor};

/// Stores the state of a Dataloader, which can be serialized to a Checkpoint for saving and resuming later
#[derive(Debug, Default)]
pub struct DataLoaderState {
    pub iteration_seed: u64,
    pub buffer: Option<Buffer>,
    pub steps_remaining: Option<usize>,
    pub epoch: usize,
    pub rows_yielded: usize,
    pub rows_epoch_start: usize, // 0 for fresh runs, cursor.rows_epoch_start on resume
}

impl DataLoaderState {
    /// Start a new iteration with the given seed, buffer, and `steps_remaining`, and reset `rows_yielded`.
    pub fn new_iteration(
        &mut self,
        seed: u64,
        buffer: Buffer,
        steps_remaining: Option<usize>,
        rows_epoch_start: usize,
    ) {
        self.iteration_seed = seed;
        self.buffer = Some(buffer);
        self.steps_remaining = steps_remaining;
        self.rows_yielded = 0;
        self.rows_epoch_start = rows_epoch_start;
    }
}

/// Dataloader with prefetching.
///
/// Calling `__iter__` (or iterating with a for loop) starts:
/// 1. a thread that emits row-group metadata chunks to a reader channel (`chunk_dispatcher`)
/// 2. `num_workers` threads that receive metadata from the chunk dispatcher, load them from disk and send them to a data channel (`chunk_reader`)
/// 3. a thread that collects chunks from the data channel until it has > `buffer_size` rows, concatenates them into a fill, and sends it to a fill channel (`chunk_collector`)
/// 4. the main thread reads fills from the fill channel via a `Buffer`, which optionally shuffles each fill
///    and slices it into `batch_size` batches; any unconsumed rows from the previous fill are prepended so
///    batches never straddle a fill boundary with a gap
///
/// Up to `prefetch_factor` fills can be buffered in the channel, and the GIL is released while waiting
/// to allow the background threads to run.
///
/// The iterator runs for exactly `num_steps` batches, then raises `StopIteration`.
/// Dropping or garbage-collecting the `DataLoader` signals the background threads to stop early.
#[pyclass]
pub struct DataLoader {
    dataset: Arc<Dataset>,
    batch_size: usize,
    num_steps: Option<usize>,
    shuffle: bool,
    num_workers: usize,
    prefetch_factor: usize,
    buffer_size: Option<usize>,
    seed: u64,
    state: DataLoaderState,
    checkpoint: Option<Checkpoint>,
}

impl DataLoader {
    /// Per-epoch seed: varies each epoch so row-group and buffer shuffles differ across epochs.
    fn epoch_seed(&self) -> u64 {
        self.seed + self.state.epoch as u64
    }

    /// Spawn the feeder and worker threads and return the batch receiver
    fn spawn_pipeline(&self, seed: u64, cursor: &Cursor) -> Receiver<Result<RecordBatch>> {
        let dataset = self.dataset.clone();
        let buffer_size = self
            .buffer_size
            .unwrap_or(dataset.total_rows)
            .max(self.batch_size);
        let chunk_size = (buffer_size + self.num_workers - 1).div_ceil(self.num_workers);
        let shuffle = self.shuffle;

        // Pre-compute row group layout for feeder
        let row_group_lengths = dataset
            .row_group_index
            .iter()
            .map(|m| m.num_rows)
            .collect::<Vec<_>>();

        let (chunk_tx, chunk_rx) = bounded::<Chunk>(self.num_workers * 2);
        let (batch_tx, batch_rx) = bounded::<Result<RecordBatch>>(self.num_workers + 2);
        let (prefetch_tx, prefetch_rx) = bounded::<Result<RecordBatch>>(self.prefetch_factor);

        // chunk feeder sending row group read tasks to workers
        let cursor = EpochCursor::from(cursor);
        thread::spawn(move || {
            chunk_dispatcher(
                &chunk_tx,
                &row_group_lengths,
                chunk_size,
                shuffle,
                seed,
                cursor,
            );
        });

        // workers read a contiguous chunk from a row group
        for _ in 0..self.num_workers {
            let chunk_rx = chunk_rx.clone();
            let data_tx = batch_tx.clone();
            let dataset = dataset.clone();
            thread::spawn(move || chunk_reader(&chunk_rx, &data_tx, &dataset));
        }
        drop(batch_tx);

        // collect chunks until > buffer_size rows, then concatenate and send to batch buffer
        let schema = dataset.projected_schema.clone();
        thread::spawn(move || chunk_collector(&batch_rx, &prefetch_tx, &schema, buffer_size));

        prefetch_rx
    }
}

#[pymethods]
impl DataLoader {
    #[new]
    #[pyo3(signature = (
        dataset,
        batch_size,
        num_steps = None,
        shuffle = false,
        num_workers = 4,
        prefetch_factor = 1,
        buffer_size = None,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: &Dataset,
        batch_size: usize,
        num_steps: Option<usize>,
        shuffle: bool,
        num_workers: usize,
        prefetch_factor: usize,
        buffer_size: Option<usize>,
        seed: Option<u64>,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::InvalidBatchSize(batch_size));
        }
        if num_steps == Some(0) {
            return Err(Error::InvalidNumSteps(0));
        }
        if num_workers == 0 {
            return Err(Error::InvalidNumWorkers(0));
        }
        if prefetch_factor == 0 {
            return Err(Error::InvalidPrefetchFactor(0));
        }
        if buffer_size.is_some_and(|bs| bs < batch_size) {
            return Err(Error::InvalidBufferSize(buffer_size.unwrap()));
        }

        let available_cores = thread::available_parallelism()
            .map_err(Error::ParallelismUnavailable)?
            .get();
        let seed = seed.unwrap_or_else(rand::random);
        Ok(Self {
            dataset: Arc::new(dataset.clone()),
            batch_size,
            num_steps,
            num_workers: num_workers.min(available_cores).max(1),
            prefetch_factor,
            shuffle,
            buffer_size,
            seed,
            state: DataLoaderState::default(),
            checkpoint: None,
        })
    }

    /// Start (or restart) the prefetch pipeline and return `self`.
    pub fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.state.epoch += 1;

        // On checkpoint resume, use the saved seed/cursor/steps_remaining so the run continues
        // from exactly where it stopped. Otherwise derive a fresh per-epoch seed.
        let (seed, steps_remaining, cursor) = match slf.checkpoint.take() {
            Some(checkpoint) => (
                checkpoint.seed,
                checkpoint.steps_remaining,
                checkpoint.cursor,
            ),
            None => (slf.epoch_seed(), slf.num_steps, Cursor::default()),
        };

        let prefetch_rx = slf.spawn_pipeline(seed, &cursor);
        let buffer = Buffer::new(
            prefetch_rx,
            slf.shuffle,
            seed,
            cursor.buffer_seed_offset,
            cursor.buffer_offset,
        );

        slf.state
            .new_iteration(seed, buffer, steps_remaining, cursor.rows_epoch_start);

        slf
    }

    /// Return the next `Batch`, or raise `StopIteration` when the pipeline is exhausted.
    pub fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Result<Batch> {
        let batch_size = slf.batch_size;
        let state = &mut slf.state;

        let Some(buffer) = state.buffer.as_mut() else {
            return Err(Error::IterationNotStarted);
        };
        if state.steps_remaining == Some(0) {
            return Err(Error::DataLoaderConsumed);
        }
        match buffer.take(batch_size, py) {
            Ok(Some(batch)) => {
                if let Some(steps_remaining) = state.steps_remaining.as_mut() {
                    *steps_remaining -= 1;
                }
                state.rows_yielded += batch.num_rows();
                Ok(Batch::new(batch))
            }
            Ok(None) => Err(Error::DataLoaderConsumed),
            Err(e) => Err(e),
        }
    }

    /// Returns the current position as a [`Checkpoint`]. Errors if the iterator has not been started.
    pub fn checkpoint(&self) -> Result<Checkpoint> {
        // check if __iter__ has been called at least once
        if self.state.buffer.is_none() {
            return Err(Error::NoStateToCheckpoint);
        }
        // if a checkpoint is already stored and hasn't been consumed by __iter__, return it directly
        if let Some(checkpoint) = &self.checkpoint {
            return Ok(checkpoint.clone());
        }

        Ok(Checkpoint::from_state(
            &self.state,
            self.shuffle,
            self.dataset.identifier,
            &self.dataset.row_group_index,
        ))
    }

    /// Restore a previously saved checkpoint.
    ///
    /// The checkpoint's `steps_remaining` overrides the loader's `num_steps` for
    /// the resumed iteration; this is intentional so that the run ends at exactly
    /// the same total number of batches as the original run.
    pub fn load_checkpoint(&mut self, checkpoint: Checkpoint) -> Result<()> {
        if checkpoint.dataset_identifier != self.dataset.identifier {
            return Err(Error::DatasetMismatch {
                checkpoint: checkpoint.dataset_identifier,
                current: self.dataset.identifier,
            });
        }
        self.state.epoch = checkpoint.epoch;
        self.checkpoint = Some(checkpoint);
        Ok(())
    }

    pub fn __len__(&self) -> Result<usize> {
        self.num_steps.ok_or_else(|| Error::UndefinedLength)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DataLoader(rows={}, columns={:?}, batch_size={}, num_steps={:?}, num_workers={}, prefetch_factor={}, buffer_size={:?})",
            self.dataset.total_rows,
            self.dataset.columns,
            self.batch_size,
            self.num_steps,
            self.num_workers,
            self.prefetch_factor,
            self.buffer_size,
        )
    }
}
