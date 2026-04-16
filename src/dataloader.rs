use std::sync::Arc;
use std::thread;

use arrow::record_batch::RecordBatch;
use crossbeam_channel::{bounded, Receiver};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

use crate::batch::Batch;
use crate::buffer::Buffer;
use crate::checkpoint::{Checkpoint, Cursor};
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::pipeline::{chunk_feeder, collector, read_feeder, Chunk};

/// Stores the state of a Dataloader, which can be serialized to a Checkpoint for saving and resuming later
#[derive(Debug, Default)]
pub struct DataLoaderState {
    pub buffer: Option<Buffer>,
    pub steps_remaining: Option<usize>,
    pub epoch_count: usize,
    pub rows_within_epoch: usize,
}

/// Dataloader with prefetching.
///
/// Calling `__iter__` (or iterating with a for loop) starts:
/// 1. a thread that emit chunks of rows groups to a chunk channel (feeder)
/// 2. `num_workers` threads that read chunks from the chunk channel, load them from disk and send to a data channel (workers)
/// 3. a thread that collects chunks from the data channel until it has > `buffer_size` rows, concatenates them and sends to a batch channel (collector)
///
/// The main thread consumes batches from the batch channel, yielding them to Python. Up to `prefetch_factor` batches can be buffered in the batch channel,
/// and the GIL is released while waiting for batches to allow the background threads to run.
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
        self.seed + self.state.epoch_count as u64
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
        let (data_tx, data_rx) = bounded::<Result<RecordBatch>>(self.num_workers + 2);
        let (buffer_tx, buffer_rx) = bounded::<Result<RecordBatch>>(self.prefetch_factor);

        // chunk feeder sending row group read tasks to workers
        let epoch_offset = cursor.epoch_offset;
        let row_group_offset = cursor.row_group_offset;
        let intra_row_group_offset = cursor.intra_row_group_offset;
        thread::spawn(move || {
            chunk_feeder(
                &chunk_tx,
                &row_group_lengths,
                chunk_size,
                shuffle,
                seed,
                epoch_offset,
                row_group_offset,
                intra_row_group_offset,
            );
        });

        // workers read a contiguous chunk from a row group
        for _ in 0..self.num_workers {
            let chunk_rx = chunk_rx.clone();
            let data_tx = data_tx.clone();
            let dataset = dataset.clone();
            thread::spawn(move || read_feeder(&chunk_rx, &data_tx, &dataset));
        }
        drop(data_tx);

        // collect chunks until > buffer_size rows, then concatenate and send to batch buffer
        let schema = dataset.projected_schema.clone();
        thread::spawn(move || collector(&data_rx, &buffer_tx, &schema, buffer_size));

        buffer_rx
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
    pub fn py_new(
        dataset: &Dataset,
        batch_size: usize,
        num_steps: Option<usize>,
        shuffle: bool,
        num_workers: usize,
        prefetch_factor: usize,
        buffer_size: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }
        if num_steps == Some(0) {
            return Err(PyValueError::new_err("num_steps must be > 0"));
        }
        if num_workers == 0 {
            return Err(PyValueError::new_err("num_workers must be > 0"));
        }
        if prefetch_factor == 0 {
            return Err(PyValueError::new_err("prefetch_factor must be > 0"));
        }
        if buffer_size.is_some_and(|bs| bs < batch_size) {
            return Err(PyValueError::new_err("buffer_size must be >= batch_size"));
        }

        let available_cores = thread::available_parallelism()
            .map_err(|e| Error::ThreadDetermination(e.to_string()))?
            .get();

        Ok(Self {
            dataset: Arc::new(dataset.clone()),
            batch_size,
            num_steps,
            num_workers: num_workers.min(available_cores).max(1),
            prefetch_factor,
            shuffle,
            buffer_size,
            seed: seed.unwrap_or_else(rand::random),
            state: DataLoaderState::default(),
            checkpoint: None,
        })
    }

    /// Start (or restart) the prefetch pipeline and return `self`.
    pub fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.state.epoch_count += 1;

        // On checkpoint resume, use the saved seed/cursor/steps_remaining so the run continues
        // from exactly where it stopped. Otherwise derive a fresh per-epoch seed.
        let (seed, steps_remaining, cursor) = match slf.checkpoint.take() {
            Some(checkpoint) => (
                checkpoint.seed,
                checkpoint.steps_remaining,
                checkpoint.cursor,
            ),
            None => (
                slf.epoch_seed(),
                slf.num_steps,
                Cursor::default(),
            ),
        };

        let buffer_rx = slf.spawn_pipeline(seed, &cursor);
        let buffer = Buffer::new(
            buffer_rx,
            slf.shuffle,
            seed,
            cursor.buffer_seed_offset,
            cursor.buffer_offset,
        );

        // update state for new iteration
        slf.state.buffer = Some(buffer);
        slf.state.steps_remaining = steps_remaining;
        slf.state.rows_within_epoch = 0;

        slf
    }

    /// Return the next `Batch`, or raise `StopIteration` when the pipeline is exhausted.
    pub fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Batch> {
        let batch_size = slf.batch_size;
        let state = &mut slf.state;

        let Some(buffer) = state.buffer.as_mut() else {
            return Err(PyRuntimeError::new_err(
                "iteration not started, call iter(dataloader) first",
            ));
        };
        if state.steps_remaining == Some(0) {
            return Err(PyStopIteration::new_err("DataLoader consumed"));
        }
        match buffer.take(batch_size, py) {
            Ok(Some(batch)) => {
                if let Some(steps_remaining) = state.steps_remaining.as_mut() {
                    *steps_remaining -= 1;
                }
                state.rows_within_epoch += batch.num_rows();
                Ok(Batch::new(batch))
            }
            Ok(None) => Err(PyStopIteration::new_err("DataLoader consumed")),
            Err(e) => Err(PyRuntimeError::new_err(format!("worker error: {e}"))),
        }
    }

    pub fn checkpoint(&self) -> PyResult<Checkpoint> {
        // check if __iter__ has been called at least once
        if self.state.buffer.is_none() {
            return Err(PyRuntimeError::new_err(
                "no state to serialize, call iter(dataloader) first",
            ));
        }
        // if a checkpoint is already stored and hasn't been consumed by __iter__, return it directly
        if let Some(checkpoint) = &self.checkpoint {
            return Ok(checkpoint.clone());
        }

        Ok(Checkpoint::from_state(
            &self.state,
            self.shuffle,
            self.epoch_seed(),
            self.dataset.identifier,
            &self.dataset.row_group_index,
        ))
    }

    /// Restore a previously saved checkpoint.
    ///
    /// The checkpoint's `steps_remaining` overrides the loader's `num_steps` for
    /// the resumed iteration; this is intentional so that the run ends at exactly
    /// the same total number of batches as the original run.
    pub fn load_checkpoint(&mut self, checkpoint: Checkpoint) -> PyResult<()> {
        if checkpoint.dataset_identifier != self.dataset.identifier {
            return Err(PyValueError::new_err(format!(
                "dataset identifier mismatch: checkpoint={:#x}, current={:#x}",
                checkpoint.dataset_identifier, self.dataset.identifier
            )));
        }
        self.state.epoch_count = checkpoint.epoch;
        self.checkpoint = Some(checkpoint);
        Ok(())
    }

    pub fn __len__(&self) -> PyResult<usize> {
        self.num_steps.ok_or_else(|| {
            PyValueError::new_err(
                "length is undefined for infinite DataLoader; set num_steps to a finite value to enable __len__",
            )
        })
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
