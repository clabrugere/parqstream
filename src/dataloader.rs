use std::sync::Arc;
use std::thread;

use arrow::record_batch::RecordBatch;
use crossbeam_channel::{bounded, Receiver};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

use crate::batch::Batch;
use crate::buffer::Buffer;
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::pipeline::{chunk_feeder, collector, read_feeder, Chunk};

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
    seed: Option<u64>,
    // iteration state
    epoch_counter: usize,
    buffer: Option<Buffer>,
    steps_remaining: Option<usize>,
}

impl DataLoader {
    /// Spawn the feeder and worker threads and return the batch receiver
    fn spawn_pipeline(&self, seed: Option<u64>) -> Receiver<Result<RecordBatch>> {
        let dataset = self.dataset.clone();
        let buffer_size = self
            .buffer_size
            .unwrap_or(dataset.total_rows)
            .max(self.batch_size);
        let chunk_size = (buffer_size + self.num_workers - 1).div_ceil(self.num_workers);
        let shuffle = self.shuffle;

        // Pre-compute row group layout for feeder (avoids capturing dataset Arc)
        let row_groups = dataset
            .row_group_index
            .iter()
            .map(|m| (m.row_offset, m.num_rows))
            .collect::<Vec<_>>();

        let (chunk_tx, chunk_rx) = bounded::<Chunk>(self.num_workers * 2);
        let (data_tx, data_rx) = bounded::<Result<RecordBatch>>(self.num_workers + 2);
        let (buffer_tx, buffer_rx) = bounded::<Result<RecordBatch>>(self.prefetch_factor);

        // chunk feeder sending row group read tasks to workers
        thread::spawn(move || {
            chunk_feeder(&chunk_tx, &row_groups, chunk_size, shuffle, seed);
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
        if let Some(num_steps) = num_steps {
            if num_steps == 0 {
                return Err(PyValueError::new_err("num_steps must be > 0"));
            }
        }
        if num_workers == 0 {
            return Err(PyValueError::new_err("num_workers must be > 0"));
        }
        if prefetch_factor == 0 {
            return Err(PyValueError::new_err("prefetch_factor must be > 0"));
        }
        if let Some(buffer_size) = buffer_size {
            if buffer_size < batch_size {
                return Err(PyValueError::new_err("buffer_size must be >= batch_size"));
            }
        }

        let available_cores = thread::available_parallelism()
            .map_err(|e| Error::ThreadDetermination(e.to_string()))?
            .get();
        let num_workers = num_workers.min(available_cores).max(1);

        Ok(Self {
            dataset: Arc::new(dataset.clone()),
            batch_size,
            num_steps,
            num_workers,
            prefetch_factor,
            shuffle,
            buffer_size,
            epoch_counter: 0,
            buffer: None,
            steps_remaining: None,
            seed,
        })
    }

    /// Start (or restart) the prefetch pipeline and return `self`.
    pub fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        let seed = slf.seed.map(|s| s + slf.epoch_counter as u64);
        let buffer_rx = slf.spawn_pipeline(seed);
        slf.buffer = Some(Buffer::new(buffer_rx, slf.shuffle, seed));
        slf.steps_remaining = slf.num_steps;
        slf.epoch_counter += 1;
        slf
    }

    /// Return the next `Batch`, or raise `StopIteration` when the pipeline is exhausted.
    pub fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Batch> {
        let batch_size = slf.batch_size;
        if slf.steps_remaining == Some(0) {
            return Err(PyStopIteration::new_err("dataloader consumed"));
        }
        let Some(buffer) = slf.buffer.as_mut() else {
            return Err(PyStopIteration::new_err("dataloader not initialized"));
        };
        match buffer.take(batch_size, py) {
            Ok(Some(batch)) => {
                if let Some(steps_remaining) = slf.steps_remaining.as_mut() {
                    *steps_remaining -= 1;
                }
                Ok(Batch::new(batch))
            }
            Ok(None) => Err(PyStopIteration::new_err("dataloader consumed")),
            Err(e) => Err(PyRuntimeError::new_err(format!("worker error: {e}"))),
        }
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
