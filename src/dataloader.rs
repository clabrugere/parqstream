use std::sync::Arc;
use std::thread;

use arrow::record_batch::RecordBatch;
use crossbeam_channel::{bounded, Receiver};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use rand::prelude::*;

use crate::batch::Batch;
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::reader::read_batch;

/// Dataloader with prefetching.
///
/// Calling `__iter__` (or iterating with a for loop) starts a pipeline of background threads that
/// prefetch `prefetch_factor` batches (sampled uniformly with replacement) ahead of the Python consumer.
/// The GIL is released while waiting for the next batch.
///
/// The iterator runs for exactly `num_steps` batches, then raises `StopIteration`.
/// Dropping or garbage-collecting the `DataLoader` signals the background threads to stop early.
#[pyclass]
pub struct DataLoader {
    dataset: Arc<Dataset>,
    batch_size: usize,
    num_steps: usize,
    shuffle: bool,
    num_workers: usize,
    prefetch_factor: usize,
    // Set when iteration starts, cleared on exhaustion
    batch_rx: Option<Receiver<Result<RecordBatch>>>,
}

impl DataLoader {
    /// Spawn the feeder and worker threads and return the batch receiver
    fn spawn_pipeline(&self) -> Receiver<Result<RecordBatch>> {
        let dataset = self.dataset.clone();
        let batch_size = self.batch_size;
        let num_steps = self.num_steps;
        let shuffle = self.shuffle;
        let num_workers = self.num_workers;
        let total_rows = self.dataset.total_rows;

        // Index channel: feeder -> workers
        let (index_tx, index_rx) = bounded::<Vec<usize>>(num_workers * 2);
        // Batch channel: workers -> Python
        let (batch_tx, batch_rx) = bounded::<Result<RecordBatch>>(self.prefetch_factor);

        // Generates index batches and feeds them to workers
        thread::spawn(move || {
            let mut rng = rand::rng();
            for step in 0..num_steps {
                let indices = if shuffle {
                    (0..batch_size)
                        .map(|_| rng.random_range(0..total_rows))
                        .collect()
                } else {
                    (0..batch_size)
                        .map(|i| (step * batch_size + i) % total_rows)
                        .collect()
                };
                if index_tx.send(indices).is_err() {
                    break; // workers have stopped
                }
            }
            // Dropping index_tx signals workers to exit
        });

        // Each worker pulls an index batch from the channel, reads from disk, and pushes the result batch
        for _ in 0..num_workers {
            let index_rx = index_rx.clone();
            let batch_tx = batch_tx.clone();
            let dataset = dataset.clone();

            thread::spawn(move || {
                for indices in &index_rx {
                    match read_batch(&dataset, &indices) {
                        Ok(batch) => {
                            if batch_tx.send(Ok(batch)).is_err() {
                                break; // consumer dropped
                            }
                        }
                        Err(e) => {
                            let _ = batch_tx.send(Err(Error::WorkerThread(e.to_string())));
                            break; // signal error and exit
                        }
                    }
                }
                // Dropping batch_tx clone; when all workers drop it the channel closes
            });
        }
        // Drop the main batch_tx so the channel closes when all workers finish
        drop(batch_tx);

        batch_rx
    }
}

#[pymethods]
impl DataLoader {
    #[new]
    #[pyo3(signature = (
        dataset,
        batch_size,
        num_steps,
        shuffle = false,
        num_workers = 4,
        prefetch_factor = 4,
    ))]
    pub fn py_new(
        dataset: &Dataset,
        batch_size: usize,
        num_steps: usize,
        shuffle: bool,
        num_workers: usize,
        prefetch_factor: usize,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }
        if num_steps == 0 {
            return Err(PyValueError::new_err("num_steps must be > 0"));
        }
        if num_workers == 0 {
            return Err(PyValueError::new_err("num_workers must be > 0"));
        }
        if prefetch_factor == 0 {
            return Err(PyValueError::new_err("prefetch_factor must be > 0"));
        }

        let count = thread::available_parallelism()
            .map_err(|e| Error::ThreadDetermination(e.to_string()))?
            .get();
        let num_workers = num_workers.min(count).max(1);

        Ok(Self {
            dataset: Arc::new(dataset.clone()),
            batch_size,
            num_steps,
            num_workers,
            prefetch_factor,
            shuffle,
            batch_rx: None,
        })
    }

    /// Start (or restart) the prefetch pipeline and return `self`.
    pub fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.batch_rx = Some(slf.spawn_pipeline());
        slf
    }

    /// Return the next `Batch`, or raise `StopIteration` when the pipeline is exhausted.
    pub fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Batch> {
        let Some(batch_rx) = slf.batch_rx.take() else {
            return Err(PyStopIteration::new_err("not started"));
        };

        // Release the GIL while waiting for the next batch.
        let result = py.detach(|| batch_rx.recv());

        match result {
            Ok(Ok(batch)) => {
                // Put the receiver back for the next iteration.
                slf.batch_rx = Some(batch_rx);
                Ok(Batch::new(batch))
            }
            Ok(Err(e)) => {
                // Worker thread reported an error
                slf.batch_rx = None;
                Err(PyRuntimeError::new_err(format!("worker error: {e}")))
            }
            Err(_) => {
                // Channel closed: iteration complete.
                slf.batch_rx = None;
                Err(PyStopIteration::new_err("dataloader consumed"))
            }
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DataLoader(rows={}, columns={:?}, batch_size={}, num_steps={}, num_workers={}, prefetch_factor={})",
            self.dataset.total_rows,
            self.dataset.columns,
            self.batch_size,
            self.num_steps,
            self.num_workers,
            self.prefetch_factor,
        )
    }
}
