use std::sync::Arc;
use std::thread;

use arrow::record_batch::RecordBatch;
use crossbeam_channel::{bounded, Receiver, Sender};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use rand::prelude::*;

use crate::batch::Batch;
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::reader::read_batch;

/// Rows are yielded in file order, wrapping at `total_rows`
fn sequential_feeder(
    index_tx: Sender<Vec<usize>>,
    total_rows: usize,
    batch_size: usize,
    num_steps: usize,
) {
    for step in 0..num_steps {
        let indices = (0..batch_size)
            .map(|i| (step * batch_size + i) % total_rows)
            .collect();
        if index_tx.send(indices).is_err() {
            break;
        }
    }
}

/// Row groups are shuffled at the start of each epoch and rows are yielded sequentially
/// within each row group. Each batch touches at most 2 row groups
fn shuffle_feeder(
    index_tx: Sender<Vec<usize>>,
    row_groups: Vec<(usize, usize)>,
    batch_size: usize,
    num_steps: usize,
) {
    let mut rng = rand::rng();
    let num_row_groups = row_groups.len();

    let mut epoch_order = (0..num_row_groups).collect::<Vec<_>>();
    epoch_order.shuffle(&mut rng);
    let mut row_group_cursor = 0;
    let mut within_cursor = 0;

    for _ in 0..num_steps {
        let mut indices = Vec::with_capacity(batch_size);

        while indices.len() < batch_size {
            if row_group_cursor >= num_row_groups {
                // Start of new epoch: re-shuffle row group order
                epoch_order.shuffle(&mut rng);
                row_group_cursor = 0;
                within_cursor = 0;
            }

            let row_group_idx = epoch_order[row_group_cursor];
            let (row_offset, num_rows) = row_groups[row_group_idx];
            let available = num_rows - within_cursor;
            let need = batch_size - indices.len();
            let take = need.min(available);

            indices.extend((within_cursor..within_cursor + take).map(|i| row_offset + i));
            within_cursor += take;

            if within_cursor >= num_rows {
                row_group_cursor += 1;
                within_cursor = 0;
            }
        }

        if index_tx.send(indices).is_err() {
            break;
        }
    }
}

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

        // Pre-compute row group layout so the feeder closure doesn't need to capture `dataset`
        let row_groups: Vec<(usize, usize)> = dataset
            .row_group_index
            .iter()
            .map(|m| (m.row_offset, m.num_rows))
            .collect();

        // Generates batch indices and sends them to workers
        thread::spawn(move || {
            if shuffle {
                shuffle_feeder(index_tx, row_groups, batch_size, num_steps);
            } else {
                sequential_feeder(index_tx, total_rows, batch_size, num_steps);
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
