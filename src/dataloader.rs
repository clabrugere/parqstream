use std::sync::Arc;
use std::thread;

use arrow::compute::concat_batches;
use arrow::record_batch::RecordBatch;
use crossbeam_channel::{bounded, Receiver, Sender};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use rand::prelude::*;

use crate::batch::Batch;
use crate::buffer::Buffer;
use crate::dataset::Dataset;
use crate::error::{Error, Result};

// thread that continuously sends row group read tasks to the chunk channel, shuffled if needed
fn chunk_feeder(
    chunk_tx: &Sender<(usize, usize, usize)>,
    row_groups: &[(usize, usize)],
    chunk_size: usize,
    shuffle: bool,
) {
    let mut rng = rand::rng();
    let num_groups = row_groups.len();

    let mut order = (0..num_groups).collect::<Vec<_>>();
    if shuffle {
        order.shuffle(&mut rng);
    }

    let mut row_group_offset = 0;
    let mut intra_row_group_offset = 0;

    loop {
        // new epoch , re-shuffle if needed
        if row_group_offset >= num_groups {
            if shuffle {
                order.shuffle(&mut rng);
            }
            row_group_offset = 0;
            intra_row_group_offset = 0;
        }
        let row_group_idx = order[row_group_offset];
        let (_, num_rows) = row_groups[row_group_idx];
        let len = chunk_size.min(num_rows - intra_row_group_offset);

        if chunk_tx
            .send((row_group_idx, intra_row_group_offset, len))
            .is_err()
        {
            break; // consumer dropped
        }
        intra_row_group_offset += len;
        if intra_row_group_offset >= num_rows {
            row_group_offset += 1;
            intra_row_group_offset = 0;
        }
    }
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
    num_steps: usize,
    shuffle: bool,
    num_workers: usize,
    prefetch_factor: usize,
    buffer_size: Option<usize>,
    // iteration state
    buffer: Option<Buffer>,
    steps_remaining: usize,
}

impl DataLoader {
    /// Spawn the feeder and worker threads and return the batch receiver
    fn spawn_pipeline(&self) -> Receiver<Result<RecordBatch>> {
        let dataset = self.dataset.clone();
        let buffer_size = self
            .buffer_size
            .unwrap_or(dataset.total_rows)
            .max(self.batch_size);
        let chunk_size = (buffer_size + self.num_workers - 1).div_ceil(self.num_workers);
        let shuffle = self.shuffle;

        // Pre-compute row group layout for feeder (avoids capturing dataset Arc)
        let row_groups: Vec<(usize, usize)> = dataset
            .row_group_index
            .iter()
            .map(|m| (m.row_offset, m.num_rows))
            .collect();

        let (chunk_tx, chunk_rx) = bounded::<(usize, usize, usize)>(self.num_workers * 2);
        let (data_tx, data_rx) = bounded::<Result<RecordBatch>>(self.num_workers + 2);
        let (buffer_tx, buffer_rx) = bounded::<Result<RecordBatch>>(self.prefetch_factor);

        // chunk feeder sending row group read tasks to workers
        thread::spawn(move || chunk_feeder(&chunk_tx, &row_groups, chunk_size, shuffle));

        // workers read a contiguous chunk from a row group
        for _ in 0..self.num_workers {
            let chunk_rx = chunk_rx.clone();
            let data_tx = data_tx.clone();
            let dataset = dataset.clone();
            thread::spawn(move || {
                for (rg_meta_idx, start, len) in &chunk_rx {
                    let meta = &dataset.row_group_index[rg_meta_idx];
                    match dataset.read_row_group_range(
                        meta.file_idx,
                        meta.row_group_idx,
                        start,
                        len,
                    ) {
                        Ok(b) => {
                            if data_tx.send(Ok(b)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = data_tx.send(Err(e));
                            break;
                        }
                    }
                }
            });
        }
        drop(data_tx);

        // collect chunks until > buffer_size rows, then concatenate and send to batch buffer
        let schema = dataset.projected_schema.clone();
        thread::spawn(move || loop {
            let mut parts: Vec<RecordBatch> = Vec::new();
            let mut rows = 0usize;
            while rows < buffer_size {
                match data_rx.recv() {
                    Ok(Ok(chunk)) => {
                        rows += chunk.num_rows();
                        parts.push(chunk);
                    }
                    Ok(Err(e)) => {
                        let _ = buffer_tx.send(Err(e));
                        return;
                    }
                    Err(_) => return,
                }
            }
            let buffer = match concat_batches(&schema, &parts) {
                Ok(b) => b,
                Err(e) => {
                    let _ = buffer_tx.send(Err(e.into()));
                    return;
                }
            };
            if buffer_tx.send(Ok(buffer)).is_err() {
                return;
            }
        });

        buffer_rx
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
        prefetch_factor = 1,
        buffer_size = None,
    ))]
    pub fn py_new(
        dataset: &Dataset,
        batch_size: usize,
        num_steps: usize,
        shuffle: bool,
        num_workers: usize,
        prefetch_factor: usize,
        buffer_size: Option<usize>,
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
        if let Some(buffer_size) = buffer_size {
            if buffer_size < batch_size {
                return Err(PyValueError::new_err("buffer_size must be >= batch_size"));
            }
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
            buffer_size,
            buffer: None,
            steps_remaining: 0,
        })
    }

    /// Start (or restart) the prefetch pipeline and return `self`.
    pub fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.buffer = Some(Buffer::new(slf.spawn_pipeline(), slf.shuffle));
        slf.steps_remaining = slf.num_steps;
        slf
    }

    /// Return the next `Batch`, or raise `StopIteration` when the pipeline is exhausted.
    pub fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Batch> {
        let batch_size = slf.batch_size;
        if slf.steps_remaining == 0 {
            return Err(PyStopIteration::new_err("dataloader consumed"));
        }
        let Some(buffer) = slf.buffer.as_mut() else {
            return Err(PyStopIteration::new_err("dataloader not initialized"));
        };
        match buffer.take(batch_size, py) {
            Ok(Some(batch)) => {
                slf.steps_remaining -= 1;
                Ok(Batch::new(batch))
            }
            Ok(None) => Err(PyStopIteration::new_err("dataloader consumed")),
            Err(e) => Err(PyRuntimeError::new_err(format!("worker error: {e}"))),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DataLoader(rows={}, columns={:?}, batch_size={}, num_steps={}, num_workers={}, prefetch_factor={}, buffer_size={:?})",
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
