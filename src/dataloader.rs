use std::sync::Arc;
use std::thread;

use arrow::record_batch::RecordBatch;
use crossbeam_channel::{bounded, Receiver};
use pyo3::prelude::*;

use crate::batch::Batch;
use crate::buffer::Buffer;
use crate::checkpoint::{Checkpoint, CheckpointCursor};
use crate::dataset::Dataset;
use crate::distributed::DistributedConfig;
use crate::error::{Error, Result};
use crate::pipeline::{buffer_builder, chunk_reader, job_dispatcher, Job};

#[derive(Debug, Copy, Clone)]
pub struct DataLoaderConfig {
    batch_size: usize,
    num_steps: Option<usize>,
    shuffle: bool,
    num_workers: usize,
    prefetch_factor: usize,
    buffer_size: Option<usize>,
    seed: u64,
}

impl DataLoaderConfig {
    fn new(
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
        if buffer_size.is_some_and(|buffer_size| buffer_size < batch_size) {
            return Err(Error::InvalidBufferSize(buffer_size.unwrap()));
        }
        let available_cores = thread::available_parallelism()
            .map_err(Error::ParallelismUnavailable)?
            .get();
        Ok(Self {
            batch_size,
            num_steps,
            shuffle,
            num_workers: num_workers.min(available_cores),
            prefetch_factor,
            buffer_size,
            seed: seed.unwrap_or_else(rand::random),
        })
    }

    // Falls back to the full epoch size, then clamps up to batch_size so a fill is never smaller than one batch.
    fn resolve_buffer_size(&self, rank_local_epoch_size: usize) -> usize {
        self.buffer_size
            .unwrap_or(rank_local_epoch_size)
            .max(self.batch_size)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ShuffleConfig {
    pub shuffle: bool,
    pub seed: u64,
}

/// Stores the state of a [`DataLoader`], which can be serialized to a [`Checkpoint`] for saving and resuming later
#[derive(Debug, Default)]
pub struct DataLoaderState {
    pub iteration_seed: u64,
    pub buffer: Option<Buffer>,
    pub steps_remaining: Option<usize>,
    pub epoch: usize,
    pub rows_yielded: usize,
    pub rows_epoch_start: usize, // 0 for fresh runs, cursor.rows_epoch_start on resume
}

/// Dataloader with prefetching.
///
/// Calling `__iter__` (or iterating with a for loop) starts:
/// 1. a thread that emits row-group metadata read tasks and an associated result channel to a job channel ([`job_dispatcher`])
/// 2. `num_workers` threads that receive jobs, load chunks from disk and send results to the job's result channel ([`chunk_reader`])
/// 3. a thread that accumulates chunks pulled from the pending receiver FIFO queue until it has >= `buffer_size` rows, concatenates
///    and optionally shuffles them, and sends the assembled buffer ([`buffer_builder`])
/// 4. the main thread receives assembled buffers, stitches any unconsumed tail rows to the front,
///    and slices into `batch_size` batches; the GIL is released for the entire operation
///
/// Up to `prefetch_factor` assembled buffers can be queued between stages 3 and 4.
///
/// If set, the iterator runs for exactly `num_steps` batches, then raises `StopIteration`.
/// Dropping or garbage-collecting the [`DataLoader`] signals the background threads to stop early.
#[pyclass]
pub struct DataLoader {
    dataset: Arc<Dataset>,
    config: DataLoaderConfig,
    dist_config: DistributedConfig,
    state: DataLoaderState,
    checkpoint: Option<Checkpoint>,
}

impl DataLoader {
    /// Per-epoch seed: varies each epoch so row-group and buffer shuffles differ across epochs.
    fn epoch_seed(&self) -> u64 {
        self.config.seed + self.state.epoch as u64
    }

    /// Spawn the pipeline threads and return the receiving end of the assembled-buffer channel.
    fn spawn_pipeline(
        &self,
        shuffle_config: ShuffleConfig,
        cursor: &CheckpointCursor,
    ) -> Receiver<Result<RecordBatch>> {
        let dataset = self.dataset.clone();
        // Default to the rank-local epoch size so a single buffer fill never spans two epochs.
        // For world_size=1 this equals dataset.total_rows (same as before).
        let rank_local_epoch_size =
            self.dist_config
                .epoch_row_count(shuffle_config, 0, &dataset.row_group_index);
        let buffer_size = self.config.resolve_buffer_size(rank_local_epoch_size);
        let chunk_size = buffer_size.div_ceil(self.config.num_workers);

        // Pre-compute row group layout for task dispatcher so it doesn't have to query the dataset while emitting tasks.
        // The row group index is already in memory, so this is cheap.
        let row_group_lengths = dataset
            .row_group_index
            .iter()
            .map(|m| m.num_rows)
            .collect::<Vec<_>>();

        // Declare all the channels
        let (job_tx, job_rx) = bounded::<Job>(self.config.num_workers * 2);
        let (pending_tx, pending_rx) =
            bounded::<Receiver<Result<RecordBatch>>>(self.config.num_workers * 2);
        let (buffer_tx, buffer_rx) = bounded::<Result<RecordBatch>>(self.config.prefetch_factor);

        // Prepare and sends jobs to workers
        let dist_config = self.dist_config;
        let position = cursor.stream_pos;
        thread::spawn(move || {
            job_dispatcher(
                &job_tx,
                &pending_tx,
                &row_group_lengths,
                chunk_size,
                position,
                shuffle_config,
                dist_config,
            );
        });

        // Workers read a contiguous chunk from a row group
        for _ in 0..self.config.num_workers {
            let job_rx = job_rx.clone();
            let dataset = dataset.clone();
            thread::spawn(move || chunk_reader(&job_rx, &dataset));
        }
        drop(job_rx); // original instance wasn't moved into a worker; drop it so workers see channel closure and can exit

        // Collect chunks until > buffer_size rows, then concatenate and send to the buffer builder.
        let schema = dataset.projected_schema.clone();
        let refill_count = cursor.buffer_pos.refill_count;
        thread::spawn(move || {
            buffer_builder(
                &pending_rx,
                &buffer_tx,
                &schema,
                buffer_size,
                shuffle_config,
                refill_count,
            );
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
        shuffle,
        num_workers,
        prefetch_factor,
        buffer_size,
        seed,
        rank,
        world_size,
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
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        let num_row_groups = dataset.row_group_index.len();
        if world_size > 1 && num_row_groups < world_size {
            return Err(Error::WorldSizeTooLarge {
                world_size,
                num_row_groups,
            });
        }

        Ok(Self {
            dataset: Arc::new(dataset.clone()),
            config: DataLoaderConfig::new(
                batch_size,
                num_steps,
                shuffle,
                num_workers,
                prefetch_factor,
                buffer_size,
                seed,
            )?,
            dist_config: DistributedConfig::new(rank, world_size)?,
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
            None => (
                slf.epoch_seed(),
                slf.config.num_steps,
                CheckpointCursor::default(),
            ),
        };
        let shuffle_config = ShuffleConfig {
            shuffle: slf.config.shuffle,
            seed,
        };
        let buffer_rx = slf.spawn_pipeline(shuffle_config, &cursor);
        let buffer = Buffer::new(buffer_rx, cursor.buffer_pos);

        slf.state = DataLoaderState {
            epoch: slf.state.epoch,
            iteration_seed: seed,
            buffer: Some(buffer),
            steps_remaining,
            rows_yielded: 0,
            rows_epoch_start: cursor.rows_epoch_start,
        };

        slf
    }

    /// Return the next [`Batch`], or raise `StopIteration` when the pipeline is exhausted.
    pub fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Result<Batch> {
        let batch_size = slf.config.batch_size;
        let state = &mut slf.state;

        let Some(buffer) = state.buffer.as_mut() else {
            return Err(Error::IterationNotStarted);
        };
        if state.steps_remaining == Some(0) {
            return Err(Error::DataLoaderConsumed);
        }
        match py.detach(|| buffer.take(batch_size)) {
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

        let shuffle_config = ShuffleConfig {
            shuffle: self.config.shuffle,
            seed: self.state.iteration_seed,
        };

        Ok(Checkpoint::from_state(
            &self.state,
            shuffle_config,
            self.dist_config,
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
        if checkpoint.dist_config.rank != self.dist_config.rank
            || checkpoint.dist_config.world_size != self.dist_config.world_size
        {
            return Err(Error::DistributedMismatch {
                checkpoint: checkpoint.dist_config,
                current: self.dist_config,
            });
        }
        self.state.epoch = checkpoint.epoch;
        self.checkpoint = Some(checkpoint);
        Ok(())
    }

    pub fn __len__(&self) -> Result<usize> {
        self.config.num_steps.ok_or_else(|| Error::UndefinedLength)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DataLoader(rows={}, columns={:?}, batch_size={}, num_steps={:?}, shuffle={}, seed={}, num_workers={}, prefetch_factor={}, buffer_size={:?}, rank={}, world_size={})",
            self.dataset.total_rows,
            self.dataset.columns,
            self.config.batch_size,
            self.config.num_steps,
            self.config.shuffle,
            self.config.seed,
            self.config.num_workers,
            self.config.prefetch_factor,
            self.config.buffer_size,
            self.dist_config.rank,
            self.dist_config.world_size,
        )
    }
}
