use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyType};

use crate::buffer::{Buffer, BufferSnapshot};
use crate::dataloader::{DataLoaderState, ShuffleConfig};
use crate::dataset::RowGroupMeta;
use crate::distributed::DistributedConfig;
use crate::error::{Error, Result};

fn pydict_get<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> Result<T>
where
    for<'a> T: FromPyObject<'a, 'py, Error = PyErr>,
{
    dict.get_item(key)?
        .ok_or_else(|| Error::MissingKeyInPyDict(key.into()))
        .and_then(|v| Ok(v.extract::<T>()?))
}

/// Walk the rank-local ordered row groups to find which one contains `rows`.
fn locate_row_in_order(
    epoch: usize,
    shuffle_config: ShuffleConfig,
    dist_config: DistributedConfig,
    row_group_index: &[RowGroupMeta],
    rows: usize,
) -> (usize, usize) {
    let order = dist_config.epoch_order(shuffle_config, epoch, row_group_index.len());

    let mut rows = rows;
    let mut row_group_pos = 0;
    let mut row_in_group = 0;
    // always terminates because rows < rank-local epoch total
    for (seq_idx, &rg_idx) in order.iter().enumerate() {
        let row_group_rows = row_group_index[rg_idx].num_rows;
        if rows < row_group_rows {
            row_group_pos = seq_idx;
            row_in_group = rows;
            break;
        }
        rows -= row_group_rows;
    }

    (row_group_pos, row_in_group)
}

/// Position within the infinite row-group stream, used to resume a `DataLoader`.
///
/// `epoch` + `row_group_pos` locate the feeder's starting row group.
/// `refill_count` and `buffer_offset` locate the starting position within the Buffer.
#[derive(Debug, Clone, Default)]
pub struct CheckpointCursor {
    pub stream_epoch: usize, // feeder's pass count, used to seed the shuffle
    pub row_group_pos: usize, // position within the rank-local epoch order
    pub row_in_group: usize,
    pub refill_count: usize,
    pub buffer_offset: usize,
    pub rows_epoch_start: usize, // cumulative baseline; see DataLoaderState::rows_epoch_start
}

impl CheckpointCursor {
    /// Derives the resume position from the current iteration state and a buffer snapshot.
    fn from_state(
        state: &DataLoaderState,
        shuffle_config: ShuffleConfig,
        dist_config: DistributedConfig,
        row_group_index: &[RowGroupMeta],
        buffer_snapshot: &BufferSnapshot,
    ) -> Self {
        // rows_epoch_start accumulates the absolute epoch-row baseline across resume levels
        // so that rows_at_buffer_start stays correct even after chained resumes.
        let rows_epoch_start = state.rows_epoch_start + state.rows_yielded;
        let rows_at_buffer_start = rows_epoch_start - buffer_snapshot.offset;

        // On resume, data=None so no tail is stitched. Shift the feeder to the fresh-fill
        // start and correct buffer_offset by -tail_size so resume_offset lands at the right spot.
        // If buffer_offset < tail_size (cursor is mid-tail), fall back to the tail-start position.
        let buffer_offset = buffer_snapshot.offset;
        let buffer_tail_size = buffer_snapshot.tail_size;
        let (dispatched_rows, corrected_buffer_offset) = if buffer_offset >= buffer_tail_size {
            (
                rows_at_buffer_start + buffer_tail_size,
                buffer_offset - buffer_tail_size,
            )
        } else {
            (rows_at_buffer_start, buffer_offset)
        };

        // The snapshot captures seed_offset after the last increment; subtract 1 so
        // Buffer::new starts with the value it had before that refill fired.
        let refill_count = buffer_snapshot.seed_offset.saturating_sub(1);

        // Locate the feeder's epoch and position within it. For world_size=1, locate_epoch
        // reduces to a single-step division (epoch_row_count == dataset total every epoch).
        let (stream_epoch, rows_into_epoch) =
            dist_config.locate_epoch(shuffle_config, dispatched_rows, row_group_index);
        let (row_group_pos, row_in_group) = locate_row_in_order(
            stream_epoch,
            shuffle_config,
            dist_config,
            row_group_index,
            rows_into_epoch,
        );

        Self {
            stream_epoch,
            row_group_pos,
            row_in_group,
            refill_count,
            buffer_offset: corrected_buffer_offset,
            rows_epoch_start,
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for CheckpointCursor {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let dict = ob
            .cast::<PyDict>()
            .map_err(|_| Error::InvalidCheckpointFormat("'cursor' must be a dict".into()))?;
        let cursor = CheckpointCursor {
            stream_epoch: pydict_get(&dict, "epoch")?,
            row_group_pos: pydict_get(&dict, "row_group_pos")?,
            row_in_group: pydict_get(&dict, "row_in_group")?,
            refill_count: pydict_get(&dict, "refill_count")?,
            buffer_offset: pydict_get(&dict, "buffer_offset")?,
            rows_epoch_start: pydict_get(&dict, "rows_epoch_start")?,
        };
        Ok(cursor)
    }
}

impl<'py> IntoPyObject<'py> for &CheckpointCursor {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let dict = PyDict::new(py);
        dict.set_item("epoch", self.stream_epoch)?;
        dict.set_item("row_group_pos", self.row_group_pos)?;
        dict.set_item("row_in_group", self.row_in_group)?;
        dict.set_item("refill_count", self.refill_count)?;
        dict.set_item("buffer_offset", self.buffer_offset)?;
        dict.set_item("rows_epoch_start", self.rows_epoch_start)?;
        Ok(dict)
    }
}

/// Serializable snapshot of a `DataLoader` mid-run, sufficient to resume from the exact same position.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub seed: u64,
    pub dataset_identifier: u64,
    pub epoch: usize,
    pub steps_remaining: Option<usize>,
    pub dist_config: DistributedConfig,
    pub cursor: CheckpointCursor,
}

impl<'py> IntoPyObject<'py> for &Checkpoint {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let dict = PyDict::new(py);
        dict.set_item("seed", self.seed)?;
        dict.set_item("dataset_identifier", self.dataset_identifier)?;
        dict.set_item("epoch", self.epoch)?;
        dict.set_item("steps_remaining", self.steps_remaining)?;
        dict.set_item("rank", self.dist_config.rank)?;
        dict.set_item("world_size", self.dist_config.world_size)?;
        dict.set_item("cursor", &self.cursor)?;
        Ok(dict)
    }
}

impl Checkpoint {
    /// Builds a `Checkpoint` from the current `DataLoader` state.
    pub fn from_state(
        state: &DataLoaderState,
        shuffle_config: ShuffleConfig,
        dist_config: DistributedConfig,
        dataset_identifier: u64,
        row_group_index: &[RowGroupMeta],
    ) -> Self {
        let buffer_snapshot = state
            .buffer
            .as_ref()
            .map_or(BufferSnapshot::default(), Buffer::snapshot);
        let cursor = CheckpointCursor::from_state(
            state,
            shuffle_config,
            dist_config,
            row_group_index,
            &buffer_snapshot,
        );

        Self {
            seed: state.iteration_seed,
            dataset_identifier,
            epoch: state.epoch,
            steps_remaining: state.steps_remaining,
            dist_config,
            cursor,
        }
    }
}

#[pymethods]
impl Checkpoint {
    /// Serializes to a Python dict. The result can be restored via [`Checkpoint::from_dict`].
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.into_pyobject(py)
    }

    /// Deserializes a checkpoint from a dict produced by [`Checkpoint::to_dict`].
    #[classmethod]
    pub fn from_dict(_cls: &Bound<'_, PyType>, dict: &Bound<'_, PyDict>) -> Result<Self> {
        let rank = dict
            .get_item("rank")?
            .map(|v| v.extract::<usize>())
            .transpose()?
            .unwrap_or(0);
        let world_size = dict
            .get_item("world_size")?
            .map(|v| v.extract::<usize>())
            .transpose()?
            .unwrap_or(1);
        Ok(Self {
            seed: pydict_get(dict, "seed")?,
            dataset_identifier: pydict_get(dict, "dataset_identifier")?,
            epoch: pydict_get(dict, "epoch")?,
            steps_remaining: pydict_get(dict, "steps_remaining")?,
            dist_config: DistributedConfig { rank, world_size },
            cursor: pydict_get(dict, "cursor")?,
        })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Checkpoint(epoch={}, steps_remaining={:?}, rank={}, world_size={}, cursor={{row_group_pos={}, row_in_group={}, refill_count={}, buffer_offset={}}})",
            self.epoch,
            self.steps_remaining,
            self.dist_config.rank,
            self.dist_config.world_size,
            self.cursor.row_group_pos,
            self.cursor.row_in_group,
            self.cursor.refill_count,
            self.cursor.buffer_offset,
        )
    }
}
