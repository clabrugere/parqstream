use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyType};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::buffer::{Buffer, BufferSnapshot};
use crate::dataloader::DataLoaderState;
use crate::dataset::RowGroupMeta;
use crate::error::{Error, Result};

fn pydict_get<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> Result<T>
where
    for<'a> T: FromPyObject<'a, 'py, Error = PyErr>,
{
    dict.get_item(key)?
        .ok_or_else(|| Error::MissingKeyInPyDict(key.into()))
        .and_then(|v| Ok(v.extract::<T>()?))
}

/// Position within the infinite row-group stream, used to resume a `DataLoader`.
///
/// `epoch_offset` + `row_group_offset` locate the feeder's starting row group.
/// `buffer_seed_offset` and `buffer_offset` locate the starting position within the Buffer.
#[derive(Debug, Clone, Default)]
pub struct Cursor {
    pub epoch_offset: usize,
    pub row_group_offset: usize,
    pub intra_row_group_offset: usize,
    pub buffer_seed_offset: usize,
    pub buffer_offset: usize,
    pub rows_epoch_start: usize, // cumulative baseline; see DataLoaderState::rows_epoch_start
}

impl Cursor {
    fn from_pydict<'py>(dict: &Bound<'py, PyDict>) -> Result<Self> {
        Ok(Cursor {
            epoch_offset: pydict_get(dict, "epoch_offset")?,
            row_group_offset: pydict_get(dict, "row_group_offset")?,
            intra_row_group_offset: pydict_get(dict, "intra_row_group_offset")?,
            buffer_seed_offset: pydict_get(dict, "buffer_seed_offset")?,
            buffer_offset: pydict_get(dict, "buffer_offset")?,
            rows_epoch_start: pydict_get(dict, "rows_epoch_start")?,
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Cursor {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let dict = ob
            .cast::<PyDict>()
            .map_err(|_| Error::InvalidCheckpointFormat("'cursor' must be a dict".into()))?;
        Ok(Cursor::from_pydict(&dict)?)
    }
}

impl<'py> IntoPyObject<'py> for &Cursor {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let dict = PyDict::new(py);
        dict.set_item("epoch_offset", self.epoch_offset)?;
        dict.set_item("row_group_offset", self.row_group_offset)?;
        dict.set_item("intra_row_group_offset", self.intra_row_group_offset)?;
        dict.set_item("buffer_seed_offset", self.buffer_seed_offset)?;
        dict.set_item("buffer_offset", self.buffer_offset)?;
        dict.set_item("rows_epoch_start", self.rows_epoch_start)?;
        Ok(dict)
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub seed: u64,
    pub dataset_identifier: u64,
    pub epoch: usize,
    pub steps_remaining: Option<usize>,
    pub cursor: Cursor,
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
        dict.set_item("cursor", &self.cursor)?;
        Ok(dict)
    }
}

impl Checkpoint {
    fn resolve_cursor(
        state: &DataLoaderState,
        shuffle: bool,
        row_group_index: &[RowGroupMeta],
        buffer_snapshot: &BufferSnapshot,
    ) -> Cursor {
        let total_rows = row_group_index.iter().map(|rg| rg.num_rows).sum::<usize>();

        // rows_epoch_start accumulates the absolute epoch-row baseline across resume levels
        // so that rows_at_buffer_start stays correct even after chained resumes.
        let rows_epoch_start = state.rows_epoch_start + state.rows_yielded;
        let rows_at_buffer_start = rows_epoch_start - buffer_snapshot.offset;

        // On resume, data=None so no tail is stitched. Shift the feeder to the fresh-epoch
        // start and correct buffer_offset by -tail_size so resume_offset lands at the right spot.
        // If buffer_offset < tail_size (cursor is mid-tail), fall back to the tail-start position.
        let buffer_offset = buffer_snapshot.offset;
        let buffer_tail_size = buffer_snapshot.tail_size;
        let (epoch_rows, corrected_buffer_offset) = if buffer_offset >= buffer_tail_size {
            (
                rows_at_buffer_start + buffer_tail_size,
                buffer_offset - buffer_tail_size,
            )
        } else {
            (rows_at_buffer_start, buffer_offset)
        };

        // The snapshot captures seed_offset after the last increment; subtract 1 so
        // Buffer::new starts with the value it had before that refill fired.
        let buffer_seed_offset = buffer_snapshot.seed_offset.saturating_sub(1);

        let epoch_offset = epoch_rows / total_rows;
        let rows = epoch_rows % total_rows;

        // Reconstruct the row group visit order for this epoch, mirroring chunk_feeder's logic:
        // always shuffle [0..N) so each epoch's order is independent of the previous one.
        let num_groups = row_group_index.len();
        let seed = state.iteration_seed + epoch_offset as u64;
        let mut order: Vec<usize> = (0..num_groups).collect();
        if shuffle {
            order.shuffle(&mut SmallRng::seed_from_u64(seed));
        }

        // Walk the ordered row groups to find which one contains `rows` (always terminates
        // because rows < total_rows after the modulo above).
        let mut rows = rows;
        let mut row_group_offset = 0;
        let mut intra_row_group_offset = 0;
        for (seq_idx, &rg_idx) in order.iter().enumerate() {
            let row_group_rows = row_group_index[rg_idx].num_rows;
            if rows < row_group_rows {
                row_group_offset = seq_idx;
                intra_row_group_offset = rows;
                break;
            }
            rows -= row_group_rows;
        }

        Cursor {
            epoch_offset,
            row_group_offset,
            intra_row_group_offset,
            buffer_seed_offset,
            buffer_offset: corrected_buffer_offset,
            rows_epoch_start,
        }
    }

    pub fn from_state(
        state: &DataLoaderState,
        shuffle: bool,
        dataset_identifier: u64,
        row_group_index: &[RowGroupMeta],
    ) -> Self {
        let buffer_snapshot = state
            .buffer
            .as_ref()
            .map_or(BufferSnapshot::default(), Buffer::snapshot);
        let cursor = Self::resolve_cursor(state, shuffle, row_group_index, &buffer_snapshot);

        Self {
            seed: state.iteration_seed,
            dataset_identifier,
            epoch: state.epoch,
            steps_remaining: state.steps_remaining,
            cursor,
        }
    }
}

#[pymethods]
impl Checkpoint {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.into_pyobject(py)
    }

    #[classmethod]
    pub fn from_dict(_cls: &Bound<'_, PyType>, dict: &Bound<'_, PyDict>) -> Result<Self> {
        Ok(Self {
            seed: pydict_get(dict, "seed")?,
            dataset_identifier: pydict_get(dict, "dataset_identifier")?,
            epoch: pydict_get(dict, "epoch")?,
            steps_remaining: pydict_get(dict, "steps_remaining")?,
            cursor: pydict_get(dict, "cursor")?,
        })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Checkpoint(epoch={}, steps_remaining={:?}, cursor={{row_group_offset={}, intra_row_group_offset={}, buffer_seed_offset={}, buffer_offset={}}})",
            self.epoch,
            self.steps_remaining,
            self.cursor.row_group_offset,
            self.cursor.intra_row_group_offset,
            self.cursor.buffer_seed_offset,
            self.cursor.buffer_offset,
        )
    }
}
