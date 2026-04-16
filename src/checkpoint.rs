use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::buffer::{Buffer, BufferSnapshot};
use crate::dataloader::DataLoaderState;
use crate::dataset::RowGroupMeta;

#[derive(Debug, Clone, Default)]
pub struct Cursor {
    pub row_group_offset: usize,
    pub intra_row_group_offset: usize,
    pub buffer_consumed: usize,
    pub buffer_offset: usize,
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

impl Checkpoint {
    fn resolve_cursor(
        mut rows: usize,
        order: &[usize],
        row_group_index: &[RowGroupMeta],
        buffer_offset: usize,
        refill_count: usize,
    ) -> Cursor {
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
            row_group_offset,
            intra_row_group_offset,
            buffer_consumed: refill_count.saturating_sub(1),
            buffer_offset,
        }
    }

    pub fn from_state(
        state: &DataLoaderState,
        shuffle: bool,
        seed: u64,
        dataset_identifier: u64,
        row_group_index: &[RowGroupMeta],
    ) -> Self {
        let epoch = state.epoch_count.saturating_sub(1);
        let steps_remaining = state.steps_remaining;
        let buffer_snapshot = state
            .buffer
            .as_ref()
            .map_or(BufferSnapshot::default(), Buffer::snapshot);

        let rows_at_buffer_start = state.rows_within_epoch - buffer_snapshot.offset;
        let num_groups = row_group_index.len();
        let mut order: Vec<usize> = (0..num_groups).collect();
        if shuffle {
            order.shuffle(&mut SmallRng::seed_from_u64(seed + epoch as u64));
        }

        let cursor = Self::resolve_cursor(
            rows_at_buffer_start,
            &order,
            row_group_index,
            buffer_snapshot.offset,
            buffer_snapshot.refill_count,
        );

        Self {
            seed,
            dataset_identifier,
            epoch,
            steps_remaining,
            cursor,
        }
    }
}

#[pymethods]
impl Checkpoint {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let cursor = PyDict::new(py);
        cursor.set_item("row_group_offset", self.cursor.row_group_offset)?;
        cursor.set_item("intra_row_group_offset", self.cursor.intra_row_group_offset)?;
        cursor.set_item("buffer_consumed", self.cursor.buffer_consumed)?;
        cursor.set_item("buffer_offset", self.cursor.buffer_offset)?;

        let dict = PyDict::new(py);
        dict.set_item("seed", self.seed)?;
        dict.set_item("dataset_identifier", self.dataset_identifier)?;
        dict.set_item("epoch", self.epoch)?;
        dict.set_item("steps_remaining", self.steps_remaining)?;
        dict.set_item("cursor", cursor)?;

        Ok(dict)
    }

    #[classmethod]
    pub fn from_dict(_cls: &Bound<'_, PyType>, dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let cursor_obj = dict
            .get_item("cursor")?
            .ok_or_else(|| PyValueError::new_err("missing key 'cursor'"))?;
        let cursor_dict = cursor_obj
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("'cursor' must be a dict"))?;

        Ok(Self {
            seed: dict.get_item("seed")?.ok_or_else(|| PyValueError::new_err("missing key 'seed'"))?.extract::<u64>()?,
            dataset_identifier: dict.get_item("dataset_identifier")?.ok_or_else(|| PyValueError::new_err("missing key 'dataset_identifier'"))?.extract::<u64>()?,
            epoch: dict.get_item("epoch")?.ok_or_else(|| PyValueError::new_err("missing key 'epoch'"))?.extract::<usize>()?,
            steps_remaining: dict.get_item("steps_remaining")?.ok_or_else(|| PyValueError::new_err("missing key 'steps_remaining'"))?.extract::<Option<usize>>()?,
            cursor: Cursor {
                row_group_offset: cursor_dict.get_item("row_group_offset")?.ok_or_else(|| PyValueError::new_err("missing key 'cursor.row_group_offset'"))?.extract::<usize>()?,
                intra_row_group_offset: cursor_dict.get_item("intra_row_group_offset")?.ok_or_else(|| PyValueError::new_err("missing key 'cursor.intra_row_group_offset'"))?.extract::<usize>()?,
                buffer_consumed: cursor_dict.get_item("buffer_consumed")?.ok_or_else(|| PyValueError::new_err("missing key 'cursor.buffer_consumed'"))?.extract::<usize>()?,
                buffer_offset: cursor_dict.get_item("buffer_offset")?.ok_or_else(|| PyValueError::new_err("missing key 'cursor.buffer_offset'"))?.extract::<usize>()?,
            },
        })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Checkpoint(epoch={}, steps_remaining={:?}, cursor={{row_group_offset={}, intra_row_group_offset={}, buffer_consumed={}, buffer_offset={}}})",
            self.epoch,
            self.steps_remaining,
            self.cursor.row_group_offset,
            self.cursor.intra_row_group_offset,
            self.cursor.buffer_consumed,
            self.cursor.buffer_offset,
        )
    }
}
