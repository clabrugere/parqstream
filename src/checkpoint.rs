use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::buffer::{Buffer, BufferSnapshot};
use crate::dataloader::DataLoaderState;
use crate::dataset::RowGroupMeta;

/// Position within the infinite row-group stream, used to resume a `DataLoader`.
///
/// `epoch_offset` + `row_group_offset` locate the feeder's starting row group.
/// `buffer_seed_offset` and `buffer_offset` locate the starting position within the Buffer.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, Default)]
pub struct Cursor {
    pub epoch_offset: usize,
    pub row_group_offset: usize,
    pub intra_row_group_offset: usize,
    pub buffer_seed_offset: usize,
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
        rows_at_buffer_start: usize,
        seed: u64, // per-iteration seed
        shuffle: bool,
        row_group_index: &[RowGroupMeta],
        buffer_offset: usize,
        buffer_seed_offset: usize,
        buffer_tail_size: usize,
    ) -> Cursor {
        let total_rows = row_group_index.iter().map(|rg| rg.num_rows).sum::<usize>();

        // On resume, data=None so no tail is stitched. Shift the feeder to the fresh-epoch
        // start and correct buffer_offset by -tail_size so initial_offset lands at the right spot.
        // If buffer_offset < tail_size (cursor is mid-tail), fall back to the tail-start position.
        let (epoch_rows, corrected_buffer_offset) = if buffer_offset >= buffer_tail_size {
            (
                rows_at_buffer_start + buffer_tail_size,
                buffer_offset - buffer_tail_size,
            )
        } else {
            (rows_at_buffer_start, buffer_offset)
        };

        let epoch_offset = epoch_rows / total_rows;
        let rows = epoch_rows % total_rows;

        // Reconstruct the row group visit order for this epoch, mirroring chunk_feeder's logic:
        // always shuffle [0..N) so each epoch's order is independent of the previous one.
        let num_groups = row_group_index.len();
        let mut order: Vec<usize> = (0..num_groups).collect();
        if shuffle {
            order.shuffle(&mut SmallRng::seed_from_u64(seed + epoch_offset as u64));
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
            // The snapshot captures seed_offset after the last increment; subtract 1 so
            // Buffer::new starts with the value it had before that refill fired.
            buffer_seed_offset: buffer_seed_offset.saturating_sub(1),
            buffer_offset: corrected_buffer_offset,
        }
    }

    pub fn from_state(
        state: &DataLoaderState,
        shuffle: bool,
        seed: u64,
        dataset_identifier: u64,
        row_group_index: &[RowGroupMeta],
    ) -> Self {
        let buffer_snapshot = state
            .buffer
            .as_ref()
            .map_or(BufferSnapshot::default(), Buffer::snapshot);

        // rows_within_epoch counts rows delivered to Python; subtracting the current buffer
        // read position gives the total rows that had been fed into the pipeline before the
        // current buffer fill started.
        let rows_at_buffer_start = state.rows_within_epoch - buffer_snapshot.offset;

        let cursor = Self::resolve_cursor(
            rows_at_buffer_start,
            seed,
            shuffle,
            row_group_index,
            buffer_snapshot.offset,
            buffer_snapshot.seed_offset,
            buffer_snapshot.tail_size,
        );

        Self {
            seed,
            dataset_identifier,
            epoch: state.epoch_count,
            steps_remaining: state.steps_remaining,
            cursor,
        }
    }
}

#[pymethods]
impl Checkpoint {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let cursor = PyDict::new(py);
        cursor.set_item("epoch_offset", self.cursor.epoch_offset)?;
        cursor.set_item("row_group_offset", self.cursor.row_group_offset)?;
        cursor.set_item("intra_row_group_offset", self.cursor.intra_row_group_offset)?;
        cursor.set_item("buffer_seed_offset", self.cursor.buffer_seed_offset)?;
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

        let cursor = Cursor {
            epoch_offset: cursor_dict
                .get_item("epoch_offset")?
                .ok_or_else(|| PyValueError::new_err("missing key 'epoch_offset'"))?
                .extract::<usize>()?,
            row_group_offset: cursor_dict
                .get_item("row_group_offset")?
                .ok_or_else(|| PyValueError::new_err("missing key 'cursor.row_group_offset'"))?
                .extract::<usize>()?,
            intra_row_group_offset: cursor_dict
                .get_item("intra_row_group_offset")?
                .ok_or_else(|| {
                    PyValueError::new_err("missing key 'cursor.intra_row_group_offset'")
                })?
                .extract::<usize>()?,
            buffer_seed_offset: cursor_dict
                .get_item("buffer_seed_offset")?
                .ok_or_else(|| PyValueError::new_err("missing key 'cursor.buffer_seed_offset'"))?
                .extract::<usize>()?,
            buffer_offset: cursor_dict
                .get_item("buffer_offset")?
                .ok_or_else(|| PyValueError::new_err("missing key 'cursor.buffer_offset'"))?
                .extract::<usize>()?,
        };

        Ok(Self {
            seed: dict
                .get_item("seed")?
                .ok_or_else(|| PyValueError::new_err("missing key 'seed'"))?
                .extract::<u64>()?,
            dataset_identifier: dict
                .get_item("dataset_identifier")?
                .ok_or_else(|| PyValueError::new_err("missing key 'dataset_identifier'"))?
                .extract::<u64>()?,
            epoch: dict
                .get_item("epoch")?
                .ok_or_else(|| PyValueError::new_err("missing key 'epoch'"))?
                .extract::<usize>()?,
            steps_remaining: dict
                .get_item("steps_remaining")?
                .ok_or_else(|| PyValueError::new_err("missing key 'steps_remaining'"))?
                .extract::<Option<usize>>()?,
            cursor,
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
