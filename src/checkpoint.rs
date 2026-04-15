use pyo3::prelude::*;
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
