use crate::{dataloader::ShuffleConfig, dataset::RowGroupMeta, distributed::DistributedConfig};

/// Represents the position in the infinite stream of rows a `DataLoader` has yielded so far.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamPosition {
    pub epoch: usize, // number of passes over the dataset. Used to seed the per-epoch shuffle
    pub row_group_pos: usize, // index into this rank's epoch order
    pub row_in_group: usize, // rows already consumed from the current row_group
}

impl StreamPosition {
    /// Advances past `num_rows`, rolling over to the next row group when the current one is exhausted.
    pub fn advance(&mut self, num_rows: usize, row_group_length: usize) {
        self.row_in_group += num_rows;
        if self.row_in_group >= row_group_length {
            self.row_group_pos += 1;
            self.row_in_group = 0;
        }
    }

    /// Resets to the start of the next epoch.
    pub fn new_epoch(&mut self) {
        self.epoch += 1;
        self.row_group_pos = 0;
        self.row_in_group = 0;
    }

    /// Walk the rank-local ordered row groups to find which one contains `rows`.
    pub fn locate(
        dist_config: DistributedConfig,
        shuffle_config: ShuffleConfig,
        dispatched_rows: usize,
        row_group_index: &[RowGroupMeta],
    ) -> Self {
        // Locate the feeder's epoch and position within it for this rank. For world_size=1, locate_epoch
        // reduces to a single-step division (epoch_row_count == dataset total every epoch).
        let (epoch, mut rows_into_epoch) =
            dist_config.locate_epoch(shuffle_config, dispatched_rows, row_group_index);

        let order = dist_config.epoch_order(shuffle_config, epoch, row_group_index.len());
        let mut row_group_pos = 0;
        let mut row_in_group = 0;

        // always terminates because rows < rank-local epoch total
        for (seq_idx, &rg_idx) in order.iter().enumerate() {
            let row_group_rows = row_group_index[rg_idx].num_rows;
            if rows_into_epoch < row_group_rows {
                row_group_pos = seq_idx;
                row_in_group = rows_into_epoch;
                break;
            }
            rows_into_epoch -= row_group_rows;
        }

        Self {
            epoch,
            row_group_pos,
            row_in_group,
        }
    }
}

/// Represents the position in an assembled `Buffer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BufferPosition {
    pub refill_count: usize, // number of refills so far
    pub offset: usize,       // rows already consumed from the current fill
}
