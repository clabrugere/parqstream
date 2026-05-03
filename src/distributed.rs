use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::dataloader::ShuffleConfig;
use crate::dataset::RowGroupMeta;

#[derive(Debug, Clone, Copy)]
pub struct DistributedConfig {
    pub rank: usize,
    pub world_size: usize,
}

impl Default for DistributedConfig {
    // Manual impl required: usize's derived Default is 0, which would produce
    // the invalid `world_size=0`. The identity (single-process) config is rank=0, world_size=1.
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }
}

impl DistributedConfig {
    /// Build the visit order for a specific epoch: globally shuffle (if enabled) then
    /// retain only this rank's strided positions. `world_size=1` returns all groups unchanged.
    pub fn epoch_order(
        &self,
        shuffle_config: ShuffleConfig,
        epoch: usize,
        num_groups: usize,
    ) -> Vec<usize> {
        let mut order: Vec<usize> = (0..num_groups).collect();
        if shuffle_config.shuffle {
            order.shuffle(&mut SmallRng::seed_from_u64(
                shuffle_config.seed + epoch as u64,
            ));
        }
        if self.world_size == 1 {
            return order;
        }
        order
            .into_iter()
            .skip(self.rank)
            .step_by(self.world_size)
            .collect()
    }

    /// Total rows this rank processes in the given epoch.
    fn epoch_row_count(
        &self,
        shuffle_config: ShuffleConfig,
        epoch: usize,
        rgi: &[RowGroupMeta],
    ) -> usize {
        self.epoch_order(shuffle_config, epoch, rgi.len())
            .iter()
            .map(|&i| rgi[i].num_rows)
            .sum()
    }

    /// Given `dispatched_rows` (total rows this rank's feeder has sent since iteration start,
    /// adjusted for buffer position), return `(stream_epoch, rows_into_epoch)`.
    ///
    /// The loop is needed because rank-local epoch sizes vary when shuffle is on
    /// (different row groups land in the strided slice each epoch).
    /// For `world_size=1`, `epoch_row_count` is the dataset total (invariant), so the loop terminates in one step.
    pub fn locate_epoch(
        &self,
        shuffle_config: ShuffleConfig,
        dispatched_rows: usize,
        row_group_index: &[RowGroupMeta],
    ) -> (usize, usize) {
        let mut remaining = dispatched_rows;
        let mut epoch = 0;
        loop {
            let epoch_rows = self.epoch_row_count(shuffle_config, epoch, row_group_index);
            if remaining < epoch_rows {
                break;
            }
            remaining -= epoch_rows;
            epoch += 1;
        }
        (epoch, remaining)
    }
}
