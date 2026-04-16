use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use crossbeam_channel::{Receiver, Sender};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::dataset::Dataset;
use crate::error::Result;

#[derive(Debug)]
struct EpochCursor {
    pub epoch: u64,
    pub row_group_offset: usize,
    pub intra_row_group_offset: usize,
}

impl EpochCursor {
    pub fn advance(&mut self, num_rows: usize, row_group_length: usize) {
        self.intra_row_group_offset += num_rows;
        if self.intra_row_group_offset >= row_group_length {
            self.row_group_offset += 1;
            self.intra_row_group_offset = 0;
        }
    }
    pub fn new_epoch(&mut self) {
        self.epoch += 1;
        self.row_group_offset = 0;
        self.intra_row_group_offset = 0;
    }
}

#[derive(Debug)]
pub struct Chunk {
    pub row_group_idx: usize,
    pub start_row: usize,
    pub num_rows: usize,
}

// continuously sends row group read tasks to the chunk channel, shuffled if needed
#[allow(clippy::too_many_arguments)]
pub fn chunk_feeder(
    chunk_tx: &Sender<Chunk>,
    row_group_lengths: &[usize],
    chunk_size: usize,
    shuffle: bool,
    seed: u64,
    epoch: u64,
    row_group_offset: usize,
    intra_row_group_offset: usize,
) {
    let mut cursor = EpochCursor {
        epoch,
        row_group_offset,
        intra_row_group_offset,
    };

    let num_groups = row_group_lengths.len();
    let mut order = (0..num_groups).collect::<Vec<_>>();
    if shuffle {
        order.shuffle(&mut SmallRng::seed_from_u64(seed + cursor.epoch));
    }

    loop {
        // new epoch , re-shuffle if needed
        if cursor.row_group_offset >= num_groups {
            cursor.new_epoch();
            if shuffle {
                order.shuffle(&mut SmallRng::seed_from_u64(seed + cursor.epoch));
            }
        }
        let row_group_idx = order[cursor.row_group_offset];
        let row_group_length = row_group_lengths[row_group_idx];
        let num_rows = chunk_size.min(row_group_length - cursor.intra_row_group_offset);

        if chunk_tx
            .send(Chunk {
                row_group_idx,
                start_row: cursor.intra_row_group_offset,
                num_rows,
            })
            .is_err()
        {
            return; // consumer dropped
        }
        cursor.advance(num_rows, row_group_length);
    }
}

// reads chunks received from the chunk channel, gather rows and sends record batches to the batch channel
pub fn read_feeder(
    chunk_rx: &Receiver<Chunk>,
    data_tx: &Sender<Result<RecordBatch>>,
    dataset: &Dataset,
) {
    for Chunk {
        row_group_idx,
        start_row,
        num_rows,
    } in chunk_rx
    {
        let meta = &dataset.row_group_index[row_group_idx];
        match dataset.read_row_group_range(meta.file_idx, meta.row_group_idx, start_row, num_rows) {
            Ok(chunk_data) => {
                if data_tx.send(Ok(chunk_data)).is_err() {
                    return; // consumer dropped
                }
            }
            Err(e) => {
                let _ = data_tx.send(Err(e));
                return; // upstream error
            }
        }
    }
}

// continuously collects chunks until > buffer_size rows, concatenates and sends to batch channel
pub fn collector(
    data_rx: &Receiver<Result<RecordBatch>>,
    buffer_tx: &Sender<Result<RecordBatch>>,
    schema: &SchemaRef,
    buffer_size: usize,
) {
    loop {
        let mut parts: Vec<RecordBatch> = Vec::new();
        let mut rows = 0;
        while rows < buffer_size {
            match data_rx.recv() {
                Ok(Ok(chunk_data)) => {
                    rows += chunk_data.num_rows();
                    parts.push(chunk_data);
                }
                Ok(Err(e)) => {
                    let _ = buffer_tx.send(Err(e));
                    return; // upstream error
                }
                Err(_) => return, // consumer dropped
            }
        }
        let buffer = match concat_batches(schema, &parts) {
            Ok(buffer) => buffer,
            Err(e) => {
                let _ = buffer_tx.send(Err(e.into()));
                return; // error concatenating batches
            }
        };
        if buffer_tx.send(Ok(buffer)).is_err() {
            return; // consumer dropped
        }
    }
}
