use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use crossbeam_channel::{Receiver, Sender};
use rand::prelude::*;

use crate::dataset::Dataset;
use crate::error::Result;

#[derive(Debug)]
pub struct Chunk {
    pub row_group_idx: usize,
    pub start_row: usize,
    pub num_rows: usize,
}

// continuously sends row group read tasks to the chunk channel, shuffled if needed
pub fn chunk_feeder(
    chunk_tx: &Sender<Chunk>,
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
        let num_rows = chunk_size.min(num_rows - intra_row_group_offset);

        if chunk_tx
            .send(Chunk {
                row_group_idx,
                start_row: intra_row_group_offset,
                num_rows,
            })
            .is_err()
        {
            break; // consumer dropped
        }
        intra_row_group_offset += num_rows;
        if intra_row_group_offset >= num_rows {
            row_group_offset += 1;
            intra_row_group_offset = 0;
        }
    }
}

// continuously reads chunks of rows from row groups and sends them to the data channel
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
            Ok(b) => {
                if data_tx.send(Ok(b)).is_err() {
                    break; // consumer dropped
                }
            }
            Err(e) => {
                let _ = data_tx.send(Err(e));
                break; // upstream error
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
                Ok(Ok(chunk)) => {
                    rows += chunk.num_rows();
                    parts.push(chunk);
                }
                Ok(Err(e)) => {
                    let _ = buffer_tx.send(Err(e));
                    return; // upstream error
                }
                Err(_) => return, // consumer dropped
            }
        }
        let buffer = match concat_batches(schema, &parts) {
            Ok(b) => b,
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
