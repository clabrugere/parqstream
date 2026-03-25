use std::collections::{BinaryHeap, HashMap};
use std::fs::File;

use arrow::compute::concat_batches;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
use parquet::arrow::ProjectionMask;

use crate::dataset::Dataset;
use crate::error::{Error, Result};

/// Read a batch of rows from `dataset` identified by `global_indices`.
///
/// Rows are grouped by `(file, row_group)` to minimize seeks.
/// The output batch may contain rows in a different order than `global_indices` (sorted by file then row group)
///
/// Duplicate indices within the same row group are deduplicated (`RowSelection` selects each row exactly once).
/// This means that if `global_indices` contains duplicates, the resulting batch may be smaller than expected.
pub fn read_batch(
    dataset: &Dataset,
    global_indices: &[usize],
    columns: &[String],
) -> Result<RecordBatch> {
    // Group sorted indices by (file_idx, row_group_idx)
    let mut groups: HashMap<(usize, usize), BinaryHeap<usize>> = HashMap::new();
    for &global in global_indices {
        let (meta, local) = dataset.locate_row(global);
        groups
            .entry((meta.file_idx, meta.row_group_idx))
            .or_default()
            .push(local);
    }

    // Read each (file, row_group) and collect RecordBatches
    let col_indices = dataset.column_indices(columns)?;
    let projected_schema = std::sync::Arc::new(dataset.schema.project(&col_indices)?);
    let mut batches = Vec::with_capacity(groups.len());

    for ((file_idx, rg_idx), local_rows) in groups {
        let path = &dataset.files[file_idx];
        let file = File::open(path).map_err(|e| Error::OpenFile {
            path: path.clone(),
            source: e,
        })?;

        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| Error::ReadParquet {
                path: path.clone(),
                source: e,
            })?;

        let parquet_schema = builder.parquet_schema();
        let projection = ProjectionMask::roots(parquet_schema, col_indices.clone());

        let local_rows = {
            let mut local_rows = local_rows.into_sorted_vec();
            local_rows.dedup();
            local_rows
        };
        let row_selection = make_row_selection(&local_rows);

        let mut reader = builder
            .with_row_groups(vec![rg_idx])
            .with_projection(projection)
            .with_row_selection(row_selection)
            .build()
            .map_err(|e| Error::BuildReader {
                path: path.clone(),
                rg: rg_idx,
                source: e,
            })?;

        // Collect all RecordBatches
        for rb in &mut reader {
            batches.push(rb?);
        }
    }

    let batch = if batches.is_empty() {
        // Return an empty batch with the correct schema
        RecordBatch::new_empty(projected_schema)
    } else {
        // Concatenate batches from different row groups into a single batch.
        concat_batches(&projected_schema, &batches)?
    };

    Ok(batch)
}

/// Build a `RowSelection` that selects `sorted_local_indices` (must be sorted and deduplicated) from a row group.
fn make_row_selection(sorted_local_indices: &[usize]) -> RowSelection {
    let mut selectors = Vec::new();
    let mut prev = 0;

    for &idx in sorted_local_indices {
        if idx > prev {
            selectors.push(RowSelector::skip(idx - prev));
        }
        selectors.push(RowSelector::select(1));
        prev = idx + 1;
    }

    RowSelection::from(selectors)
}
