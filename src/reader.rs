use std::collections::{BinaryHeap, HashMap};
use std::fs::File;

use arrow::compute::concat_batches;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};

use crate::dataset::Dataset;
use crate::error::{Error, Result};

/// Group indices by (`file_idx`, `row_group_idx`) and sort them within each group for efficient row selection.
fn group_indices(
    dataset: &Dataset,
    global_indices: &[usize],
) -> HashMap<(usize, usize), Vec<usize>> {
    let mut groups: HashMap<(usize, usize), BinaryHeap<usize>> = HashMap::new();
    for &global_idx in global_indices {
        let (meta, local) = dataset.locate_row(global_idx);
        groups
            .entry((meta.file_idx, meta.row_group_idx))
            .or_default()
            .push(local);
    }
    // Convert BinaryHeaps to sorted and deduplicated Vecs
    groups
        .into_iter()
        .map(|(key, local_rows)| {
            (key, {
                let mut local_rows = local_rows.into_sorted_vec();
                local_rows.dedup();
                local_rows
            })
        })
        .collect()
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

/// Read a batch of rows from `dataset` identified by `global_indices`
///
/// Rows are grouped by `(file_idx, row_group_idx)` to minimize seeks
/// The output batch may contain rows in a different order than `global_indices` (sorted by file then row group)
///
/// Duplicate indices within the same row group are deduplicated (`RowSelection` selects each row exactly once)
/// This means that if `global_indices` contains duplicates, the resulting batch may be smaller than expected
pub fn read_batch(dataset: &Dataset, global_indices: &[usize]) -> Result<RecordBatch> {
    let groups = group_indices(dataset, global_indices);
    let mut batches = Vec::with_capacity(groups.len());

    for ((file_idx, row_group_idx), local_rows) in groups {
        let parquet_file = &dataset.files[file_idx];
        let file = File::open(parquet_file.path.clone()).map_err(|e| Error::OpenFile {
            path: parquet_file.path.clone(),
            source: e,
        })?;
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(
            file,
            parquet_file.arrow_meta.clone(),
        );

        let row_selection = make_row_selection(&local_rows);

        let reader = builder
            .with_row_groups(vec![row_group_idx])
            .with_projection(dataset.projection.clone())
            .with_row_selection(row_selection)
            .build()
            .map_err(|e| Error::BuildReader {
                path: parquet_file.path.clone(),
                row_group_idx,
                source: e,
            })?;

        // Collect all RecordBatches
        for rb in reader {
            batches.push(rb?);
        }
    }

    let batch = if batches.is_empty() {
        // Return an empty batch with the correct schema
        RecordBatch::new_empty(dataset.projected_schema.clone())
    } else {
        // Concatenate batches from different row groups into a single batch otherwise
        concat_batches(&dataset.projected_schema, &batches)?
    };

    Ok(batch)
}
