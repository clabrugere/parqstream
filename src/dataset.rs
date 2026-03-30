use std::path::{Path, PathBuf};
use std::result::Result as StdResult;
use std::{fs::File, sync::Arc};

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelector,
};
use parquet::arrow::ProjectionMask;
use pyo3::prelude::*;

use crate::error::{Error, Result};

/// Resolve `columns` to their indices in the schema, sorted by schema order, returning an error if any is missing.
pub fn column_indices(schema: &SchemaRef, columns: &[String]) -> Result<Vec<usize>> {
    let mut indices = columns
        .iter()
        .map(|name| {
            schema
                .index_of(name)
                .map_err(|_| Error::ColumnNotFound { name: name.clone() })
        })
        .collect::<Result<Vec<_>>>()?;
    indices.sort_unstable();
    Ok(indices)
}

pub fn load_arrow_meta(path: impl AsRef<Path>) -> Result<ArrowReaderMetadata> {
    let file = File::open(&path).map_err(|e| Error::OpenFile {
        path: path.as_ref().to_path_buf(),
        source: e,
    })?;
    let arrow_meta =
        ArrowReaderMetadata::load(&file, ArrowReaderOptions::default()).map_err(|e| {
            Error::ReadParquet {
                path: path.as_ref().to_path_buf(),
                source: e,
            }
        })?;
    Ok(arrow_meta)
}

/// Metadata for a single Parquet row group, with its position in the global index.
#[derive(Debug, Clone)]
pub struct RowGroupMeta {
    pub file_idx: usize,
    pub row_group_idx: usize, // within file
    pub row_offset: usize,    // global first row of this group
    pub num_rows: usize,
}

#[derive(Debug, Clone)]
pub struct ParquetFile {
    pub path: PathBuf,
    pub arrow_meta: ArrowReaderMetadata,
}

/// Reads only footer metadata at construction time, no data is loaded until `read_batch` is called.
/// `RowGroupIndex` flattens the parquet files to allow for random row access in the whole dataset
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Dataset {
    pub files: Vec<ParquetFile>,
    pub columns: Vec<String>,
    pub projected_schema: SchemaRef,
    pub projection: ProjectionMask,
    pub row_group_index: Vec<RowGroupMeta>,
    pub total_rows: usize,
}

impl Dataset {
    /// Construct a single logical dataset from `paths`, while validating that all files share the same schema.
    pub fn new(paths: Vec<String>, columns: Option<Vec<String>>) -> Result<Self> {
        let mut paths = paths.into_iter().enumerate();
        let mut files = Vec::with_capacity(paths.len());
        let mut row_group_index = Vec::new();
        let mut total_rows = 0;

        // Read first file metadata to determine schema
        let (file_idx, first_path) = paths.next().ok_or(Error::EmptyPaths)?;
        let arrow_meta = load_arrow_meta(&first_path)?;

        let schema = arrow_meta.schema().clone();
        let columns =
            columns.unwrap_or_else(|| schema.fields().iter().map(|f| f.name().clone()).collect());
        let col_indices = column_indices(&schema, &columns)?;
        let projected_schema = Arc::new(schema.project(&col_indices)?);
        let projection = ProjectionMask::roots(arrow_meta.parquet_schema(), col_indices);

        for row_group_idx in 0..arrow_meta.metadata().num_row_groups() {
            let num_rows =
                usize::try_from(arrow_meta.metadata().row_group(row_group_idx).num_rows())?;
            row_group_index.push(RowGroupMeta {
                file_idx,
                row_group_idx,
                row_offset: total_rows,
                num_rows,
            });
            total_rows += num_rows;
        }

        files.push(ParquetFile {
            path: first_path.into(),
            arrow_meta: arrow_meta.clone(),
        });

        // Process remaining files, validating schema consistency and indexing row groups
        for (file_idx, path) in paths {
            let arrow_meta = load_arrow_meta(&path)?;
            if arrow_meta.schema().fields() != projected_schema.fields() {
                return Err(Error::SchemaMismatch { path: path.into() });
            }

            for row_group_idx in 0..arrow_meta.metadata().num_row_groups() {
                let num_rows =
                    usize::try_from(arrow_meta.metadata().row_group(row_group_idx).num_rows())?;
                row_group_index.push(RowGroupMeta {
                    file_idx,
                    row_group_idx,
                    row_offset: total_rows,
                    num_rows,
                });
                total_rows += num_rows;
            }

            files.push(ParquetFile {
                path: path.into(),
                arrow_meta: arrow_meta.clone(),
            });
        }

        Ok(Self {
            files,
            columns,
            projected_schema,
            projection,
            row_group_index,
            total_rows,
        })
    }

    pub fn read_row_group_range(
        &self,
        file_idx: usize,
        row_group_idx: usize,
        start: usize,
        len: usize,
    ) -> Result<RecordBatch> {
        let parquet_file = &self.files[file_idx];
        let file = File::open(parquet_file.path.clone()).map_err(|e| Error::OpenFile {
            path: parquet_file.path.clone(),
            source: e,
        })?;

        // skip first `start`rows and read `len` following rows
        let mut selectors = Vec::with_capacity(2);
        if start > 0 {
            selectors.push(RowSelector::skip(start));
        }
        selectors.push(RowSelector::select(len));

        let parts = ParquetRecordBatchReaderBuilder::new_with_metadata(
            file,
            parquet_file.arrow_meta.clone(),
        )
        .with_row_groups(vec![row_group_idx])
        .with_projection(self.projection.clone())
        .with_row_selection(RowSelection::from(selectors))
        .build()
        .map_err(|e| Error::BuildReader {
            path: parquet_file.path.clone(),
            row_group_idx,
            source: e,
        })?
        .collect::<StdResult<Vec<_>, _>>()?;

        Ok(concat_batches(&self.projected_schema, &parts)?)
    }
}

#[pymethods]
impl Dataset {
    #[new]
    #[pyo3(signature = (
        paths,
        columns = None,
    ))]
    pub fn py_new(paths: Vec<String>, columns: Option<Vec<String>>) -> PyResult<Self> {
        Ok(Self::new(paths, columns)?)
    }

    #[getter]
    pub fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    #[getter]
    pub fn num_files(&self) -> usize {
        self.files.len()
    }

    #[getter]
    pub fn num_row_groups(&self) -> usize {
        self.row_group_index.len()
    }

    pub fn __len__(&self) -> usize {
        self.total_rows
    }

    pub fn __repr__(&self) -> String {
        let paths: Vec<_> = self.files.iter().map(|f| &f.path).collect();
        format!(
            "Dataset(files={:?}, rows={}, columns={:?})",
            paths, self.total_rows, self.columns,
        )
    }
}
