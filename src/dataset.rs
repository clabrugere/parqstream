use std::path::{Path, PathBuf};
use std::{fs::File, sync::Arc};

use arrow::datatypes::SchemaRef;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::{arrow_reader::ArrowReaderMetadata, ProjectionMask};
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
        .collect::<Result<Vec<usize>>>()?;
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

    /// Find the `RowGroupMeta` that contains `global_row` and return it along with the local row index within that group
    pub fn locate_row(&self, global_row: usize) -> (&RowGroupMeta, usize) {
        // Binary search on row_offset.
        let idx = self
            .row_group_index
            .partition_point(|m| m.row_offset <= global_row)
            .saturating_sub(1);
        let meta = &self.row_group_index[idx];
        let local = global_row - meta.row_offset;
        (meta, local)
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
