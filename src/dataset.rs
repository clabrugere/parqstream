use std::fs::File;
use std::path::PathBuf;

use arrow::datatypes::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::{Error, Result};

/// Metadata for a single Parquet row group, with its position in the global index.
#[derive(Debug, Clone)]
pub struct RowGroupMeta {
    pub file_idx: usize,
    pub row_group_idx: usize, // within file
    pub row_offset: usize,    // global first row of this group
    pub num_rows: usize,
}

/// Reads only footer metadata at construction time, no data is loaded until `read_batch` is called.
/// RowGroupIndex flatten the parquet files to allow for random row access in the whole dataset
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Dataset {
    pub files: Vec<PathBuf>,
    pub schema: SchemaRef,
    pub row_group_index: Vec<RowGroupMeta>,
    pub total_rows: usize,
    columns: Vec<String>,
}

impl Dataset {
    /// Construct a single logical dataset from `paths`, while validating that all files share the same schema.
    pub fn from_files(paths: Vec<String>) -> Result<Self> {
        if paths.is_empty() {
            return Err(Error::EmptyPaths);
        }

        let mut files = Vec::with_capacity(paths.len());
        let mut row_group_index = Vec::new();
        let mut total_rows = 0;
        let mut schema = None;

        for (file_idx, path) in paths.into_iter().enumerate() {
            let file = File::open(&path).map_err(|e| Error::OpenFile {
                path: path.as_str().into(),
                source: e,
            })?;
            let builder =
                ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| Error::ReadParquet {
                    path: path.as_str().into(),
                    source: e,
                })?;

            // Validate schema consistency across files.
            let file_schema = builder.schema().clone();
            match &schema {
                None => schema = Some(file_schema),
                Some(s) => {
                    if s.fields() != file_schema.fields() {
                        return Err(Error::SchemaMismatch {
                            path: path.as_str().into(),
                        });
                    }
                }
            }

            // Index all row groups.
            let metadata = builder.metadata();
            for rg_idx in 0..metadata.num_row_groups() {
                let num_rows = metadata.row_group(rg_idx).num_rows() as usize;
                row_group_index.push(RowGroupMeta {
                    file_idx,
                    row_group_idx: rg_idx,
                    row_offset: total_rows,
                    num_rows,
                });
                total_rows += num_rows;
            }

            files.push(path.into());
        }

        let schema = schema.unwrap(); // guaranteed by non-empty paths check
        let columns = schema.fields().iter().map(|f| f.name().clone()).collect();

        Ok(Self {
            files,
            schema,
            row_group_index,
            total_rows,
            columns,
        })
    }

    /// Find the RowGroupMeta that contains `global_row` and return it along with the local row index within that group
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

    /// Resolve `columns` to their indices in the Arrow schema, sorted by schema order.
    /// Returns an error if any column name is not found.
    pub fn column_indices(&self, columns: &[String]) -> Result<Vec<usize>> {
        let mut indices: Vec<usize> = columns
            .iter()
            .map(|name| {
                self.schema
                    .index_of(name)
                    .map_err(|_| Error::ColumnNotFound { name: name.clone() })
            })
            .collect::<Result<_>>()?;
        indices.sort_unstable();
        Ok(indices)
    }
}

#[pymethods]
impl Dataset {
    #[new]
    pub fn py_new(paths: Vec<String>) -> PyResult<Self> {
        if paths.is_empty() {
            return Err(PyValueError::new_err("at least one file is required"));
        }
        Ok(Self::from_files(paths)?)
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
        format!(
            "Dataset(files={:?}, rows={}, columns={:?})",
            self.files, self.total_rows, self.columns,
        )
    }
}
