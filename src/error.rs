use std::io::Error as IoError;
use std::num::TryFromIntError;
use std::path::PathBuf;
use std::result::Result as StdResult;

use arrow::error::ArrowError;
use parquet::errors::ParquetError;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("at least one file path is required")]
    EmptyPaths,

    #[error("cannot open {path}")]
    OpenFile {
        path: PathBuf,
        #[source]
        source: IoError,
    },

    #[error("cannot read {path}")]
    ReadParquet {
        path: PathBuf,
        #[source]
        source: ParquetError,
    },

    #[error("cannot build reader for row group {rg} in {path}")]
    BuildReader {
        path: PathBuf,
        rg: usize,
        #[source]
        source: ParquetError,
    },

    #[error("schema mismatch: {path} has a different schema than the first file")]
    SchemaMismatch { path: PathBuf },

    #[error("column '{name}' not found in schema")]
    ColumnNotFound { name: String },

    #[error("cannot convert usize")]
    TryFromInt {
        #[from]
        source: TryFromIntError,
    },

    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

pub type Result<T> = StdResult<T, Error>;

impl From<Error> for PyErr {
    fn from(e: Error) -> Self {
        match &e {
            Error::EmptyPaths | Error::SchemaMismatch { .. } => {
                PyValueError::new_err(e.to_string())
            }
            Error::ColumnNotFound { .. } => PyKeyError::new_err(e.to_string()),
            _ => PyRuntimeError::new_err(e.to_string()),
        }
    }
}
