use std::io::Error as IoError;
use std::num::TryFromIntError;
use std::path::PathBuf;
use std::result::Result as StdResult;

use arrow::error::ArrowError;
use parquet::errors::ParquetError;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyStopIteration, PyTypeError, PyValueError};
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

    #[error("cannot build reader for row group {row_group_idx} in {path}")]
    BuildReader {
        path: PathBuf,
        row_group_idx: usize,
        #[source]
        source: ParquetError,
    },

    #[error("schema mismatch: {path} has a different schema than the first file")]
    SchemaMismatch { path: PathBuf },

    #[error("column '{name}' not found in schema")]
    ColumnNotFound { name: String },

    #[error(transparent)]
    TryFromInt(#[from] TryFromIntError),

    #[error(transparent)]
    Arrow(#[from] ArrowError),

    #[error("batch size must be > 0, got {0}")]
    InvalidBatchSize(usize),

    #[error("num_steps must be > 0, got {0}")]
    InvalidNumSteps(usize),

    #[error("length is undefined for infinite DataLoader, set num_steps to enable __len__")]
    UndefinedLength,

    #[error("num_workers must be > 0, got {0}")]
    InvalidNumWorkers(usize),

    #[error("prefetch_factor must be > 0, got {0}")]
    InvalidPrefetchFactor(usize),

    #[error("buffer_size must be >= batch_size, got {0}")]
    InvalidBufferSize(usize),

    #[error("iteration not started, call iter(loader) before next(loader)")]
    IterationNotStarted,

    #[error("dataloader exhausted")]
    DataLoaderConsumed,

    #[error("no state to checkpoint, call iter(loader) first")]
    NoStateToCheckpoint,

    #[error("dataset mismatch: checkpoint={checkpoint:#x}, current={current:#x}")]
    DatasetMismatch { checkpoint: u64, current: u64 },

    #[error("missing key in checkpoint: '{0}'")]
    MissingKeyInPyDict(String),

    #[error("invalid checkpoint format: {0}")]
    InvalidCheckpointFormat(String),

    #[error(transparent)]
    Py(#[from] PyErr),
}

pub type Result<T> = StdResult<T, Error>;

impl From<Error> for PyErr {
    fn from(e: Error) -> Self {
        if let Error::Py(inner) = e {
            return inner;
        }
        match &e {
            Error::ColumnNotFound { .. } | Error::MissingKeyInPyDict(_) => {
                PyKeyError::new_err(e.to_string())
            }
            Error::DataLoaderConsumed => PyStopIteration::new_err(e.to_string()),
            Error::UndefinedLength => PyTypeError::new_err(e.to_string()),
            Error::EmptyPaths
            | Error::SchemaMismatch { .. }
            | Error::InvalidBatchSize(_)
            | Error::InvalidNumSteps(_)
            | Error::InvalidNumWorkers(_)
            | Error::InvalidPrefetchFactor(_)
            | Error::InvalidBufferSize(_)
            | Error::DatasetMismatch { .. }
            | Error::InvalidCheckpointFormat(_) => PyValueError::new_err(e.to_string()),
            _ => PyRuntimeError::new_err(e.to_string()),
        }
    }
}
