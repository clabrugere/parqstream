use std::ffi::CString;
use std::sync::Arc;

use arrow::array::Array;
use arrow::ffi;
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[pyclass]
pub struct Batch {
    data: RecordBatch,
    columns: Vec<String>,
}

impl Batch {
    pub fn new(data: RecordBatch) -> Self {
        let columns = data
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        Self { data, columns }
    }
}

#[pyclass]
pub struct Column {
    array: Arc<dyn Array>,
}

impl Column {
    pub fn new(array: Arc<dyn Array>) -> Self {
        Self { array }
    }
}

#[pymethods]
impl Batch {
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    fn column(&self, name: &str) -> PyResult<Column> {
        let idx = self
            .data
            .schema()
            .index_of(name)
            .map_err(|_| PyKeyError::new_err(format!("no column '{name}'")))?;
        Ok(Column::new(self.data.column(idx).clone()))
    }

    fn __len__(&self) -> usize {
        self.data.num_rows()
    }
}

#[pymethods]
impl Column {
    /// Arrow PyCapsule Interface.
    /// Returns `(schema_capsule, array_capsule)` per the Arrow specification.
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        _requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let (ffi_array, ffi_schema) = ffi::to_ffi(&self.array.to_data())
            .map_err(|e: arrow::error::ArrowError| PyRuntimeError::new_err(e.to_string()))?;

        let schema_cap =
            PyCapsule::new(py, ffi_schema, Some(CString::new("arrow_schema").unwrap()))?;
        let array_cap = PyCapsule::new(py, ffi_array, Some(CString::new("arrow_array").unwrap()))?;

        Ok((schema_cap, array_cap))
    }
}
