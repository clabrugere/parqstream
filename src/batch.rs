use std::sync::Arc;

use arrow::array::{Array, RecordBatchIterator};
use arrow::ffi;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[pyclass]
pub struct Batch {
    data: RecordBatch,
}

impl Batch {
    pub fn new(data: RecordBatch) -> Self {
        Self { data }
    }
}

#[pyclass]
pub struct Column {
    array: Arc<dyn Array>,
    name: String,
}

impl Column {
    pub fn new(array: Arc<dyn Array>, name: String) -> Self {
        Self { array, name }
    }
}

#[pymethods]
impl Batch {
    fn columns(&self) -> Vec<Column> {
        self.data
            .columns()
            .iter()
            .zip(self.data.schema().fields())
            .map(|(array, field)| Column::new(array.clone(), field.name().clone()))
            .collect()
    }

    fn column(&self, name: &str) -> PyResult<Column> {
        let array = self
            .data
            .column_by_name(name)
            .ok_or_else(|| PyKeyError::new_err(format!("no column '{name}'")))?;
        Ok(Column::new(array.clone(), name.to_string()))
    }

    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        _requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let iterator = std::iter::once(Ok(self.data.clone()));
        let reader = RecordBatchIterator::new(iterator, self.data.schema());
        let stream = FFI_ArrowArrayStream::new(Box::new(reader));
        PyCapsule::new(py, stream, Some(c"arrow_array_stream".to_owned()))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __len__(&self) -> usize {
        self.data.num_rows()
    }
}

#[pymethods]
impl Column {
    /// Arrow `PyCapsule` Interface.
    /// Returns `(schema_capsule, array_capsule)` per the Arrow specification.
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        _requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let (ffi_array, ffi_schema) = ffi::to_ffi(&self.array.to_data())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let schema_cap = PyCapsule::new(py, ffi_schema, Some(c"arrow_schema".to_owned()))?;
        let array_cap = PyCapsule::new(py, ffi_array, Some(c"arrow_array".to_owned()))?;

        Ok((schema_cap, array_cap))
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
}
