use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::Error;

pub struct Batch {
    data: RecordBatch,
}

impl Batch {
    pub fn new(record_batch: RecordBatch) -> Self {
        Self { data: record_batch }
    }

    /// Export the batch to Python as `dict[str, np.ndarray]`
    /// Supported dtypes: bool, i32, i64, f32, f64, utf8 (object array)
    pub fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let schema = self.data.schema();
        let arrays = self.data.columns();

        for (field, array) in schema.fields().iter().zip(arrays) {
            let np_array = array_to_numpy(py, array, field.name())?;
            dict.set_item(field.name(), np_array)?;
        }

        Ok(dict)
    }
}

fn array_to_numpy<'py>(py: Python<'py>, array: &dyn Array, name: &str) -> PyResult<Py<PyAny>> {
    match array.data_type() {
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>().unwrap();
            Ok(slice_to_numpy_f32(py, a).into_any().unbind())
        }
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>().unwrap();
            Ok(slice_to_numpy_f64(py, a).into_any().unbind())
        }
        DataType::Int32 => {
            let a = array.as_any().downcast_ref::<Int32Array>().unwrap();
            Ok(slice_to_numpy_i32(py, a).into_any().unbind())
        }
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok(slice_to_numpy_i64(py, a).into_any().unbind())
        }
        DataType::Boolean => {
            let a = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            // Boolean arrays are bit-packed in Arrow; collect to Vec<bool> then to numpy.
            let v: Vec<bool> = (0..a.len()).map(|i| a.value(i)).collect();
            Ok(v.to_pyarray(py).into_any().unbind())
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().unwrap();
            let list = PyList::new(py, (0..a.len()).map(|i| a.value(i)))?;
            // Convert to numpy object array via numpy.asarray().
            let numpy = py.import("numpy")?;
            let obj_arr = numpy.call_method1("asarray", (list,))?;
            Ok(obj_arr.unbind())
        }
        dt => Err(Error::UnsupportedDtype {
            name: name.to_owned(),
            dtype: dt.clone(),
        }
        .into()),
    }
}

// Arrow primitive arrays can have a non-zero offset when they are slices of
// a larger buffer. We must respect that offset when copying into numpy.
fn slice_to_numpy_f32<'py>(py: Python<'py>, a: &Float32Array) -> Bound<'py, PyArray1<f32>> {
    let buf = a.values();
    let s = &buf.as_ref()[a.offset()..a.offset() + a.len()];
    PyArray1::from_slice(py, s)
}

fn slice_to_numpy_f64<'py>(py: Python<'py>, a: &Float64Array) -> Bound<'py, PyArray1<f64>> {
    let buf = a.values();
    let s = &buf.as_ref()[a.offset()..a.offset() + a.len()];
    PyArray1::from_slice(py, s)
}

fn slice_to_numpy_i32<'py>(py: Python<'py>, a: &Int32Array) -> Bound<'py, PyArray1<i32>> {
    let buf = a.values();
    let s = &buf.as_ref()[a.offset()..a.offset() + a.len()];
    PyArray1::from_slice(py, s)
}

fn slice_to_numpy_i64<'py>(py: Python<'py>, a: &Int64Array) -> Bound<'py, PyArray1<i64>> {
    let buf = a.values();
    let s = &buf.as_ref()[a.offset()..a.offset() + a.len()];
    PyArray1::from_slice(py, s)
}
