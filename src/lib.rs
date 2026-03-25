mod batch;
mod dataset;
mod error;
mod loader;
mod reader;

use pyo3::prelude::*;

use dataset::Dataset;
use loader::DataLoader;

#[pymodule]
fn _parqstream(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Dataset>()?;
    m.add_class::<DataLoader>()?;
    Ok(())
}
