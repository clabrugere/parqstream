mod batch;
mod buffer;
mod checkpoint;
mod dataloader;
mod dataset;
mod error;
mod pipeline;

use pyo3::prelude::*;

use batch::{Batch, Column};
use checkpoint::Checkpoint;
use dataloader::DataLoader;
use dataset::Dataset;

#[pymodule]
fn _parqstream(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Dataset>()?;
    m.add_class::<DataLoader>()?;
    m.add_class::<Checkpoint>()?;
    m.add_class::<Batch>()?;
    m.add_class::<Column>()?;
    Ok(())
}
