pub mod algorithms;
pub mod error;
pub mod models;

pub mod helper;
pub mod libs;
pub mod python;

use pyo3::{prelude::*, types::PyList};

/// Initialize env logger so we can control Rust logging from outside.
#[pyfunction]
pub fn init_env_logger() -> PyResult<()> {
    env_logger::init();
    Ok(())
}

#[pymodule]
fn gp_core(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.setattr("__path__", PyList::empty(py))?;

    m.add_function(wrap_pyfunction!(init_env_logger, m)?)?;

    python::models::register(py, m)?;
    python::literal_matchers::register(py, m)?;
    python::algorithms::register(py, m)?;

    Ok(())
}
