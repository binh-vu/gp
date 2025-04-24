pub mod cangraph;
pub mod context;
pub mod db;
pub mod table;

use self::{
    cangraph::PyCGNode,
    db::{PyGramsDBTypeAliasMarker, PyLocalGramsDB, PyRemoteGramsDB},
};
use crate::models::{
    table::{CandidateEntityId, Column, Context, EntityId, Link, Table},
    TableCells,
};
use pyo3::prelude::*;

pub use self::context::PyAlgoContext;

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "models")?;

    m.add_submodule(submodule)?;

    submodule.add_class::<Table>()?;
    submodule.add_class::<TableCells>()?;
    submodule.add_class::<Context>()?;
    submodule.add_class::<Column>()?;
    submodule.add_class::<Link>()?;
    submodule.add_class::<CandidateEntityId>()?;
    submodule.add_class::<EntityId>()?;
    submodule.add_class::<PyLocalGramsDB>()?;
    submodule.add_class::<PyRemoteGramsDB>()?;
    submodule.add_class::<PyGramsDBTypeAliasMarker>()?;
    submodule.add_class::<PyAlgoContext>()?;
    submodule.add_class::<PyCGNode>()?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("gp_core.models", submodule)?;

    Ok(())
}
