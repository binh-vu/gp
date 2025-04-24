use kgdata_core::python::models::entity::StatementView;

use pyo3::prelude::*;

use crate::models::AlgoContext;

#[pyclass(module = "gp_core.models", name = "AlgoContext")]
pub struct PyAlgoContext(pub AlgoContext);

#[pymethods]
impl PyAlgoContext {
    pub fn get_entity_statement(
        &self,
        entity_id: &str,
        prop: &str,
        stmt_index: usize,
    ) -> PyResult<StatementView> {
        let stmt = self
            .0
            .entities
            .get(entity_id)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Entity {} not found",
                    entity_id
                ))
            })?
            .props
            .get(prop)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Property {} not found",
                    prop
                ))
            })?
            .get(stmt_index)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Statement index {} out of bounds",
                    stmt_index
                ))
            })?;
        Ok(StatementView::new(stmt))
    }
}
