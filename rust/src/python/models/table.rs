use crate::error::into_pyerr;
use crate::{
    libs::literal_matchers::ParsedTextRepr,
    models::table::{CandidateEntityId, Column, Context, EntityId, Link, Table, TableCells},
};
use postcard::{from_bytes, to_allocvec};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pymethods]
impl Table {
    #[new]
    pub fn new(
        id: String,
        links: Vec<Vec<Vec<Link>>>,
        columns: Vec<Column>,
        context: Context,
    ) -> Self {
        Self {
            id,
            links,
            columns,
            context,
        }
    }

    pub fn get_links(&self, row: usize, col: usize) -> PyResult<Vec<Link>> {
        Ok(self
            .links
            .get(row)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Row index {} out of bounds",
                    row
                ))
            })?
            .get(col)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Column index {} out of bounds",
                    col
                ))
            })?
            .clone())
    }

    pub fn save(&self, outfile: &str) -> PyResult<()> {
        crate::helper::save_object(self, outfile).map_err(into_pyerr)
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        let out = to_allocvec(&self).map_err(into_pyerr)?;
        Ok(PyBytes::new(py, &out))
    }

    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = from_bytes::<Table>(state.as_bytes()).map_err(into_pyerr)?;
        Ok(())
    }

    pub fn __getnewargs__(&self) -> PyResult<(String, Vec<Vec<Vec<Link>>>, Vec<Column>, Context)> {
        Ok((
            String::new(),
            Vec::new(),
            Vec::new(),
            Context::new(None, None, Vec::new()),
        ))
    }
}

#[pymethods]
impl Context {
    #[new]
    #[pyo3(signature = (page_title, page_url, page_entities))]
    pub fn new(
        page_title: Option<String>,
        page_url: Option<String>,
        page_entities: Vec<CandidateEntityId>,
    ) -> Self {
        Self {
            page_title,
            page_url,
            page_entities,
        }
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        let out = to_allocvec(&self).map_err(into_pyerr)?;
        Ok(PyBytes::new(py, &out))
    }

    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = from_bytes::<Context>(state.as_bytes()).map_err(into_pyerr)?;
        Ok(())
    }

    pub fn __getnewargs__(
        &self,
    ) -> PyResult<(Option<String>, Option<String>, Vec<CandidateEntityId>)> {
        Ok((None, None, Vec::new()))
    }
}

#[pymethods]
impl Link {
    #[new]
    #[pyo3(signature = (start, end, url, entities, candidates))]
    pub fn new(
        start: usize,
        end: usize,
        url: Option<String>,
        entities: Vec<EntityId>,
        candidates: Vec<CandidateEntityId>,
    ) -> Self {
        Self {
            start,
            end,
            url,
            entities,
            candidates,
        }
    }
}

#[pymethods]
impl EntityId {
    #[new]
    fn new(id: String) -> Self {
        Self(id)
    }

    #[getter]
    fn id(&self) -> &str {
        &self.0
    }
}

#[pymethods]
impl CandidateEntityId {
    #[new]
    pub fn new(id: EntityId, probability: f64) -> Self {
        Self { id, probability }
    }
}

#[pymethods]
impl Column {
    #[new]
    #[pyo3(signature = (index, name, values))]
    pub fn new(index: usize, name: Option<String>, values: Vec<String>) -> Self {
        Self {
            index,
            name,
            values,
        }
    }
}

#[pymethods]
impl TableCells {
    #[new]
    pub fn new(cells: Vec<Vec<ParsedTextRepr>>) -> Self {
        Self(cells)
    }

    pub fn save(&self, outfile: &str) -> PyResult<()> {
        crate::helper::save_object(self, outfile).map_err(into_pyerr)
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        let out = to_allocvec(&self).map_err(into_pyerr)?;
        Ok(PyBytes::new(py, &out))
    }

    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = from_bytes::<TableCells>(state.as_bytes()).map_err(into_pyerr)?;
        Ok(())
    }

    pub fn __getnewargs__(&self) -> PyResult<(Vec<Vec<ParsedTextRepr>>,)> {
        Ok((Vec::new(),))
    }
}
