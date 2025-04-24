use std::path::PathBuf;

use super::context::PyAlgoContext;
use crate::{
    error::GramsError,
    helper::get_progress_bar_style,
    models::{db::RemoteGramsDB, AlgoContext, LocalGramsDB, Table},
};
use indicatif::{ParallelProgressIterator, ProgressFinish};
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::error::into_pyerr;
use kgdata_core::{
    db::Map,
    python::models::{PyEntityMetadata, PyProperty},
};

// This is okay to use unsync once cell due to the GIL from Python
static DB_INSTANCE: OnceCell<Py<PyLocalGramsDB>> = OnceCell::new();
static REMOTE_DB_INSTANCE: OnceCell<Py<PyRemoteGramsDB>> = OnceCell::new();

#[pyclass(module = "gp_core.models", name = "LocalGramsDB", subclass)]
pub struct PyLocalGramsDB(pub LocalGramsDB);

#[pymethods]
impl PyLocalGramsDB {
    #[new]
    pub fn pynew(datadir: &str) -> PyResult<Self> {
        Ok(Self(LocalGramsDB::new(datadir).map_err(into_pyerr)?))
    }

    #[staticmethod]
    pub fn init(py: Python<'_>, datadir: &str) -> PyResult<()> {
        if let Some(db) = DB_INSTANCE.get() {
            if !(&db.borrow(py)).0 .0.datadir.as_os_str().eq(datadir) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "The database is already initialized with a different data directory. Call deinit first.",
                ));
            }
        } else {
            DB_INSTANCE
                .set(Py::new(py, Self::pynew(datadir)?)?)
                .unwrap();
        }

        Ok(())
    }

    #[staticmethod]
    pub fn get_instance(py: Python<'_>) -> PyResult<Py<Self>> {
        Ok(DB_INSTANCE
            .get()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "The database is not initialized. Call init first.",
                )
            })?
            .clone_ref(py))
    }

    #[pyo3(name = "get_algo_context")]
    pub fn get_py_algo_context(
        &self,
        table: &Table,
        n_hop: usize,
        parallel: bool,
    ) -> PyResult<PyAlgoContext> {
        Ok(PyAlgoContext(
            self.0
                .get_algo_context(table, n_hop, parallel)
                .map_err(into_pyerr)?,
        ))
    }

    #[pyo3(name = "get_algo_contexts")]
    pub fn get_py_algo_contexts(
        &self,
        py: Python<'_>,
        pytables: Vec<Py<Table>>,
        n_hop: usize,
        verbose: bool,
    ) -> PyResult<Vec<PyAlgoContext>> {
        let reftables = pytables.iter().map(|x| x.borrow(py)).collect::<Vec<_>>();
        let tables = reftables.iter().map(|x| &**x).collect::<Vec<_>>();

        let out = if verbose {
            tables
                .into_par_iter()
                .progress_with_style(get_progress_bar_style("get algorithm context"))
                .with_finish(ProgressFinish::AndLeave)
                .map(|table| self.0.get_algo_context(table, n_hop, true))
                .collect::<Result<Vec<_>, GramsError>>()
        } else {
            tables
                .into_par_iter()
                .map(|table| self.0.get_algo_context(table, n_hop, true))
                .collect::<Result<Vec<_>, GramsError>>()
        };

        Ok(out
            .map_err(into_pyerr)?
            .into_iter()
            .map(|x| PyAlgoContext(x))
            .collect::<Vec<_>>())
    }

    pub fn get_property(&self, id: &str) -> PyResult<PyProperty> {
        if let Some(prop) = self.0 .0.props.get(id).map_err(into_pyerr)? {
            Ok(PyProperty(prop))
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Property {} not found",
                id
            )))
        }
    }

    pub fn has_entity(&self, id: &str) -> PyResult<bool> {
        self.0
             .0
            .entity_pagerank
            .contains_key(id)
            .map_err(into_pyerr)
    }

    pub fn get_redirected_entity_id(&self, id: &str) -> PyResult<Option<String>> {
        self.0 .0.entity_redirection.get(id).map_err(into_pyerr)
    }

    pub fn get_entity_metadata(&self, id: &str) -> PyResult<PyEntityMetadata> {
        if let Some(ent) = self.0 .0.entity_metadata.get(id).map_err(into_pyerr)? {
            Ok(PyEntityMetadata(ent))
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Entity {} not found",
                id
            )))
        }
    }

    pub fn get_entity_pagerank(&self, id: &str) -> PyResult<f64> {
        if let Some(score) = self.0 .0.entity_pagerank.get(id).map_err(into_pyerr)? {
            Ok(score)
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Entity {} not found",
                id
            )))
        }
    }
}

#[pyclass(module = "gp_core.models", name = "RemoteGramsDB", subclass)]
pub struct PyRemoteGramsDB(pub RemoteGramsDB);

#[pyclass(module = "gp_core.models", name = "GramsDB", subclass)]
pub struct PyGramsDBTypeAliasMarker;

#[pymethods]
impl PyRemoteGramsDB {
    #[new]
    pub fn pynew(
        datadir: &str,
        entity_urls: Vec<String>,
        entity_metadata_urls: Vec<String>,
        entity_batch_size: usize,
        entity_metadata_batch_size: usize,
    ) -> PyResult<Self> {
        Ok(Self(
            RemoteGramsDB::new(
                datadir,
                &entity_urls,
                &entity_metadata_urls,
                entity_batch_size,
                entity_metadata_batch_size,
            )
            .map_err(into_pyerr)?,
        ))
    }

    #[staticmethod]
    pub fn init(
        py: Python<'_>,
        datadir: &str,
        entity_urls: Vec<String>,
        entity_metadata_urls: Vec<String>,
        entity_batch_size: usize,
        entity_metadata_batch_size: usize,
    ) -> PyResult<()> {
        if let Some(db) = REMOTE_DB_INSTANCE.get() {
            if !(&db.borrow(py)).0 .0.datadir.as_os_str().eq(datadir) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "The database is already initialized with a different data directory. Call deinit first.",
                ));
            }
        } else {
            REMOTE_DB_INSTANCE
                .set(Py::new(
                    py,
                    Self::pynew(
                        datadir,
                        entity_urls,
                        entity_metadata_urls,
                        entity_batch_size,
                        entity_metadata_batch_size,
                    )?,
                )?)
                .unwrap();
        }

        Ok(())
    }

    #[staticmethod]
    pub fn get_instance(py: Python<'_>) -> PyResult<Py<Self>> {
        Ok(REMOTE_DB_INSTANCE
            .get()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "The database is not initialized. Call init first.",
                )
            })?
            .clone_ref(py))
    }

    #[pyo3(name = "get_algo_context")]
    pub fn get_py_algo_context(
        &self,
        table: &Table,
        n_hop: usize,
        parallel: bool,
    ) -> PyResult<PyAlgoContext> {
        Ok(PyAlgoContext(
            self.0
                .get_algo_context(table, n_hop, parallel)
                .map_err(into_pyerr)?,
        ))
    }

    #[pyo3(name = "get_algo_contexts")]
    pub fn get_py_algo_contexts(
        &self,
        py: Python<'_>,
        pytables: Vec<Py<Table>>,
        n_hop: usize,
        verbose: bool,
    ) -> PyResult<Vec<PyAlgoContext>> {
        let reftables = pytables.iter().map(|x| x.borrow(py)).collect::<Vec<_>>();
        let tables = reftables.iter().map(|x| &**x).collect::<Vec<_>>();

        let out = if verbose {
            tables
                .into_par_iter()
                .progress_with_style(get_progress_bar_style("create algorithm context"))
                .with_finish(ProgressFinish::AndLeave)
                .map(|table| self.0.get_algo_context(table, n_hop, true))
                .collect::<Result<Vec<_>, GramsError>>()
        } else {
            tables
                .into_par_iter()
                .map(|table| self.0.get_algo_context(table, n_hop, true))
                .collect::<Result<Vec<_>, GramsError>>()
        };

        Ok(out
            .map_err(into_pyerr)?
            .into_iter()
            .map(|x| PyAlgoContext(x))
            .collect::<Vec<_>>())
    }

    pub fn get_property(&self, id: &str) -> PyResult<PyProperty> {
        if let Some(prop) = self.0 .0.props.get(id).map_err(into_pyerr)? {
            Ok(PyProperty(prop))
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Property {} not found",
                id
            )))
        }
    }

    pub fn has_entity(&self, id: &str) -> PyResult<bool> {
        self.0
             .0
            .entity_pagerank
            .contains_key(id)
            .map_err(into_pyerr)
    }

    pub fn get_redirected_entity_id(&self, id: &str) -> PyResult<Option<String>> {
        self.0 .0.entity_redirection.get(id).map_err(into_pyerr)
    }

    pub fn get_entity_metadata(&self, id: &str) -> PyResult<PyEntityMetadata> {
        if let Some(ent) = self.0 .0.entity_metadata.get(id).map_err(into_pyerr)? {
            Ok(PyEntityMetadata(ent))
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Entity {} not found",
                id
            )))
        }
    }

    pub fn get_entity_pagerank(&self, id: &str) -> PyResult<f64> {
        if let Some(score) = self.0 .0.entity_pagerank.get(id).map_err(into_pyerr)? {
            Ok(score)
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Entity {} not found",
                id
            )))
        }
    }
}

#[derive(FromPyObject)]
pub enum PyGramsDB<'t> {
    LocalGramsDB(PyRef<'t, PyLocalGramsDB>),
    RemoteGramsDB(PyRef<'t, PyRemoteGramsDB>),
}

pub enum GramsDB<'t> {
    LocalGramsDB(&'t LocalGramsDB),
    RemoteGramsDB(&'t RemoteGramsDB),
}

impl<'t> PyGramsDB<'t> {
    pub fn deref<'t1>(&'t1 self) -> GramsDB<'t1> {
        match self {
            PyGramsDB::LocalGramsDB(db) => GramsDB::LocalGramsDB(&db.0),
            PyGramsDB::RemoteGramsDB(db) => GramsDB::RemoteGramsDB(&db.0),
        }
    }
}

impl<'t> GramsDB<'t> {
    pub fn get_algo_context(
        &self,
        table: &Table,
        n_hop: usize,
        parallel: bool,
    ) -> Result<AlgoContext, GramsError> {
        match self {
            GramsDB::LocalGramsDB(db) => db.get_algo_context(table, n_hop, parallel),
            GramsDB::RemoteGramsDB(db) => db.get_algo_context(table, n_hop, parallel),
        }
    }

    pub fn get_data_dir(&self) -> &PathBuf {
        match self {
            GramsDB::LocalGramsDB(db) => &db.0.datadir,
            GramsDB::RemoteGramsDB(db) => &db.0.datadir,
        }
    }
}
