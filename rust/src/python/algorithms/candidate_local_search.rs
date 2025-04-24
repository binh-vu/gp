use std::ops::{Deref, DerefMut};

use crate::algorithms::candidate_local_search::candidate_local_search;
use crate::error::into_pyerr;
use crate::helper::get_progress_bar_style;
use crate::libs::index::IndexTraversal;
use crate::models::{AlgoContext, Table};
use crate::python::models::db::PyGramsDB;
use crate::python::models::PyAlgoContext;
use indicatif::{ParallelProgressIterator, ProgressFinish};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass(module = "gp_core.algorithms", name = "CandidateLocalSearchConfig")]
pub struct CandidateLocalSearchConfig {
    pub strsim: String,
    pub threshold: f64,
    pub use_column_name: bool,
    pub use_language: Option<String>,
    pub search_all_columns: bool,
}

#[pymethods]
impl CandidateLocalSearchConfig {
    #[new]
    #[pyo3(signature = (strsim, threshold, use_column_name, use_language, search_all_columns))]
    pub fn new(
        strsim: String,
        threshold: f64,
        use_column_name: bool,
        use_language: Option<String>,
        search_all_columns: bool,
    ) -> Self {
        Self {
            strsim,
            threshold,
            use_column_name,
            use_language,
            search_all_columns,
        }
    }
}

#[pyfunction(name = "candidate_local_search")]
pub fn py_candidate_local_search<'t>(
    table: &Table,
    context: &'t mut PyAlgoContext,
    cfg: &CandidateLocalSearchConfig,
    parallel: bool,
) -> PyResult<Table> {
    candidate_local_search_common(table, &mut context.0, cfg, parallel)
}

#[pyfunction(name = "par_candidate_local_search")]
#[pyo3(signature = (
    refdb,
    reftables,
    refcontexts,
    cfg
))]
pub fn py_par_candidate_local_search<'t>(
    refdb: PyGramsDB<'t>,
    reftables: Vec<PyRef<'t, Table>>,
    refcontexts: Option<Vec<PyRefMut<'t, PyAlgoContext>>>,
    cfg: &CandidateLocalSearchConfig,
) -> PyResult<Vec<Table>> {
    let tables = reftables.iter().map(|x| x.deref()).collect::<Vec<_>>();
    let db = refdb.deref();

    match refcontexts {
        None => (0..tables.len())
            .into_par_iter()
            .map(|i| {
                let table = tables[i];
                let mut context = db.get_algo_context(table, 1, true).map_err(into_pyerr)?;
                candidate_local_search_common(table, &mut context, cfg, true)
            })
            .progress_with_style(get_progress_bar_style("candidate local search"))
            .with_finish(ProgressFinish::AndLeave)
            .collect::<PyResult<Vec<_>>>(),
        Some(mut refcontexts) => {
            let contexts = refcontexts
                .iter_mut()
                .map(|x| x.deref_mut())
                .collect::<Vec<_>>();
            contexts
                .into_par_iter()
                .enumerate()
                .map(|(i, context)| {
                    let table = tables[i];
                    candidate_local_search_common(table, &mut context.0, cfg, true)
                })
                .progress_with_style(get_progress_bar_style("candidate local search"))
                .with_finish(ProgressFinish::AndLeave)
                .collect::<PyResult<Vec<_>>>()
        }
    }
}

#[inline]
pub fn candidate_local_search_common<'t>(
    table: &Table,
    context: &'t mut AlgoContext,
    cfg: &CandidateLocalSearchConfig,
    parallel: bool,
) -> PyResult<Table> {
    context.init_object_1hop_index(true, false);
    let mut char_tokenizer = yass::CharacterTokenizer {};
    let mut traversal = IndexTraversal::from_context(&context);

    match cfg.strsim.as_str() {
        "levenshtein" => candidate_local_search(
            table,
            &mut traversal,
            &mut yass::SeqStrSim::new(&mut char_tokenizer, yass::Levenshtein::default())
                .map_err(into_pyerr)?,
            cfg.threshold,
            cfg.use_column_name,
            cfg.use_language.as_ref(),
            cfg.search_all_columns,
            parallel,
        )
        .map_err(into_pyerr),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid strsim",
        )),
    }
}
