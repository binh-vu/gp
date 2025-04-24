use std::ops::{Deref, DerefMut};

use crate::{
    algorithms::{
        candidate_graph_builder::create_candidate_graph,
        data_graph_builder::create_data_graph,
        features::{
            feature_store::{FeatureStore, FeatureStoreCache},
            rel_features::{
                self, detect_contradicted_info::SimpleMissingInfoDetector,
                funcdep::SimpleFunctionalDependencyDetector,
            },
        },
        kginference::KGInference,
    },
    error::{into_pyerr, GramsError},
    helper::get_progress_bar_style,
    libs::{
        index::IndexTraversal,
        literal_matchers::{LiteralMatcher, LiteralMatcherConfig},
    },
    models::{cangraph as cgraph, db::BaseGramsDB, AlgoContext, Table, TableCells},
    python::models::{
        cangraph::PyCGNode,
        db::{GramsDB, PyGramsDB},
        PyAlgoContext,
    },
};
use hashbrown::{HashMap, HashSet};
use indicatif::{ParallelProgressIterator, ProgressFinish};
use kgdata_core::{
    db::Map,
    models::{Entity, EntityMetadata},
};
use polars::prelude::DataFrame;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass(module = "gp_core.algorithms", name = "CanGraphExtractorConfig")]
#[derive(Serialize, Deserialize)]
pub struct CanGraphExtractorCfg {
    literal_matcher_config: LiteralMatcherConfig,

    ignored_columns: Vec<usize>,
    ignored_props: HashSet<String>,
    allow_same_ent_search: bool,
    allow_ent_matching: bool,
    use_context: bool,
    add_missing_property: bool,
    // `run_subproperty_inference` - whether to run subproperty inference to complete missing links
    pub run_subproperty_inference: bool,
    // `run_transitive_inference` - whether to run transitive inference to complete missing links
    pub run_transitive_inference: bool,
    deterministic_order: bool,

    correct_entity_threshold: f64,

    validate: bool,

    pub n_hop: usize,
}

#[pyclass(module = "gp_core.algorithms", name = "CanGraphExtractedResult")]
pub struct PyCanGraphExtractedResult {
    #[pyo3(get, set)]
    nodes: Py<PyList>,
    #[pyo3(get, set)]
    edges: Py<PyAny>,
    #[pyo3(get, set)]
    edgedf: PyObject,
    // #[pyo3(get, set)]
    // node_features: PyObject,
}

pub struct CanGraphExtractedResult {
    cg: cgraph::CGraph,
    props: HashMap<String, u32>,
    edgedf: DataFrame,
}

/// Extract candidate graph of semantic description of a table
#[pyfunction(name = "par_extract_cangraphs")]
#[pyo3(signature = (
    tables,
    table_cells,
    db,
    cfg,
    algocontexts = None,
    verbose = false
))]
pub fn py_par_extract_cangraph<'t>(
    py: Python<'t>,
    tables: Vec<PyRef<'t, Table>>,
    table_cells: Vec<PyRef<'t, TableCells>>,
    db: PyGramsDB<'t>,
    cfg: &CanGraphExtractorCfg,
    algocontexts: Option<Vec<PyRefMut<'t, PyAlgoContext>>>,
    verbose: bool,
) -> PyResult<Vec<PyCanGraphExtractedResult>> {
    let deref_tables = tables.iter().map(|x| x.deref()).collect::<Vec<_>>();
    let deref_cells = table_cells
        .iter()
        .map(|x| x.deref())
        .collect::<Vec<&TableCells>>();
    let deref_db = db.deref();

    let results = match algocontexts {
        None => {
            let it = (0..deref_tables.len()).into_par_iter().map(|i| {
                let table = deref_tables[i];
                let cells = deref_cells[i];
                wrap_extract_cangraph(&table, &cells, &deref_db, None, cfg, true)
            });

            if verbose {
                it.progress_with_style(get_progress_bar_style("extract candidate graph"))
                    .with_finish(ProgressFinish::AndLeave)
                    .collect::<Result<Vec<_>, GramsError>>()
            } else {
                it.collect::<Result<Vec<_>, GramsError>>()
            }
        }
        Some(mut refcontexts) => {
            let contexts = refcontexts
                .iter_mut()
                .map(|x| x.deref_mut())
                .collect::<Vec<_>>();
            let it = contexts.into_par_iter().enumerate().map(|(i, context)| {
                let table = deref_tables[i];
                let cells = deref_cells[i];
                wrap_extract_cangraph(&table, &cells, &deref_db, Some(&mut context.0), cfg, true)
            });
            if verbose {
                it.progress_with_style(get_progress_bar_style("extract candidate graph"))
                    .with_finish(ProgressFinish::AndLeave)
                    .collect::<Result<Vec<_>, GramsError>>()
            } else {
                it.collect::<Result<Vec<_>, GramsError>>()
            }
        }
    };

    results
        .map_err(into_pyerr)?
        .into_iter()
        .map(|res| res.to_python(py))
        .collect::<PyResult<Vec<_>>>()
}

#[pyfunction(name = "extract_cangraph")]
/// Extract candidate graph of semantic description of a table
#[pyo3(signature = (table, cells, refdb, cfg, algocontext, parallel = false))]
pub fn py_extract_cangraph<'t>(
    py: Python<'t>,
    table: &Table,
    cells: &TableCells,
    refdb: PyGramsDB<'t>,
    cfg: &CanGraphExtractorCfg,
    algocontext: Option<PyRefMut<'t, PyAlgoContext>>,
    parallel: bool,
) -> PyResult<PyCanGraphExtractedResult> {
    let db = refdb.deref();

    match algocontext {
        None => wrap_extract_cangraph(table, cells, &db, None, cfg, parallel)
            .map_err(into_pyerr)?
            .to_python(py),
        Some(mut algocontext) => wrap_extract_cangraph(
            table,
            cells,
            &db,
            Some(&mut algocontext.deref_mut().0),
            cfg,
            parallel,
        )
        .map_err(into_pyerr)?
        .to_python(py),
    }
}

#[inline]
pub fn wrap_extract_cangraph(
    table: &Table,
    cells: &TableCells,
    db: &GramsDB,
    prebuilt_algocontext: Option<&mut AlgoContext>,
    cfg: &CanGraphExtractorCfg,
    parallel: bool,
) -> Result<CanGraphExtractedResult, GramsError> {
    match db {
        GramsDB::LocalGramsDB(db) => {
            extract_cangraph(table, cells, db, prebuilt_algocontext, cfg, parallel)
        }
        GramsDB::RemoteGramsDB(db) => {
            extract_cangraph(table, cells, db, prebuilt_algocontext, cfg, parallel)
        }
    }
}

pub fn extract_cangraph<ED: Map<String, Entity>, EMD: Map<String, EntityMetadata>>(
    table: &Table,
    cells: &TableCells,
    db: &BaseGramsDB<ED, EMD>,
    prebuilt_algocontext: Option<&mut AlgoContext>,
    cfg: &CanGraphExtractorCfg,
    parallel: bool,
) -> Result<CanGraphExtractedResult, GramsError> {
    let mut rebuilt_algocontext;
    let mut algocontext = match prebuilt_algocontext {
        None => {
            rebuilt_algocontext = db
                .get_algo_context(table, cfg.n_hop, parallel)
                .map_err(into_pyerr)?;
            &mut rebuilt_algocontext
        }
        Some(algocontext) => algocontext,
    };

    let kginf = if cfg.run_subproperty_inference || cfg.run_transitive_inference {
        let mut kginf = KGInference::new();
        if cfg.run_subproperty_inference {
            kginf.infer_subproperty();
        }
        if cfg.run_transitive_inference {
            kginf.infer_transitive_property(&mut algocontext, parallel)?;
        }

        Some(kginf)
    } else {
        None
    };

    algocontext.init_object_1hop_index(cfg.allow_ent_matching, parallel);

    let mut traversal = IndexTraversal::from_context(algocontext);
    let literal_matcher = LiteralMatcher::new(&cfg.literal_matcher_config).map_err(into_pyerr)?;

    let dg = create_data_graph(
        &table,
        &cells.0,
        &algocontext,
        kginf.as_ref(),
        &mut traversal,
        &literal_matcher,
        &cfg.ignored_columns,
        &cfg.ignored_props,
        cfg.allow_same_ent_search,
        cfg.allow_ent_matching,
        cfg.use_context,
        cfg.add_missing_property,
        cfg.deterministic_order,
        parallel,
    )
    .map_err(into_pyerr)?;
    let cg = create_candidate_graph(&table, &dg).map_err(into_pyerr)?;

    if cfg.validate {
        if !dg.is_valid() {
            return Err(GramsError::PyErr(PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >("Data graph is invalid")));
        }
        if !cg.is_valid(&dg) {
            return Err(GramsError::PyErr(PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(
                "Candidate graph is invalid"
            )));
        }
    }

    let mut featstore = FeatureStore::new(table, db, algocontext, &cg, &dg, parallel)?;
    let mut cache = FeatureStoreCache::new(&featstore, db.get_data_dir()).map_err(into_pyerr)?;
    let mut missing_info_detector = SimpleMissingInfoDetector {};
    let mut func_dep_detector = SimpleFunctionalDependencyDetector::new(table, cells);
    let (props, edgedf) = rel_features::extract_rel_features(
        &mut featstore,
        &mut cache,
        &mut missing_info_detector,
        &mut func_dep_detector,
        cfg.correct_entity_threshold,
    )?;

    Ok(CanGraphExtractedResult { cg, props, edgedf })
}

#[pymethods]
impl CanGraphExtractorCfg {
    #[new]
    #[pyo3(signature = (
        literal_matcher_config,
        ignored_columns,
        ignored_props,
        allow_same_ent_search,
        allow_ent_matching,
        use_context,
        add_missing_property,
        run_subproperty_inference,
        run_transitive_inference,
        deterministic_order,
        correct_entity_threshold,
        validate,
        n_hop,
    ))]
    pub fn new(
        literal_matcher_config: LiteralMatcherConfig,

        ignored_columns: Vec<usize>,
        ignored_props: HashSet<String>,
        allow_same_ent_search: bool,
        allow_ent_matching: bool,
        use_context: bool,
        add_missing_property: bool,
        run_subproperty_inference: bool,
        run_transitive_inference: bool,
        deterministic_order: bool,

        correct_entity_threshold: f64,

        validate: bool,

        n_hop: usize,
    ) -> Self {
        Self {
            literal_matcher_config,

            ignored_columns,
            ignored_props,
            allow_same_ent_search,
            allow_ent_matching,
            use_context,
            add_missing_property,
            run_subproperty_inference,
            run_transitive_inference,
            deterministic_order,

            correct_entity_threshold,

            validate,

            n_hop,
        }
    }

    pub fn save(&self, outfile: &str) -> PyResult<()> {
        crate::helper::save_object(self, outfile).map_err(into_pyerr)
    }
}

#[pymethods]
impl PyCanGraphExtractedResult {
    #[new]
    pub fn new(py: Python<'_>) -> Self {
        PyCanGraphExtractedResult {
            nodes: PyList::empty(py).into_py(py),
            edges: py.None(),
            edgedf: py.None(),
        }
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let out = PyDict::new(py);
        out.set_item("nodes", self.nodes.as_ref(py))?;
        out.set_item("edges", self.edges.as_ref(py))?;
        out.set_item("edgedf", self.edgedf.as_ref(py))?;
        Ok(out)
    }

    pub fn __setstate__(&mut self, py: Python, state: &PyDict) -> PyResult<()> {
        self.nodes = state
            .get_item("nodes")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing key 'nodes'"))?
            .downcast::<PyList>()?
            .into_py(py);

        self.edges = state
            .get_item("edges")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing key 'edges'"))?
            .into_py(py);

        self.edgedf = state
            .get_item("edgedf")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing key 'edgedf'"))?
            .into_py(py);

        Ok(())
    }
}

impl CanGraphExtractedResult {
    pub fn to_python(self, py: Python<'_>) -> PyResult<PyCanGraphExtractedResult> {
        let nodes = self
            .cg
            .graph
            .iter_nodes()
            .enumerate()
            .map(|(i, u)| {
                assert_eq!(u.id().0, i);
                Py::new(py, PyCGNode(u.clone()))
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut edges = vec![""; self.props.len()];
        for (prop, idx) in &self.props {
            edges[*idx as usize] = prop;
        }

        Ok(PyCanGraphExtractedResult {
            nodes: PyList::new(py, nodes).into(),
            edges: edges.into_py(py),
            edgedf: PyDataFrame(self.edgedf).into_py(py),
        })
    }
}
