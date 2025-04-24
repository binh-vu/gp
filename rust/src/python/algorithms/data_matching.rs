use std::ops::{Deref, DerefMut};

use crate::algorithms::data_matching::{
    CellNode, MatchedEntRel, MatchedQualifier, MatchedStatement,
};
use crate::algorithms::data_matching::{DataMatching, Node, PotentialRelationships};
use crate::error::into_pyerr;
use crate::helper::get_progress_bar_style;
use crate::libs::index::IndexTraversal;
use crate::libs::literal_matchers::PyLiteralMatcher;
use crate::models::{AlgoContext, Match, MatchMethod, Table, TableCells};
use crate::python::models::{db::PyGramsDB, PyAlgoContext};
use hashbrown::HashSet;
use indicatif::{ParallelProgressIterator, ProgressFinish};
use kgdata_core::pyo3helper::unsafe_update_view_lifetime_signature;
use kgdata_core::{pylist, pyview};
use postcard::{from_bytes, to_allocvec};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction(name = "matching")]
#[pyo3(signature = (
    table,
    table_cells,
    context,
    literal_matcher,
    ignored_columns,
    ignored_props,
    allow_same_ent_search = false,
    allow_ent_matching = true,
    use_context = true,
    deterministic_order = true,
    parallel = false
))]
pub fn matching(
    table: &Table,
    table_cells: &TableCells,
    context: &mut PyAlgoContext,
    literal_matcher: &PyLiteralMatcher,
    ignored_columns: Vec<usize>,
    ignored_props: Vec<String>,
    allow_same_ent_search: bool,
    allow_ent_matching: bool,
    use_context: bool,
    deterministic_order: bool,
    parallel: bool,
) -> PyResult<PyDataMatchesResult> {
    wrap_matching(
        table,
        table_cells,
        &mut context.0,
        literal_matcher,
        &ignored_columns,
        &HashSet::from_iter(ignored_props),
        allow_same_ent_search,
        allow_ent_matching,
        use_context,
        deterministic_order,
        parallel,
    )
}

#[pyfunction(name = "par_matching")]
#[pyo3(signature = (
    db,
    tables,
    table_cells,
    contexts,
    literal_matcher,
    ignored_columns,
    ignored_props,
    allow_same_ent_search = false,
    allow_ent_matching = true,
    use_context = true,
    deterministic_order = true,
    verbose = false,
))]
pub fn par_matching<'t>(
    db: PyGramsDB<'t>,
    tables: Vec<PyRef<'t, Table>>,
    table_cells: Vec<PyRef<'t, TableCells>>,
    contexts: Option<Vec<PyRefMut<'t, PyAlgoContext>>>,
    literal_matcher: &PyLiteralMatcher,
    ignored_columns: Vec<Vec<usize>>,
    ignored_props: Vec<String>,
    allow_same_ent_search: bool,
    allow_ent_matching: bool,
    use_context: bool,
    deterministic_order: bool,
    verbose: bool,
) -> PyResult<Vec<PyDataMatchesResult>> {
    let deref_tables = tables.iter().map(|x| x.deref()).collect::<Vec<_>>();
    let deref_cells = table_cells
        .iter()
        .map(|x| x.deref())
        .collect::<Vec<&TableCells>>();
    let deref_db = db.deref();
    let ignored_props_set = HashSet::from_iter(ignored_props);

    match contexts {
        None => {
            let it = (0..deref_tables.len()).into_par_iter().map(|i| {
                let table = deref_tables[i];
                let cells = deref_cells[i];

                let mut context = deref_db
                    .get_algo_context(table, 1, true)
                    .map_err(into_pyerr)?;
                wrap_matching(
                    table,
                    cells,
                    &mut context,
                    literal_matcher,
                    &ignored_columns[i],
                    &ignored_props_set,
                    allow_same_ent_search,
                    allow_ent_matching,
                    use_context,
                    deterministic_order,
                    true,
                )
            });

            if verbose {
                it.progress_with_style(get_progress_bar_style("data matching"))
                    .with_finish(ProgressFinish::AndLeave)
                    .collect::<PyResult<Vec<_>>>()
            } else {
                it.collect::<PyResult<Vec<_>>>()
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
                wrap_matching(
                    table,
                    cells,
                    &mut context.0,
                    literal_matcher,
                    &ignored_columns[i],
                    &ignored_props_set,
                    allow_same_ent_search,
                    allow_ent_matching,
                    use_context,
                    deterministic_order,
                    true,
                )
            });

            if verbose {
                it.progress_with_style(get_progress_bar_style("data matching"))
                    .with_finish(ProgressFinish::AndLeave)
                    .collect::<PyResult<Vec<_>>>()
            } else {
                it.collect::<PyResult<Vec<_>>>()
            }
        }
    }
}

#[inline]
pub fn wrap_matching(
    table: &Table,
    table_cells: &TableCells,
    context: &mut AlgoContext,
    literal_matcher: &PyLiteralMatcher,
    ignored_columns: &[usize],
    ignored_props: &HashSet<String>,
    allow_same_ent_search: bool,
    allow_ent_matching: bool,
    use_context: bool,
    deterministic_order: bool,
    parallel: bool,
) -> PyResult<PyDataMatchesResult> {
    context.init_object_1hop_index(allow_ent_matching, parallel);
    let mut traversal = IndexTraversal::from_context(&context);
    let (nodes, edges) = DataMatching::exec(
        table,
        &table_cells.0,
        &context,
        &mut traversal,
        &literal_matcher.0,
        ignored_columns,
        ignored_props,
        allow_same_ent_search,
        allow_ent_matching,
        use_context,
        deterministic_order,
        parallel,
    )
    .map_err(into_pyerr)?;
    Ok(PyDataMatchesResult { nodes, edges })
}

#[pyclass(
    module = "gp_core.algorithms.data_matching",
    name = "DataMatchesResult"
)]
#[derive(Debug)]
pub struct PyDataMatchesResult {
    pub nodes: Vec<Node>,
    pub edges: Vec<PotentialRelationships>,
}

#[pymethods]
impl PyDataMatchesResult {
    pub fn save(&self, file: &str) -> PyResult<()> {
        let out = to_allocvec(&(&self.nodes, &self.edges)).map_err(into_pyerr)?;
        std::fs::write(file, out).map_err(into_pyerr)
    }

    #[staticmethod]
    pub fn load(file: &str) -> PyResult<PyDataMatchesResult> {
        let (nodes, edges) = from_bytes::<(Vec<Node>, Vec<PotentialRelationships>)>(
            &std::fs::read(file).map_err(into_pyerr)?,
        )
        .map_err(into_pyerr)?;
        Ok(PyDataMatchesResult { nodes, edges })
    }

    pub fn get_n_nodes(&self) -> usize {
        return self.nodes.len();
    }

    pub fn is_cell_node(&self, idx: usize) -> bool {
        match self.nodes[idx] {
            Node::Cell(_) => true,
            _ => false,
        }
    }

    pub fn is_entity_node(&self, idx: usize) -> bool {
        match self.nodes[idx] {
            Node::Entity(_) => true,
            _ => false,
        }
    }

    pub fn get_cell_node(&self, idx: usize) -> PyResult<CellNode> {
        match &self.nodes[idx] {
            Node::Cell(cell) => Ok(cell.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("Not a cell node")),
        }
    }

    pub fn get_entity_node(&self, idx: usize) -> PyResult<&String> {
        match &self.nodes[idx] {
            Node::Entity(entity) => Ok(&entity.entity_id),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("Not an entity node")),
        }
    }

    pub fn edges(&self) -> potential_relationships_list_view::ListView {
        potential_relationships_list_view::ListView::new(&self.edges)
    }
}

pyview!(MatchView(module = "gp_core.models", name = "Match", cls = Match) {
    c(prob: f64),
    b(method: MatchMethod)
});

pyview!(MatchedQualifierView(module = "gp_core.algorithms", name = "MatchedQualifier", cls = MatchedQualifier) {
    b(qualifier: String),
    c(qualifier_index: usize),
    v(score: MatchView)
});
pyview!(MatchedStatementView(module = "gp_core.algorithms", name = "MatchedStatement", cls = MatchedStatement) {
    b(property: String),
    c(statement_index: usize),
    // r(matched_property: Option<&Match>), -- implement it so it returns MatchView
    v(matched_qualifiers: matched_qualifier_list_view::ListView),
});
pyview!(MatchedEntRelView(module = "gp_core.algorithms", name = "MatchedEntRelView", cls = MatchedEntRel) {
    b(source_entity_id: String),
    v(statements: matched_statement_list_view::ListView),
});
pyview!(PotentialRelationshipsView(module = "gp_core.algorithms", name = "PotentialRelationshipsView", cls = PotentialRelationships) {
    c(source_id: usize),
    c(target_id: usize),
    v(rels: matched_ent_rel_list_view::ListView)
});

pylist!(matched_qualifier_list_view(
    module = "gp_core.algorithms",
    item = super::MatchedQualifier as super::MatchedQualifierView
));
pylist!(matched_statement_list_view(
    module = "gp_core.algorithms",
    item = super::MatchedStatement as super::MatchedStatementView
));
pylist!(matched_ent_rel_list_view(
    module = "gp_core.algorithms",
    item = super::MatchedEntRel as super::MatchedEntRelView
));
pylist!(potential_relationships_list_view(
    module = "gp_core.algorithms",
    item = super::PotentialRelationships as super::PotentialRelationshipsView
));

#[pymethods]
impl MatchedEntRelView {
    pub fn get_matched_target_entities(&self, context: &PyAlgoContext) -> Vec<String> {
        self.0.get_matched_target_entities(&context.0)
    }
}

#[pymethods]
impl MatchedStatementView {
    #[getter]
    fn matched_property(&self) -> Option<MatchView> {
        if let Some(m) = &self.0.matched_property {
            Some(MatchView::new(m))
        } else {
            None
        }
    }
}
