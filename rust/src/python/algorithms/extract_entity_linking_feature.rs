use std::ops::{Deref, DerefMut};

use crate::algorithms::data_matching::{DataMatching, Node, PotentialRelationships};
use crate::error::into_pyerr;
use crate::helper::get_progress_bar_style;
use crate::libs::index::IndexTraversal;
use crate::libs::literal_matchers::PyLiteralMatcher;
use crate::models::{AlgoContext, Table, TableCells};
use crate::python::models::{db::PyGramsDB, PyAlgoContext};
use hashbrown::{HashMap, HashSet};
use indicatif::{ParallelProgressIterator, ProgressFinish};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
#[pyo3(signature = (
    db,
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
pub fn extract_candidate_entity_link_freqs<'t>(
    db: PyGramsDB<'t>,
    table: &Table,
    table_cells: &TableCells,
    context: Option<PyRefMut<'t, PyAlgoContext>>,
    literal_matcher: &PyLiteralMatcher,
    ignored_columns: Vec<usize>,
    ignored_props: Vec<String>,
    allow_same_ent_search: bool,
    allow_ent_matching: bool,
    use_context: bool,
    deterministic_order: bool,
    parallel: bool,
) -> PyResult<Vec<HashMap<String, usize>>> {
    let ignored_props_set = HashSet::from_iter(ignored_props);

    match context {
        None => {
            let mut context = db
                .deref()
                .get_algo_context(table, 1, true)
                .map_err(into_pyerr)?;

            wrap_matching(
                table,
                table_cells,
                &mut context,
                literal_matcher,
                &ignored_columns,
                &ignored_props_set,
                allow_same_ent_search,
                allow_ent_matching,
                use_context,
                deterministic_order,
                parallel,
            )
        }
        Some(mut refcontext) => wrap_matching(
            table,
            table_cells,
            &mut refcontext.deref_mut().0,
            literal_matcher,
            &ignored_columns,
            &ignored_props_set,
            allow_same_ent_search,
            allow_ent_matching,
            use_context,
            deterministic_order,
            parallel,
        ),
    }
}

/// Calculate how many incoming and outgoing relationships to other cells
/// in a row of a candidate entity
#[pyfunction]
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
pub fn par_extract_candidate_entity_link_freqs<'t>(
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
) -> PyResult<Vec<Vec<HashMap<String, usize>>>> {
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
                it.progress_with_style(get_progress_bar_style(
                    "extract candidate entity link freqs",
                ))
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
                it.progress_with_style(get_progress_bar_style(
                    "extract candidate entity link freqs",
                ))
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
) -> PyResult<Vec<HashMap<String, usize>>> {
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
    Ok(prep_result(table, context, nodes, edges))
}

pub fn prep_result(
    table: &Table,
    context: &AlgoContext,
    nodes: Vec<Node>,
    edges: Vec<PotentialRelationships>,
) -> Vec<HashMap<String, usize>> {
    let (nrows, ncols) = table.shape();
    let mut out = vec![HashMap::<String, HashSet<usize>>::new(); nrows * ncols];

    for rels in &edges {
        let source = &nodes[rels.source_id];
        let target = &nodes[rels.target_id];

        match source {
            Node::Cell(source) => {
                let source_indx = source.row * ncols + source.col;
                match target {
                    Node::Cell(target) => {
                        let target_indx = target.row * ncols + target.col;
                        for rel in &rels.rels {
                            if !out[source_indx].contains_key(&rel.source_entity_id) {
                                out[source_indx]
                                    .insert(rel.source_entity_id.clone(), HashSet::new());
                            }
                            out[source_indx]
                                .get_mut(&rel.source_entity_id)
                                .unwrap()
                                .insert(target_indx);

                            for target_entity_id in rel.get_matched_target_entities(context) {
                                if !out[target_indx].contains_key(&target_entity_id) {
                                    let mut tmp = HashSet::new();
                                    tmp.insert(source_indx);
                                    out[target_indx].insert(target_entity_id, tmp);
                                } else {
                                    out[target_indx]
                                        .get_mut(&target_entity_id)
                                        .unwrap()
                                        .insert(source_indx);
                                }
                            }
                        }
                    }
                    _ => continue,
                }
            }
            _ => continue,
        }
    }

    out.into_iter()
        .map(|x| {
            x.into_iter()
                .map(|(k, v)| (k, v.len()))
                .collect::<HashMap<_, _>>()
        })
        .collect::<Vec<_>>()
}
