use hashbrown::HashMap;
use kgdata_core::models::Class;
use kgdata_core::models::Entity;
use kgdata_core::models::Statement;

use crate::error::GramsError;
use crate::models::cangraph as cgraph;
use crate::models::db::CacheRocksDBDict;
use crate::models::Table;

use super::feature_store::FeatureStore;
use super::feature_store::FeatureStoreCache;

pub struct TargetType {
    pub column: usize,
    pub column_type: String,
}

pub struct TargetTypes {
    pub column: usize,
    pub column_types: Vec<String>,
}

/// Extract columns that should be tagged with types
pub fn get_tagged_columns(cg: &cgraph::CGraph, table: &Table) -> Vec<usize> {
    let mut columns = vec![];

    for node in cg.graph.iter_nodes() {
        match node {
            cgraph::Node::Column(u) => {
                // using heuristic to determine if we should tag this column
                let covered_fractions = table
                    .links
                    .iter()
                    .enumerate()
                    .filter_map(|(ri, rowlinks)| {
                        let n_cans = rowlinks[u.column]
                            .iter()
                            .map(|link| link.candidates.len())
                            .sum::<usize>();

                        if n_cans == 0 {
                            None
                        } else {
                            Some(
                                rowlinks[u.column]
                                    .iter()
                                    .map(|link| (link.end - link.start))
                                    .sum::<usize>() as f32
                                    / table.columns[u.column].values[ri].len().max(1) as f32,
                            )
                        }
                    })
                    .collect::<Vec<_>>();

                if covered_fractions.len() == 0 {
                    continue;
                }

                let avg_covered_fractions =
                    covered_fractions.iter().sum::<f32>() / covered_fractions.len() as f32;

                if avg_covered_fractions < 0.8 {
                    if avg_covered_fractions < 1e-6 {
                        let avg_cell_len = table.columns[u.column]
                            .values
                            .iter()
                            .enumerate()
                            .filter_map(|(ri, cell)| {
                                let n_cans = table.links[ri][u.column]
                                    .iter()
                                    .map(|link| link.candidates.len())
                                    .sum::<usize>();

                                if n_cans == 0 {
                                    None
                                } else {
                                    Some(cell.len())
                                }
                            })
                            .sum::<usize>();

                        if avg_cell_len < 1 {
                            // links are likely to be image such as national flag, so we still model them
                            columns.push(u.column);
                        }
                    }
                    continue;
                }

                columns.push(u.column);
            }
            _ => {}
        }
    }

    columns
}

pub fn get_type_freq(
    store: &FeatureStore,
    storecache: &mut FeatureStoreCache,
    target_columns: Vec<usize>,
    target_types: Vec<TargetType>,
) -> Result<Vec<f64>, GramsError> {
    let col2typefreqs = target_columns
        .iter()
        .map(|&ci| Ok((ci, get_column_type_freq(store, storecache, ci)?)))
        .collect::<Result<HashMap<usize, _>, GramsError>>()?;

    Ok(target_types
        .iter()
        .map(|ttype| {
            // what happen when the type does not exist?
            *col2typefreqs[&ttype.column]
                .get(&ttype.column_type)
                .unwrap()
        })
        .collect::<Vec<_>>())
}

pub fn get_num_ent_rows(
    store: &FeatureStore,
    storecache: &mut FeatureStoreCache,
    target_columns: Vec<usize>,
    target_types: Vec<TargetType>,
) -> Vec<u32> {
    let (nrows, ncols) = store.table.shape();

    fn get_column_num_ent_rows(
        storecache: &FeatureStoreCache,
        nrows: usize,
        ncols: usize,
        ci: usize,
    ) -> u32 {
        (0..nrows)
            .into_iter()
            .map(|ri| {
                (storecache
                    .index2entscore
                    .get_num_candidates(ri * ncols + ci)
                    > 0) as u32
            })
            .sum::<u32>()
    }

    let col2value = target_columns
        .iter()
        .map(|&ci| (ci, get_column_num_ent_rows(storecache, nrows, ncols, ci)))
        .collect::<HashMap<usize, u32>>();

    target_types
        .iter()
        .map(|ttype| *col2value.get(&ttype.column).unwrap())
        .collect::<Vec<_>>()
}

pub fn get_type_extended_freq(
    store: &FeatureStore,
    storecache: &mut FeatureStoreCache,
    target_columns: Vec<usize>,
    target_types: Vec<TargetType>,
    extended_distance: usize,
) -> Result<Vec<f64>, GramsError> {
    let col2exttypefreqs = target_columns
        .iter()
        .map(|&ci| {
            Ok((
                ci,
                get_column_type_extended_freq(store, storecache, ci, extended_distance)?,
            ))
        })
        .collect::<Result<HashMap<usize, _>, GramsError>>()?;

    Ok(target_types
        .iter()
        .map(|ttype| {
            // what happen when the type does not exist?
            *col2exttypefreqs[&ttype.column]
                .get(&ttype.column_type)
                .unwrap()
        })
        .collect::<Vec<_>>())
}

/// The frequency of whether an entity of a type in a row has been used to construct an edge
/// in data graph (connecting to other nodes)
#[allow(unused_variables)]
pub fn get_freq_discovered_prop(
    store: &FeatureStore,
    _storecache: &mut FeatureStoreCache,
    target_types: Vec<TargetTypes>,
) -> Result<Vec<f64>, GramsError> {
    let (nrows, ncols) = store.table.shape();
    let instanceof = &store.kgns.instanceof;
    let entities = &store.context.entities;
    let dg = &store.dg;
    let cg = &store.cg;

    for target_type in target_types {
        let ci = target_type.column;
        let uid = cg.get_column_node_id(ci);

        // type2row[type][row * ncols] = score of a match
        let type2row: HashMap<&String, Vec<f64>> = target_type
            .column_types
            .iter()
            .map(|ctype| (ctype, vec![0.0; nrows]))
            .collect();

        // for outgoing edges, gather the entities from the column that we discovered the relationship.
        for outedge in cg.graph.iter_out_edges(uid) {
            // however, we do not want to take into account edges that connect into intermediate nodes that
            // ain't columns or in the context
            if !cg.graph.iter_out_edges(outedge.target).any(|stmt_outedge| {
                let cgv = cg.graph.get_node(stmt_outedge.target).unwrap();
                match cgv {
                    cgraph::Node::Column(_) => true,
                    cgraph::Node::Entity(x) => x.is_in_context,
                    cgraph::Node::Literal(x) => x.is_in_context,
                    _ => unreachable!(),
                }
            }) {
                continue;
            }

            // get dg statements that construct this outedge from the current column
            // then, for each statement, we retrieve the entity id, then we find the type of the entity
            // and record the type with probability.
            let cgstmt = cg
                .graph
                .get_node(outedge.target)
                .unwrap()
                .try_as_statement()
                .unwrap();

            unimplemented!("Hasn't migrated from dg_stmts yet.")
            // for &dgsid in &cgstmt.dg_stmts {
            //     let dgstmt = dg.get_statement_node(dgsid).unwrap();
            //     let dgu = dg
            //         .graph
            //         .get_node(storecache.get_dg_statement_source(dg, cg, cgstmt.id, dgsid))
            //         .unwrap()
            //         .as_cell();
            //     let cellindex = dgu.row * ncols + dgu.column;
            //     let ent = &entities[&dgstmt.entity_id];
            //     for enttype in iter_entity_types(ent, instanceof) {
            //         let score =
            //             &mut type2row.entry(enttype?).or_insert_with(|| vec![0.0; nrows])[dgu.row];
            //         *score = score.max(storecache.index2entscore.get_prob(cellindex, &ent.id));
            //     }
            // }
        }

        for inedge in cg.graph.iter_in_edges(uid) {
            // however, we do not want to take into account edges that connect into intermediate nodes that
            // ain't columns or in the context
            if !cg.graph.iter_in_edges(inedge.source).any(|stmt_inedge| {
                let cgu = cg.graph.get_node(stmt_inedge.source).unwrap();
                match cgu {
                    cgraph::Node::Column(_) => true,
                    cgraph::Node::Entity(x) => x.is_in_context,
                    cgraph::Node::Literal(x) => x.is_in_context,
                    _ => unreachable!(),
                }
            }) {
                continue;
            }

            let _cgstmt = cg
                .graph
                .get_node(inedge.source)
                .unwrap()
                .try_as_statement()
                .unwrap();

            // todo: get the statement target probabilities
            todo!()
        }
    }

    todo!()
}

pub fn type_distance(
    _store: &FeatureStore,
    _storecache: &mut FeatureStoreCache,
    _target_columns: Vec<usize>,
    _target_types: Vec<TargetType>,
) -> Result<Vec<f64>, GramsError> {
    todo!()
}

/// Calculating frequency of types in a column.
/// Each time a type appears in a cell, instead of counting 1, we count its probability
pub fn get_column_type_freq(
    store: &FeatureStore,
    storecache: &mut FeatureStoreCache,
    column: usize,
) -> Result<HashMap<String, f64>, GramsError> {
    let instanceof = &store.kgns.instanceof;

    let (nrows, ncols) = store.table.shape();
    let mut type2freq: HashMap<String, f64> = HashMap::new();

    for ri in 0..nrows {
        let mut classes: HashMap<&String, f64> = HashMap::new();
        for (entid, prob) in storecache
            .index2entscore
            .iter_candidate_entities(ri * ncols + column)
        {
            let ent = &store.context.entities[entid];
            for clsid in iter_entity_types(ent, instanceof) {
                let clsid = clsid?;
                if classes.get(clsid).unwrap_or(&0.0) < prob {
                    classes.insert(clsid, *prob);
                }
            }
        }
        for (&clsid, &prob) in classes.iter() {
            if !type2freq.contains_key(clsid) {
                type2freq.insert(clsid.clone(), prob);
            } else {
                *type2freq.get_mut(clsid).unwrap() += prob;
            }
        }
    }

    Ok(type2freq)
}

pub fn get_column_type_extended_freq(
    store: &FeatureStore,
    storecache: &mut FeatureStoreCache,
    column: usize,
    extended_distance: usize,
) -> Result<HashMap<String, f64>, GramsError> {
    let instanceof = &store.kgns.instanceof;

    let (nrows, ncols) = store.table.shape();
    let mut type2extfreq: HashMap<String, f64> = HashMap::new();

    for ri in 0..nrows {
        let mut classes: HashMap<&String, f64> = HashMap::new();
        for (entid, prob) in storecache
            .index2entscore
            .iter_candidate_entities(ri * ncols + column)
        {
            let ent = &store.context.entities[entid];
            for clsid in iter_entity_types(ent, instanceof) {
                let clsid = clsid?;
                if classes.get(clsid).unwrap_or(&0.0) < prob {
                    classes.insert(clsid, *prob);
                }
            }
        }
        let extclasses = extend_types(&mut storecache.classes, &classes, extended_distance)?;
        for (clsid, prob) in extclasses.into_iter() {
            if !type2extfreq.contains_key(&clsid) {
                type2extfreq.insert(clsid, prob);
            } else {
                *type2extfreq.get_mut(&clsid).unwrap() += prob;
            }
        }
    }

    Ok(type2extfreq)
}

/// Get ancestors of classes within the given distance. Distance 1 is the parent, distance 2 is the grand parent
pub fn extend_types<'t>(
    classes: &mut CacheRocksDBDict<Class>,
    types: &HashMap<&'t String, f64>,
    distance: usize,
) -> Result<HashMap<String, f64>, GramsError> {
    let mut exttypes = types
        .iter()
        .map(|(&cid, &prob)| (cid.clone(), prob))
        .collect::<HashMap<String, f64>>();

    for (&clsid, &prob) in types.iter() {
        classes
            .get(clsid)?
            .ok_or_else(|| GramsError::DBKeyError(clsid.to_owned()))?
            .ancestors
            .iter()
            .filter_map(
                |(ancid, &dis)| {
                    if dis <= distance {
                        Some(ancid)
                    } else {
                        None
                    }
                },
            )
            .for_each(|ancid| {
                if *exttypes.get(ancid).unwrap_or(&0.0) < prob {
                    exttypes.insert(ancid.clone(), prob);
                }
            });
    }

    Ok(exttypes)
}

const DEFAULT_VEC: Vec<Statement> = Vec::new();
const REF_DEFAULT_VEC: &Vec<Statement> = &DEFAULT_VEC;

/// Get types of an entity
#[inline]
pub fn iter_entity_types<'t>(
    ent: &'t Entity,
    instanceof: &'t String,
) -> impl Iterator<Item = Result<&'t String, GramsError>> {
    ent.props
        .get(instanceof)
        .unwrap_or(REF_DEFAULT_VEC)
        .iter()
        .map(|stmt| {
            Ok(&stmt
                .value
                .as_entity_id()
                .ok_or_else(|| {
                    GramsError::DBTypeError(format!("instanceof {} is not an entity", ent.id))
                })?
                .id)
        })
}
