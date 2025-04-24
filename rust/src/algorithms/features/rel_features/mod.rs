pub mod detect_contradicted_info;
pub mod freq;
pub mod funcdep;
pub mod prob;

use hashbrown::HashMap;
use polars::prelude::*;

use self::detect_contradicted_info::MissingInfoDetector;
use self::funcdep::FunctionalDependencyDetector;

use super::feature_store::{FeatureStore, FeatureStoreCache};
use crate::error::GramsError;
use crate::models::cangraph as cgraph;

// Relationship that we want to compute the features
pub struct TargetRelationship<'t> {
    pub statement: &'t cgraph::StatementNode,
    pub source: &'t cgraph::Edge,
    pub target: &'t cgraph::Edge,
}

pub fn extract_target_relationships(cg: &cgraph::CGraph) -> Vec<TargetRelationship<'_>> {
    let mut rels = Vec::new();

    for node in cg.graph.iter_nodes() {
        match node {
            cgraph::Node::Statement(u) => {
                let source = cg.graph.iter_in_edges(u.id).next().unwrap();
                for target in cg.graph.iter_out_edges(u.id) {
                    rels.push(TargetRelationship {
                        statement: u,
                        source,
                        target,
                    });
                }
            }
            _ => {}
        }
    }

    rels
}

pub fn extract_rel_features<MID: MissingInfoDetector, FDD: FunctionalDependencyDetector>(
    store: &mut FeatureStore,
    storecache: &mut FeatureStoreCache,
    missing_info_detector: &mut MID,
    func_dep_detector: &mut FDD,
    correct_entity_threshold: f64,
) -> Result<(HashMap<String, u32>, DataFrame), GramsError> {
    let rels = extract_target_relationships(store.cg);

    // turn rels into vec arrays
    let mut props = HashMap::new();
    for rel in &rels {
        if !props.contains_key(&rel.source.predicate) {
            props.insert(rel.source.predicate.clone(), props.len() as u32);
        }
        if !props.contains_key(&rel.target.predicate) {
            props.insert(rel.target.predicate.clone(), props.len() as u32);
        }
    }

    let mut source = Vec::with_capacity(rels.len());
    let mut target = Vec::with_capacity(rels.len());
    let mut statement = Vec::with_capacity(rels.len());
    let mut inedge = Vec::with_capacity(rels.len());
    let mut outedge = Vec::with_capacity(rels.len());

    let freqs = self::freq::get_rel_freqs(store, storecache, &rels);
    let probs = self::prob::get_rel_probs(store, storecache, &rels);
    let max_num_pos_ent_links = self::freq::get_max_num_pos_ent_links(store, storecache, &rels)?;
    let max_num_pos_rels = self::freq::get_max_num_pos_rels(store, storecache, &rels)?;
    let num_unmatch_links = self::freq::get_rel_num_unmatch_links(
        store,
        storecache,
        &rels,
        missing_info_detector,
        correct_entity_threshold,
    )?;
    let not_func_dep =
        self::funcdep::not_func_dependency(store, storecache, func_dep_detector, &rels);

    let mut freq_topk_series = Vec::with_capacity(4);
    for k in [1, 2, 3, 5].into_iter() {
        freq_topk_series.extend(
            vec![Series::new(
                &format!("freq_top{}", k),
                self::freq::get_rel_topk_freqs(store, storecache, &rels, k),
            )]
            .into_iter(),
        );
    }

    for rel in rels {
        source.push(rel.source.source.0 as u32);
        target.push(rel.target.target.0 as u32);
        statement.push(rel.statement.id.0 as u32);
        inedge.push(props[&rel.source.predicate]);
        outedge.push(props[&rel.target.predicate]);
    }

    let df = DataFrame::new(
        [
            Series::new("source", source),
            Series::new("target", target),
            Series::new("statement", statement),
            Series::new("inedge", inedge),
            Series::new("outedge", outedge),
            Series::new("freq", freqs),
            Series::new("max_num_pos_ent_links", max_num_pos_ent_links),
            Series::new("max_num_pos_rels", max_num_pos_rels),
            Series::new("num_unmatch_links", num_unmatch_links),
        ]
        .into_iter()
        .chain(freq_topk_series)
        .chain([
            Series::new("prob", probs),
            Series::new("not_func_dep", not_func_dep),
        ])
        .collect::<Vec<_>>(),
    )?;

    Ok((props, df))
}
