use hashbrown::HashMap;

use crate::{
    error::GramsError,
    helper::OrdF64,
    models::{basegraph::NodeId, cangraph as cgraph, MatchMethod},
};

use super::{
    super::feature_store::{FeatureStore, FeatureStoreCache},
    detect_contradicted_info::{get_contradicted_information, MissingInfoDetector},
    TargetRelationship,
};

/// for each rel, compute the maximum number of possible entity links
/// between two nodes forming the relationship. See `get_maximum_possible_ent_links_between_two_nodes`
/// for more details.
pub fn get_max_num_pos_ent_links(
    store: &mut FeatureStore,
    storecache: &mut FeatureStoreCache,
    rels: &[TargetRelationship],
) -> Result<Vec<u32>, GramsError> {
    rels.iter()
        .map(|rel| {
            let max_pos_ent_row = storecache.get_maximum_possible_ent_links_between_two_nodes(
                store,
                rel.statement,
                rel.source,
                rel.target,
            )?;
            Ok(max_pos_ent_row as u32)
        })
        .collect::<Result<Vec<_>, GramsError>>()
}
/// for each rel, compute the maxmimum number of possible relations that the two nodes can have
pub fn get_max_num_pos_rels(
    store: &mut FeatureStore,
    storecache: &mut FeatureStoreCache,
    rels: &[TargetRelationship],
) -> Result<Vec<u32>, GramsError> {
    let mut max_pos_rels: HashMap<(NodeId, NodeId), usize> = HashMap::with_capacity(rels.len());
    let mut output = Vec::with_capacity(rels.len());

    for rel in rels {
        let n_possible_links = storecache.get_unmatch_discovered_links(
            store,
            rel.statement,
            rel.source,
            rel.target,
        )? + rel.target.dgprov.as_ref().unwrap().get_num_matched_rows();

        let uv = (rel.source.source, rel.target.target);
        if *max_pos_rels.get(&uv).unwrap_or(&0) < n_possible_links {
            max_pos_rels.insert(uv, n_possible_links);
        }
    }

    for rel in rels {
        let uv = (rel.source.source, rel.target.target);
        output.push(max_pos_rels[&uv] as u32);
    }
    Ok(output)
}

/// for each rel, compute rel freq
pub fn get_rel_freqs(
    store: &mut FeatureStore,
    storecache: &mut FeatureStoreCache,
    rels: &[TargetRelationship],
) -> Vec<f64> {
    rels.iter()
        .map(|rel| storecache.get_or_compute__rel_freq(store, rel.target))
        .collect::<Vec<_>>()
}

/// for each rel, compute rel freq of topk candidate entities
pub fn get_rel_topk_freqs(
    store: &mut FeatureStore,
    storecache: &mut FeatureStoreCache,
    rels: &[TargetRelationship],
    topk: usize,
) -> Vec<u32> {
    rels.iter()
        .map(|rel| {
            storecache.get_or_compute__rel_freq_topk(
                store,
                rel.source,
                rel.statement,
                rel.target,
                topk,
            ) as u32
        })
        .collect::<Vec<_>>()
}

/// for each rel, calculate the number of unmatched links between two nodes
pub fn get_rel_num_unmatch_links<D: MissingInfoDetector>(
    store: &mut FeatureStore,
    storecache: &mut FeatureStoreCache,
    rels: &[TargetRelationship],
    missing_info_detector: &mut D,
    correct_entity_threshold: f64,
) -> Result<Vec<u32>, GramsError> {
    rels.iter()
        .map(|rel| {
            Ok(get_contradicted_information(
                store,
                storecache,
                missing_info_detector,
                rel.source,
                rel.target,
                correct_entity_threshold,
            )?
            .len() as u32)
        })
        .collect::<Result<Vec<u32>, GramsError>>()
}

impl FeatureStoreCache {
    /// The relative frequency of the outedge of a statement node.
    #[allow(non_snake_case)]
    pub fn get_or_compute__rel_freq(
        &mut self,
        store: &FeatureStore,
        outedge: &cgraph::Edge,
    ) -> f64 {
        fn compute(store: &FeatureStore, outedge: &cgraph::Edge) -> f64 {
            let mut sum_prob = 0.0;

            for rowedges in outedge.dgprov.as_ref().unwrap().iter_all_edges() {
                let prob_edge_row = rowedges
                    .iter()
                    .map(|&dg_sveid| {
                        OrdF64(
                            store
                                .dg
                                .graph
                                .get_edge(dg_sveid)
                                .unwrap()
                                .prov
                                .as_ref()
                                .unwrap()
                                .0
                                .prob,
                        )
                    })
                    .max()
                    .unwrap()
                    .0;
                sum_prob += prob_edge_row;
            }

            sum_prob
        }

        if !self.rel_freq.contains_key(&outedge.id) {
            self.rel_freq.insert(outedge.id, compute(store, outedge));
        }
        self.rel_freq[&outedge.id]
    }

    /// Get the relative frequency of the outedge of statement node, but for each row, we only count links
    /// that are discovered from the top K candidate entities of a cell. If the source node is a column node,
    /// then the top K candidate entities are of the source cell (target node as no effect). If the source node
    /// is an entity node, then the top K candidate entities are of the target cell if the target cell is linked
    /// by entity, otherwise, it is the same as K is infinite. If both source and target node are entity nodes,
    /// then top K has no effect (infinite) and the function is the same as `get_rel_freq`.
    #[allow(non_snake_case)]
    pub fn get_or_compute__rel_freq_topk(
        &mut self,
        store: &FeatureStore,
        inedge: &cgraph::Edge,
        s: &cgraph::StatementNode,
        outedge: &cgraph::Edge,
        topk: usize,
    ) -> f64 {
        fn compute(
            this: &mut FeatureStoreCache,
            store: &FeatureStore,
            inedge: &cgraph::Edge,
            _s: &cgraph::StatementNode,
            outedge: &cgraph::Edge,
            topk: usize,
        ) -> f64 {
            let mut sum_prob = 0.0;
            let ncols = store.table.n_cols();

            match store.cg.graph.get_node(inedge.source).unwrap() {
                cgraph::Node::Column(cgu) => {
                    for (ri, rowedges) in outedge.dgprov.as_ref().unwrap().enumerate_all_edges() {
                        let prob_edge_row = rowedges
                            .iter()
                            .filter_map(|&dg_sveid| {
                                let dg_svedge = store.dg.graph.get_edge(dg_sveid).unwrap();

                                // retrieve trace of this edge
                                let trace = dg_svedge.get_match_trace(&store.dg, &store.context);

                                let cellindex = ri * ncols + cgu.column;
                                if this.index2entscore.get_rank(cellindex, trace.0) <= topk {
                                    Some(OrdF64(trace.2.prob))
                                } else {
                                    None
                                }
                            })
                            .max()
                            .unwrap_or(OrdF64(0.0))
                            .0;
                        sum_prob += prob_edge_row;
                    }
                }
                cgraph::Node::Entity(_) => {
                    match store.cg.graph.get_node(outedge.target).unwrap() {
                        cgraph::Node::Entity(_) => {
                            return this.get_or_compute__rel_freq(store, outedge);
                        }
                        cgraph::Node::Column(cgv) => {
                            for (ri, rowedges) in
                                outedge.dgprov.as_ref().unwrap().enumerate_all_edges()
                            {
                                let prob_edge_row = rowedges
                                    .iter()
                                    .filter_map(|&dg_sveid| {
                                        let dg_svedge = store.dg.graph.get_edge(dg_sveid).unwrap();

                                        // retrieve trace of this edge
                                        let trace =
                                            dg_svedge.get_match_trace(&store.dg, &store.context);

                                        if dg_svedge.prov.as_ref().unwrap().0.method
                                            == MatchMethod::LinkMatching
                                        {
                                            let cellindex = ri * ncols + cgv.column;
                                            if this.index2entscore.get_rank(cellindex, trace.0)
                                                <= topk
                                            {
                                                Some(OrdF64(trace.2.prob))
                                            } else {
                                                None
                                            }
                                        } else {
                                            // K is infinite if the target is not linked by entity
                                            Some(OrdF64(trace.2.prob))
                                        }
                                    })
                                    .max()
                                    .unwrap_or(OrdF64(0.0))
                                    .0;
                                sum_prob += prob_edge_row;
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }

            sum_prob
        }

        if !self.rel_freq_topk.contains_key(&outedge.id) {
            let val = compute(self, store, inedge, s, outedge, topk);
            self.rel_freq_topk.insert(outedge.id, val);
        }
        self.rel_freq_topk[&outedge.id]
    }
}
