use crate::{
    helper::OrdF64,
    models::{cangraph as cgraph, datagraph as dgraph},
};

use super::{
    super::feature_store::{FeatureStore, FeatureStoreCache},
    TargetRelationship,
};

pub fn get_rel_probs(
    store: &FeatureStore,
    storecache: &mut FeatureStoreCache,
    rels: &[TargetRelationship],
) -> Vec<f64> {
    rels.iter()
        .map(|rel| storecache.get_rel_prob(store, rel.source, rel.statement, rel.target))
        .collect::<Vec<_>>()
}

impl FeatureStoreCache {
    /// Get the probability of the relationship P(u, v, e) = P(u) * P(v) * P(e | u, v).
    pub fn get_rel_prob(
        &mut self,
        store: &FeatureStore,
        inedge: &cgraph::Edge,
        s: &cgraph::StatementNode,
        outedge: &cgraph::Edge,
    ) -> f64 {
        fn compute(
            this: &mut FeatureStoreCache,
            store: &FeatureStore,
            inedge: &cgraph::Edge,
            _s: &cgraph::StatementNode,
            outedge: &cgraph::Edge,
        ) -> f64 {
            let mut sum_prob = 0.0;

            for rowedges in outedge.dgprov.as_ref().unwrap().iter_all_edges() {
                sum_prob += rowedges
                    .iter()
                    .map(|&dg_sveid| {
                        let dg_svedge = store.dg.graph.get_edge(dg_sveid).unwrap();
                        let dg_usedge = match store.cg.graph.get_node(inedge.source).unwrap() {
                            cgraph::Node::Column(cgu) => {
                                store.dg.get_row_inedge_from_cell(dg_svedge, cgu.column)
                            }
                            cgraph::Node::Entity(cgu) => store
                                .dg
                                .get_row_inedge_from_entity(dg_svedge, &cgu.entity_id),
                            _ => unreachable!(),
                        };
                        let prob = dgraph::Edge::get_full_match_prob(
                            dg_usedge,
                            dg_svedge,
                            &store.dg,
                            &store.context,
                            &this.index2entscore,
                        );

                        OrdF64(prob.source * prob.target * prob.predicate)
                    })
                    .max()
                    .unwrap()
                    .0;
            }

            sum_prob
        }

        if !self.rel_prob.contains_key(&outedge.id) {
            let prob = compute(self, store, inedge, s, outedge);
            self.rel_prob.insert(outedge.id, prob);
        }
        self.rel_prob[&outedge.id]
    }
}
