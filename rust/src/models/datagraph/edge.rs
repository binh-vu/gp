use crate::models::{
    basegraph::{BaseEdge, EdgeId, NodeId},
    AlgoContext, CellCandidateEntities, Match, MatchMethod,
};
use kgdata_core::models::Value;
use serde::{Deserialize, Serialize};

use super::{DGraph, Node};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Edge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub predicate: String,
    // whether this is a qualifier, if so, which index in the list of qualifiers
    pub qualifier_index: Option<usize>,
    pub is_inferred: bool,
    pub prov: Option<EdgeProv>, // none for incoming edge to statement
}

/// Provenance implements Ord trait so that we can compare and
/// keep the best provenance. A provenance is better than other
/// provenance if its matching score is higher and entity match
/// is preferred over literal match.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EdgeProv(pub Match);

impl BaseEdge for Edge {
    #[inline]
    fn id(&self) -> EdgeId {
        self.id
    }

    #[inline]
    fn get_source(&self) -> NodeId {
        self.source
    }

    #[inline]
    fn get_target(&self) -> NodeId {
        self.target
    }

    #[inline]
    fn key(&self) -> &str {
        &self.predicate
    }

    fn set_id(&mut self, id: EdgeId) {
        self.id = id;
    }
}

impl Edge {
    /// get the row (subgraph) that this edge belongs to
    pub fn get_row(&self, dg: &DGraph) -> usize {
        match dg.graph.get_node(self.source).unwrap() {
            Node::Statement(s) => s.row,
            _ => match dg.graph.get_node(self.target).unwrap() {
                Node::Statement(s) => s.row,
                _ => unreachable!(),
            },
        }
    }

    /// Get a data matching trace that leads to the creation of this edge. Note that
    /// the edge must be an outgoing edge of a statement node.
    ///
    /// The trace is in a form of: (entity id, matched value, probability of the match)
    pub fn get_match_trace<'t>(
        &'t self,
        dg: &'t DGraph,
        context: &'t AlgoContext,
    ) -> (&'t str, &'t Value, &'t Match) {
        match dg.graph.get_node(self.source).unwrap() {
            Node::Statement(s) => {
                let ent = &context.entities[&s.entity_id];
                let stmt = &ent.props[&s.predicate][s.statement_index];

                // if this is an inferred property, the value is directly updated in context
                let val = if let Some(qual_index) = self.qualifier_index {
                    &stmt.qualifiers[&self.predicate][qual_index]
                } else {
                    &stmt.value
                };

                (&ent.id, val, &self.prov.as_ref().unwrap().0)
            },
            _ => panic!("Can't retrieve match trace on incoming edge of a statement. It must be an outgoing edge of a statement.")
        }
    }

    /// Get the probability of (source, target, and predicate) of a match that leads to the creation of
    /// this relationship (incoming edge and outgoing edge)
    pub fn get_full_match_prob(
        inedge: &Edge,
        outedge: &Edge,
        dg: &DGraph,
        context: &AlgoContext,
        cell2entscore: &CellCandidateEntities,
    ) -> FullMatchProb {
        let (entid, val, linkprob) = outedge.get_match_trace(dg, context);

        // retrieve score of the entity id
        let source_prob = match dg.graph.get_node(inedge.source).unwrap() {
            Node::Cell(u) => cell2entscore.get_prob(cell2entscore.get_index(dg, u), entid),
            Node::Entity(u) => u.entity_prob,
            _ => unreachable!(),
        };

        // retrieve the score of the value. if it's literal, the score is 1.0
        let target_prob = if outedge.prov.as_ref().unwrap().0.method == MatchMethod::LiteralMatching
        {
            1.0
        } else {
            match dg.graph.get_node(outedge.target).unwrap() {
                Node::Cell(v) => cell2entscore.get_prob(
                    cell2entscore.get_index(dg, v),
                    &val.as_entity_id().unwrap().id,
                ),
                Node::Entity(v) => v.entity_prob,
                // it is 1.0, but why we have value_prob when its value is always 1.0?
                // we have literal node, when it's statement value.
                Node::Literal(v) => v.value_prob,
                x => unreachable!("{:?}", x),
            }
        };

        FullMatchProb {
            source: source_prob,
            target: target_prob,
            predicate: linkprob.prob,
        }
    }
}

pub struct FullMatchProb {
    pub source: f64,
    pub target: f64,
    pub predicate: f64,
}
