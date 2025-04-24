use std::ops::Index;

use derive_more::Display;
use hashbrown::HashMap;
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::{
    EdgeIndex, EdgeIndices, NodeIndex, NodeIndices, StableGraph,
};
use rustworkx_core::petgraph::{visit::EdgeRef, Directed, Direction};
use serde::{Deserialize, Serialize};

#[derive(
    FromPyObject,
    Hash,
    Serialize,
    Deserialize,
    Display,
    Default,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
)]
pub struct NodeId(pub usize);
#[derive(
    FromPyObject,
    Hash,
    Serialize,
    Deserialize,
    Display,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
)]
pub struct EdgeId(pub usize);

pub trait BaseEdge {
    fn id(&self) -> EdgeId;
    fn get_source(&self) -> NodeId;
    fn get_target(&self) -> NodeId;
    fn key(&self) -> &str;
    fn set_id(&mut self, id: EdgeId);
}

pub trait BaseNode {
    fn id(&self) -> NodeId;
    fn set_id(&mut self, id: NodeId);
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BaseDirectedGraph<N, E>(pub StableGraph<N, E, Directed>);

impl<N, E> Default for BaseDirectedGraph<N, E>
where
    N: BaseNode,
    E: BaseEdge,
{
    fn default() -> Self {
        BaseDirectedGraph(StableGraph::default())
    }
}

impl<N, E> BaseDirectedGraph<N, E>
where
    N: BaseNode,
    E: BaseEdge,
{
    pub fn num_nodes(&self) -> usize {
        self.0.node_count()
    }

    pub fn nodes(&self) -> Vec<&N> {
        self.0
            .node_indices()
            .map(|i| self.0.node_weight(i).unwrap())
            .collect()
    }

    pub fn iter_nodes(&self) -> DirectedGraphNodeIterator<N, E> {
        DirectedGraphNodeIterator {
            graph: &self.0,
            index: self.0.node_indices(),
        }
    }

    #[inline]
    pub fn has_node(&self, nodeid: NodeId) -> bool {
        self.0.contains_node(NodeIndex::new(nodeid.0))
    }

    #[inline]
    pub fn get_node(&self, nodeid: NodeId) -> Option<&N> {
        self.0.node_weight(NodeIndex::new(nodeid.0))
    }

    #[inline]
    pub fn get_mut_node(&mut self, nodeid: NodeId) -> Option<&mut N> {
        self.0.node_weight_mut(NodeIndex::new(nodeid.0))
    }

    /// Different from our graph APIs in Python that this returns a reference to the node.
    pub fn add_node(&mut self, node: N) -> &N {
        let index = self.0.add_node(node);
        let u = self.0.node_weight_mut(index).unwrap();
        u.set_id(NodeId(index.index()));
        u
    }

    #[inline]
    pub fn remove_node(&mut self, nid: NodeId) -> Option<N> {
        self.0.remove_node(NodeIndex::new(nid.0))
    }

    #[inline]
    pub fn num_edges(&self) -> usize {
        self.0.edge_count()
    }

    #[inline]
    pub fn edges(&self) -> Vec<&E> {
        self.0
            .edge_indices()
            .map(|i| self.0.edge_weight(i).unwrap())
            .collect()
    }

    #[inline]
    pub fn iter_edges(&self) -> DirectedGraphEdgeIterator<N, E> {
        DirectedGraphEdgeIterator {
            graph: &self.0,
            index: self.0.edge_indices(),
        }
    }

    #[inline]
    pub fn in_edges(&self, uid: NodeId) -> Vec<&E> {
        self.0
            .edges_directed(NodeIndex::new(uid.0), Direction::Incoming)
            .map(|e| e.weight())
            .collect()
    }

    #[inline]
    pub fn iter_in_edges(&self, uid: NodeId) -> impl Iterator<Item = &E> + '_ {
        self.0
            .edges_directed(NodeIndex::new(uid.0), Direction::Incoming)
            .map(|e| e.weight())
    }

    #[inline]
    pub fn iter_out_edges(&self, uid: NodeId) -> impl Iterator<Item = &E> + '_ {
        self.0
            .edges_directed(NodeIndex::new(uid.0), Direction::Outgoing)
            .map(|e| e.weight())
    }

    #[inline]
    pub fn out_edges(&self, uid: NodeId) -> Vec<&E> {
        self.0
            .edges_directed(NodeIndex::new(uid.0), Direction::Outgoing)
            .map(|e| e.weight())
            .collect()
    }

    #[inline]
    pub fn has_edge(&self, eid: EdgeId) -> bool {
        self.0.edge_weight(EdgeIndex::new(eid.0)).is_some()
    }

    #[inline]
    pub fn get_edge(&self, eid: EdgeId) -> Option<&E> {
        self.0.edge_weight(EdgeIndex::new(eid.0))
    }

    /// Different from our graph APIs in Python that this returns a reference to the node.
    pub fn add_edge(&mut self, edge: E) -> &E {
        let edgeid = self.0.add_edge(
            NodeIndex::new(edge.get_source().0),
            NodeIndex::new(edge.get_target().0),
            edge,
        );

        let edge = self.0.edge_weight_mut(edgeid).unwrap();
        edge.set_id(EdgeId(edgeid.index()));
        return edge;
    }

    pub fn remove_edge(&mut self, eid: usize) -> Option<E> {
        self.0.remove_edge(EdgeIndex::new(eid))
    }

    pub fn remove_edge_between_nodes(
        &mut self,
        source: NodeId,
        target: NodeId,
        predicate: &str,
    ) -> Option<E> {
        if let Some(edge) = self.get_edge_between_nodes(source, target, predicate) {
            self.0.remove_edge(EdgeIndex::new(edge.id().0))
        } else {
            None
        }
    }

    pub fn remove_edges_between_nodes(&mut self, source: NodeId, target: NodeId) {
        while let Some(edgeid) = self
            .0
            .find_edge(NodeIndex::new(source.0), NodeIndex::new(target.0))
        {
            self.0.remove_edge(edgeid);
        }
    }

    pub fn get_edge_between_nodes(&self, source: NodeId, target: NodeId, key: &str) -> Option<&E> {
        let raw_edges = self
            .0
            .edges_directed(NodeIndex::new(source.0), Direction::Outgoing);

        for edgeref in raw_edges {
            let edge = edgeref.weight();
            if edgeref.target().index() == target.0 && edge.key() == key {
                return Some(edge);
            }
        }
        return None;
    }

    pub fn get_mut_edge_between_nodes(
        &mut self,
        source: NodeId,
        target: NodeId,
        key: &str,
    ) -> Option<&mut E> {
        let eid = {
            let raw_edges = self
                .0
                .edges_connecting(NodeIndex::new(source.0), NodeIndex::new(target.0));

            let mut eid = None;
            for edgeref in raw_edges {
                let edge = edgeref.weight();
                if edge.key() == key {
                    eid = Some(edgeref.id());
                    break;
                }
            }
            eid
        };

        self.0.edge_weight_mut(eid?)
    }

    #[inline]
    pub fn has_edge_between_nodes(&self, source: NodeId, target: NodeId, key: &str) -> bool {
        self.get_edge_between_nodes(source, target, key).is_some()
    }

    /// Check if the weights of the graph contains ids that are consistent with the id in the graph.
    pub fn is_valid(&self) -> bool {
        let mut it = self.0.node_indices();
        while let Some(index) = it.next() {
            let node = self.0.node_weight(index).unwrap();
            if node.id() != NodeId(index.index()) {
                log::warn!("Node {} stores different id {}", index.index(), node.id().0);
                return false;
            }
        }
        let mut it = self.0.edge_indices();
        while let Some(index) = it.next() {
            let edge = self.0.edge_weight(index).unwrap();
            if edge.id() != EdgeId(index.index()) {
                log::warn!("Edge {} stores different id {}", index.index(), edge.id().0);
                return false;
            }
            let (source, target) = self.0.edge_endpoints(index).unwrap();
            if edge.get_source() != NodeId(source.index())
                || edge.get_target() != NodeId(target.index())
            {
                log::warn!(
                    "Edge {} stores different source and target ({}, {}), expect ({}, {})",
                    index.index(),
                    edge.get_source().0,
                    edge.get_target().0,
                    source.index(),
                    target.index()
                );
                return false;
            }
        }
        true
    }
}

pub struct DirectedGraphNodeIterator<'a, N, E> {
    graph: &'a StableGraph<N, E, Directed>,
    index: NodeIndices<'a, N>,
}

impl<'a, N, E> Iterator for DirectedGraphNodeIterator<'a, N, E> {
    type Item = &'a N;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index.next()?;
        Some(self.graph.node_weight(index).unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.index.size_hint()
    }
}

pub struct DirectedGraphEdgeIterator<'a, N, E> {
    graph: &'a StableGraph<N, E, Directed>,
    index: EdgeIndices<'a, E>,
}

impl<'a, N, E> Iterator for DirectedGraphEdgeIterator<'a, N, E> {
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index.next()?;
        Some(self.graph.edge_weight(index).unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.index.size_hint()
    }
}

#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct GraphIdMap(pub(super) HashMap<String, NodeId>);

impl Index<&str> for GraphIdMap {
    type Output = NodeId;

    fn index(&self, index: &str) -> &Self::Output {
        &self.0[index]
    }
}

impl GraphIdMap {
    pub fn new(obj: HashMap<String, NodeId>) -> Self {
        GraphIdMap(obj)
    }
}
