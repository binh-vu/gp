mod edge;
mod node;

pub use self::edge::{DGEdgeProv, Edge};
pub use self::node::{ColumnNode, DGStatementProv, Node, StatementNode};
use crate::{
    error::GramsError,
    models::basegraph::{BaseDirectedGraph, NodeId},
};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use super::basegraph::GraphIdMap;
use super::datagraph as dgraph;
use super::table::Table;
use polars::prelude::*;

/// Candidate graphs capturing potential relationships of the table. The relationships are
/// derived from the data graph in the following way:
///
/// First, dg edges that constitute statements' properties are grouped first (source, target, main property).
/// Then, for each statement, we add its qualifier. In data graph, a statement can have same property
/// to different targets. But in candidate graph, we create as many statement as the number of properties.
/// The qualifiers is then duplicated for each statement.
///
/// Also, we keep some provenance to trace back the links used to construct the relationships.
/// Note that the data graph can have more than one statement for a given property between
/// the same pairs (source, target) as an entity can have multiple statements for the same property (duplicated aren't
/// merged/resolved because those statements can have different qualifiers). Therefore, when iterating over dg
/// links that are used to construct cg links to calculate the relative frequency, we may encounter multiple statements
/// of the same pairs of the same row and we need to select one statement. Once choice that we use is selecting the one
/// with maximum probability. Note that this is a greedy choice because it doesn't consider other edges of these statements
/// as a whole.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CGraph {
    pub id: String, // id of the table
    pub graph: BaseDirectedGraph<Node, Edge>,
}

impl CGraph {
    pub fn new(table: &Table) -> CGraph {
        let mut graph = BaseDirectedGraph::default();
        for col in &table.columns {
            graph.add_node(Node::Column(ColumnNode {
                id: NodeId(0), // the id will be override
                label: col.name.clone().unwrap_or_default(),
                column: col.index,
            }));
        }

        CGraph {
            id: format!("cgraph:{}", table.id),
            graph,
        }
    }

    #[inline]
    pub fn get_column_node_id(&self, col: usize) -> NodeId {
        NodeId(col)
    }

    /// Convert graph into a data frame
    pub fn to_dataframe(&self, table: &Table) -> Result<DataFrame, GramsError> {
        let mut idmap: Vec<String> = Vec::with_capacity(self.graph.num_nodes());

        for node in self.graph.iter_nodes() {
            match node {
                Node::Column(u) => idmap.push(format!(
                    "column:{}:{}",
                    u.column,
                    table.columns[u.column]
                        .name
                        .as_ref()
                        .map(|x| x.as_str())
                        .unwrap_or("")
                )),
                Node::Entity(u) => idmap.push(format!("entity:{}", u.entity_id)),
                Node::Literal(u) => idmap.push(format!("literal:{}", u.value.to_string_repr())),
                Node::Statement(u) => idmap.push(format!(
                    "statement:{}:{}",
                    u.id,
                    self.graph.iter_in_edges(u.id).next().unwrap().predicate
                )),
            }
        }

        let nedges = self.graph.num_edges();
        let mut sources = Vec::with_capacity(nedges);
        let mut targets = Vec::with_capacity(nedges);
        let mut statements = Vec::with_capacity(nedges);
        let mut predicates = Vec::with_capacity(nedges);
        let mut dgprov = Vec::with_capacity(nedges);

        for node in self.graph.iter_nodes() {
            if !node.is_statement() {
                continue;
            }
            let nodeid = node.id();
            let stmt = idmap[nodeid.0].as_str();
            for inedge in self.graph.iter_in_edges(nodeid) {
                let source = idmap[inedge.source.0].as_str();
                for outedge in self.graph.out_edges(nodeid) {
                    let target = idmap[outedge.target.0].as_str();

                    sources.push(source);
                    targets.push(target);
                    statements.push(stmt);
                    predicates.push(outedge.predicate.as_str());
                    if let Some(prov) = outedge.dgprov.as_ref() {
                        dgprov.push(format!("{}", prov));
                    } else {
                        dgprov.push("".to_string());
                    }
                }
            }
        }

        Ok(DataFrame::new(vec![
            Series::new("source", sources),
            Series::new("target", targets),
            Series::new("statement", statements),
            Series::new("predicate", predicates),
            Series::new("dgprov", dgprov),
        ])?)
    }

    /// Get an incoming edge to a statement that has a given outgoing edge
    #[inline]
    pub fn get_inedge<'t>(&'t self, outedge: &Edge) -> &'t Edge {
        let mut it = self.graph.iter_in_edges(outedge.source);
        let inedge = it.next().unwrap();
        // it.next().is_none() -- this is checked in the is_valid() only one incoming edge to a statement
        inedge
    }

    /// Get list of pairs of source and target nodes (either cell or entity) in data graph that
    /// involved in the creation of this edge
    pub fn get_dg_node_pairs(&self, dg: &dgraph::DGraph, outedge: &Edge) -> Vec<(NodeId, NodeId)> {
        match self
            .graph
            .get_node(self.get_inedge(outedge).source)
            .unwrap()
        {
            Node::Column(u) => outedge
                .dgprov
                .as_ref()
                .unwrap()
                .enumerate_all_edges()
                .map(|(ri, rowedges)| {
                    let dguid = dg.get_cell_node_id(ri, u.column);
                    let dgvid = dg.graph.get_edge(rowedges[0]).unwrap().target;
                    (dguid, dgvid)
                })
                .collect::<Vec<_>>(),
            Node::Entity(u) => {
                let dguid = dg.get_entity_node_id(&u.entity_id);
                outedge
                    .dgprov
                    .as_ref()
                    .unwrap()
                    .iter_all_edges()
                    .map(|rowedges| {
                        let dgvid = dg.graph.get_edge(rowedges[0]).unwrap().target;
                        (dguid, dgvid)
                    })
                    .collect::<Vec<_>>()
            }
            _ => unreachable!(),
        }
    }

    // TODO: refactor this following method.
    // /// Get edges of the data graph that constitute this edge (including incoming &
    // /// outgoing edges of statements).
    // ///
    // /// Note that for each row, there can be multiple edges from different statements
    // /// that create the candidate edge.
    // pub fn get_dg_edges<'t>(
    //     &self,
    //     dg: &'t dgraph::DGraph,
    //     outedge: &Edge,
    // ) -> Vec<Vec<(&'t dgraph::Edge, &'t dgraph::Edge)>> {
    //     let inedge = self.get_inedge(outedge);
    //     let mut dg_edges = Vec::with_capacity(dg.nrows);

    //     match self.graph.get_node(inedge.source).unwrap() {
    //         Node::Column(cgu) => {
    //             for dgsveid in outedge.dgprov.unwrap().iter_best_edges(dg) {
    //                 let dgsvedge = dg.graph.get_edge(dgsveid).unwrap();
    //                 let mut it = dg.graph.iter_in_edges(dgsvedge.source).filter(|dge| {
    //                     if let dgraph::Node::Cell(c) = dg.graph.get_node(dge.source).unwrap() {
    //                         if c.column == cgu.column {
    //                             return true;
    //                         }
    //                     }
    //                     return false;
    //                 });

    //                 let dgusedge = it.next().unwrap();
    //                 // the reason we have multiple outedge is due to multiple statements
    //                 // and each statement has only one incoming edge to a cell from a particular
    //                 // entity, so we should have only one incoming edge.
    //                 assert!(it.next().is_none());
    //                 dg_edges.push((dgusedge, dgsvedge));
    //             }
    //         }
    //         Node::Entity(cgu) => {
    //             for dgsveid in outedge.dgprov.unwrap().iter_best_edges(dg) {
    //                 let dgsvedge = dg.graph.get_edge(dgsveid).unwrap();
    //                 let mut it = dg.graph.iter_in_edges(dgsvedge.source).filter(|dge| {
    //                     if let dgraph::Node::Entity(c) = dg.graph.get_node(dge.source).unwrap() {
    //                         if c.entity_id == cgu.entity_id {
    //                             return true;
    //                         }
    //                     }
    //                     return false;
    //                 });

    //                 let dgusedge = it.next().unwrap();
    //                 assert!(it.next().is_none());
    //                 dg_edges.push((dgusedge, dgsvedge));
    //             }
    //         }
    //         _ => unreachable!(),
    //     }

    //     dg_edges
    // }

    /// Validate the integrity of the candidate graph, which include the following checks:
    ///
    /// * only one incoming link toward a statement
    /// * provenance of nodes & edges are correct.
    /// * given an outedge of a statement in candidate graph cg_svedge, for each corresponding data graph edge of
    ///   a particular row, dg_svedge, we can determine its unique incoming edge dg_usedge based on the column
    ///   index/entity id without traversing the data graph.
    pub fn is_valid(&self, dg: &dgraph::DGraph) -> bool {
        if !self.graph.is_valid() {
            return false;
        }

        for node in self.graph.iter_nodes() {
            match node {
                Node::Statement(u) => {
                    if self.graph.iter_in_edges(u.id).count() != 1 {
                        log::warn!(
                            "statement node {} has {} incoming links",
                            u.id,
                            self.graph.iter_in_edges(u.id).count()
                        );
                        return false;
                    }
                    if !u.dgprov.is_valid() {
                        return false;
                    }
                }
                _ => {}
            }
        }

        for edge in self.graph.iter_edges() {
            if self.graph.get_node(edge.source).unwrap().is_statement() {
                // this is outedge of a statement
                if edge.dgprov.is_none() {
                    return false;
                }
            } else {
                // this is incoming edge to a statement
                if !self.graph.get_node(edge.target).unwrap().is_statement()
                    || edge.dgprov.is_some()
                {
                    return false;
                }
            }
            if let Some(dgprov) = edge.dgprov.as_ref() {
                if !dgprov.is_valid() {
                    return false;
                }

                // given an outedge of a statement in candidate graph cg_svedge, for each corresponding data graph edge of
                // a particular row, dg_svedge, we can determine its unique incoming edge dg_usedge based on the column
                // index/entity id without traversing the data graph.
                match self.graph.get_node(self.get_inedge(edge).source).unwrap() {
                    Node::Column(u) => {
                        for (row, rowedges) in dgprov.enumerate_all_edges() {
                            for dg_sveid in rowedges {
                                let dg_svedge = dg.graph.get_edge(*dg_sveid).unwrap();
                                let dguid = dg.get_row_inedge_from_cell(dg_svedge, u.column).source;
                                if dguid != dg.get_cell_node_id(row, u.column) {
                                    return false;
                                }
                            }
                        }
                    }
                    Node::Entity(u) => {
                        for rowedges in dgprov.iter_all_edges() {
                            for dg_sveid in rowedges {
                                let dg_svedge = dg.graph.get_edge(*dg_sveid).unwrap();
                                let dguid = dg
                                    .get_row_inedge_from_entity(dg_svedge, &u.entity_id)
                                    .source;
                                if dguid != dg.get_entity_node_id(&u.entity_id) {
                                    return false;
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        true
    }
}

impl From<&CGraph> for GraphIdMap {
    fn from(value: &CGraph) -> Self {
        let mut idmap = HashMap::new();
        for node in value.graph.iter_nodes() {
            match node {
                Node::Entity(u) => {
                    idmap.insert(u.entity_id.clone(), u.id);
                }
                Node::Literal(u) => {
                    idmap.insert(u.value.to_string_repr(), u.id);
                }
                _ => {}
            }
        }
        GraphIdMap(idmap)
    }
}
