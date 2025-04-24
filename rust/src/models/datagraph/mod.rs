pub mod edge;
pub mod node;

pub use self::edge::{Edge, EdgeProv};
pub use self::node::{CellNode, EntityNode, LiteralNode, Node, StatementNode};
use super::basegraph::GraphIdMap;
use super::Table;
use crate::error::GramsError;
use crate::models::basegraph::{BaseDirectedGraph, NodeId};
use hashbrown::{HashMap, HashSet};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Data graph capturing relationships found in the table's data.
///
/// We use intermediate statement nodes to represent n-ary relationships. A statement in the data graph
/// is uniquely determine
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DGraph {
    pub graph: BaseDirectedGraph<Node, Edge>,
    pub nrows: usize,
    pub ncols: usize,

    // contains mapping from the entity/literal value to the node id
    entity2nodeid: Option<GraphIdMap>,
}

impl DGraph {
    pub fn new(table: &Table) -> DGraph {
        let (nrows, ncols) = table.shape();
        let mut graph = BaseDirectedGraph::default();

        for ri in 0..nrows {
            for ci in 0..ncols {
                graph.add_node(Node::Cell(CellNode {
                    id: NodeId(0), // id doesn't matter here
                    row: ri,
                    column: ci,
                }));
            }
        }

        DGraph {
            graph,
            nrows,
            ncols,
            entity2nodeid: None,
        }
    }

    /// Set a mapping from entity/literal to its node id in the graph.
    pub fn set_entity2nodeid(&mut self, context2nodeid: GraphIdMap) {
        self.entity2nodeid = Some(context2nodeid);
    }

    pub fn get_statement_node(&self, nodeid: NodeId) -> Option<&StatementNode> {
        if let Some(Node::Statement(s)) = self.graph.get_node(nodeid) {
            Some(s)
        } else {
            None
        }
    }

    pub fn get_mut_statement_node(&mut self, nodeid: NodeId) -> Option<&mut StatementNode> {
        if let Some(Node::Statement(s)) = self.graph.get_mut_node(nodeid) {
            Some(s)
        } else {
            None
        }
    }

    #[inline]
    pub fn get_cell_node_id(&self, row: usize, col: usize) -> NodeId {
        NodeId(row * self.ncols + col)
    }

    #[inline]
    pub fn get_entity_node_id(&self, entity_id: &str) -> NodeId {
        self.entity2nodeid.as_ref().unwrap()[entity_id]
    }

    /// Get a corresponding incoming edge to a statement from a cell of a given column
    /// such that the incoming edge connect to a statement and then connect to a given outgoing
    /// edge.
    pub fn get_row_inedge_from_cell<'t>(&'t self, outedge: &Edge, column: usize) -> &'t Edge {
        // TODO: there is another way to get the incoming edge faster
        // self.graph.get_edge_between_nodes(self.graph.get_cell_node_id(), outedge.source)
        let mut it = self.graph.iter_in_edges(outedge.source).filter(|inedge| {
            if let Node::Cell(c) = self.graph.get_node(inedge.source).unwrap() {
                if c.column == column {
                    return true;
                }
            }
            false
        });
        let inedge = it.next().unwrap();
        // it.next().is_none() due to non-parallel edge -- checked in is_valid
        inedge
    }

    /// Get a corresponding incoming edge to a statement from a given contextual entity
    /// such that the incoming edge connect to a statement and then connect to a given outgoing
    /// edge.
    pub fn get_row_inedge_from_entity<'t>(&'t self, outedge: &Edge, ent: &String) -> &'t Edge {
        let mut it = self.graph.iter_in_edges(outedge.source).filter(|inedge| {
            if let Node::Entity(e) = self.graph.get_node(inedge.source).unwrap() {
                if e.entity_id == *ent {
                    return true;
                }
            }
            false
        });
        let inedge = it.next().unwrap();
        // it.next().is_none() due to non-parallel edge -- checked in is_valid
        inedge
    }

    /// Dumps edges of the data graph into a data frame.
    ///
    /// * Cells are converted into cell:<row>-<col> format.
    /// * Entities are converted into entity:<id> format.
    /// * Literals are converted into literal:<id> format.
    /// * Statements are converted into statement:<id> format.
    pub fn to_dataframe(&self, table: &Table) -> Result<DataFrame, GramsError> {
        let mut idmap: Vec<String> = Vec::with_capacity(self.graph.num_nodes());

        for node in self.graph.iter_nodes() {
            match node {
                Node::Cell(u) => idmap.push(format!(
                    "cell:{}-{}:{}",
                    u.row,
                    u.column,
                    table.get_cell(u.row, u.column)
                )),
                Node::Entity(u) => idmap.push(format!("entity:{}", u.entity_id)),
                Node::Literal(u) => idmap.push(format!("literal:{}", u.value.to_string_repr())),
                Node::Statement(u) => idmap.push(format!(
                    "statement:{}:{}->{}[{}]",
                    u.row, u.entity_id, u.predicate, u.statement_index
                )),
            }
        }

        let nedges = self.graph.num_edges();
        let mut edgeids = Vec::with_capacity(nedges);
        let mut sources = Vec::with_capacity(nedges);
        let mut targets = Vec::with_capacity(nedges);
        let mut statements = Vec::with_capacity(nedges);
        let mut predicates = Vec::with_capacity(nedges);
        let mut probs = Vec::with_capacity(nedges);

        for node in self.graph.iter_nodes() {
            if !node.is_statement() {
                continue;
            }
            let nodeid = node.id();
            let stmt = idmap[nodeid.0].as_str();
            for inedge in self.graph.in_edges(nodeid) {
                let source = idmap[inedge.source.0].as_str();
                for outedge in self.graph.out_edges(nodeid) {
                    let target = idmap[outedge.target.0].as_str();

                    edgeids.push(outedge.id.0 as u32);
                    sources.push(source);
                    targets.push(target);
                    statements.push(stmt);
                    predicates.push(outedge.predicate.as_str());
                    probs.push(outedge.prov.as_ref().unwrap().0.prob);
                }
            }
        }

        Ok(DataFrame::new(vec![
            Series::new("edgeid", edgeids),
            Series::new("source", sources),
            Series::new("target", targets),
            Series::new("statement", statements),
            Series::new("predicate", predicates),
            Series::new("prob", probs),
        ])?)
    }

    /// Validate the integrity of the data graph, which include the following checks:
    ///
    /// * links from/to a statement are not cross different rows.
    /// * there is no link between cells, entities, and literals. All have to go through statements.
    /// * id of nodes and edges are consistent.
    /// * there is no parallel edges
    /// * there is no gap in the node id (i.e., all node ids are consecutive)
    pub fn is_valid(&self) -> bool {
        if !self.graph.is_valid() {
            return false;
        }

        // links are from/to a statement, there is no link between nodes of other types
        for edge in self.graph.iter_edges() {
            let source = self.graph.get_node(edge.source).unwrap();
            let target = self.graph.get_node(edge.target).unwrap();
            if !(source.is_statement() ^ target.is_statement()) {
                return false;
            }
        }

        // links from/to a statement are not cross different rows
        for node in self.graph.iter_nodes() {
            if !node.is_statement() {
                continue;
            }

            let mut rows = self
                .graph
                .iter_in_edges(node.id())
                .map(|e| {
                    if let Some(Node::Cell(u)) = self.graph.get_node(e.source) {
                        return Some(u.row);
                    }
                    return None;
                })
                .filter(|x| x.is_some())
                .map(|x| x.unwrap())
                .collect::<Vec<_>>();

            rows.extend(
                self.graph
                    .out_edges(node.id())
                    .iter()
                    .map(|&e| {
                        if let Some(Node::Cell(u)) = self.graph.get_node(e.target) {
                            return Some(u.row);
                        }
                        return None;
                    })
                    .filter(|x| x.is_some())
                    .map(|x| x.unwrap())
                    .collect::<Vec<_>>(),
            );

            if HashSet::<usize>::from_iter(rows.into_iter()).len() != 1 {
                return false;
            }
        }

        // there is no parallel edge that have the same predicate
        for node in self.graph.iter_nodes() {
            let outedges = self.graph.out_edges(node.id());
            if outedges
                .iter()
                .map(|e| (e.target, &e.predicate))
                .collect::<HashSet<_>>()
                .len()
                != outedges.len()
            {
                return false;
            }
        }

        // node ids are consecutive
        let mut counter = 0;
        for node in self.graph.iter_nodes() {
            if node.id().0 != counter {
                return false;
            }
            counter += 1;
        }
        true
    }
}

impl From<&DGraph> for GraphIdMap {
    fn from(value: &DGraph) -> Self {
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
