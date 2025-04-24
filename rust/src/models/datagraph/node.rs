use crate::models::basegraph::BaseNode;

use super::NodeId;
use hashbrown::HashMap;
use kgdata_core::models::{Entity, Statement, Value};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CellNode {
    pub id: NodeId,
    pub column: usize,
    pub row: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LiteralNode {
    pub id: NodeId,
    pub value: Value,
    pub value_prob: f64,
    pub is_in_context: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EntityNode {
    pub id: NodeId,
    pub entity_id: String,
    pub entity_prob: f64,
    pub is_in_context: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StatementNode {
    pub id: NodeId,
    // the row number (or subgraph) of which this statement is belong to
    // as the statement is always connected between cell-entity or cell-cell
    // we always have the row number. note that this means multi-hop relationship
    // isn't supported. one way maybe is to create a pseudo-node that capture
    // multihop relationships.
    pub row: usize,
    // id of the entity that contains the statement
    pub entity_id: String,
    // predicate of the statement
    pub predicate: String,
    // index of the statement
    pub statement_index: usize,
    // whether this statement actually exist in KG
    pub is_in_kg: bool,
}

impl StatementNode {
    pub fn get_statement<'t>(&self, ents: &'t HashMap<String, Entity>) -> &'t Statement {
        &ents[&self.entity_id].props[&self.predicate][self.statement_index]
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Node {
    Cell(CellNode),
    Literal(LiteralNode),
    Entity(EntityNode),
    Statement(StatementNode),
}

impl Node {
    pub fn id(&self) -> NodeId {
        match self {
            Node::Cell(node) => node.id,
            Node::Literal(node) => node.id,
            Node::Entity(node) => node.id,
            Node::Statement(node) => node.id,
        }
    }

    pub fn set_id(&mut self, id: NodeId) {
        match self {
            Node::Cell(node) => node.id = id,
            Node::Literal(node) => node.id = id,
            Node::Entity(node) => node.id = id,
            Node::Statement(node) => node.id = id,
        }
    }

    pub fn is_cell(&self) -> bool {
        match self {
            Node::Cell(_) => true,
            _ => false,
        }
    }

    pub fn is_statement(&self) -> bool {
        match self {
            Node::Statement(_) => true,
            _ => false,
        }
    }

    pub fn as_cell(&self) -> &CellNode {
        match self {
            Node::Cell(node) => node,
            _ => panic!("not a cell node"),
        }
    }

    pub fn as_statement(&self) -> &StatementNode {
        match self {
            Node::Statement(s) => s,
            _ => panic!("not a statement node"),
        }
    }
}

/// ID of a node in the data graph.
#[derive(Serialize, Deserialize, Clone, Debug, Copy, PartialEq, Eq, Hash, FromPyObject)]
pub struct DGNodeId(pub usize);

impl IntoPy<PyObject> for DGNodeId {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}

impl BaseNode for Node {
    fn id(&self) -> NodeId {
        match self {
            Node::Cell(node) => node.id,
            Node::Literal(node) => node.id,
            Node::Entity(node) => node.id,
            Node::Statement(node) => node.id,
        }
    }

    fn set_id(&mut self, id: NodeId) {
        match self {
            Node::Cell(node) => node.id = id,
            Node::Literal(node) => node.id = id,
            Node::Entity(node) => node.id = id,
            Node::Statement(node) => node.id = id,
        }
    }
}
