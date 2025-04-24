use hashbrown::HashSet;
use serde::{Deserialize, Serialize};

use crate::models::basegraph::{BaseNode, NodeId};
use crate::models::datagraph::{EntityNode, LiteralNode, StatementNode as DGStatementNode};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ColumnNode {
    pub id: NodeId,
    // column name
    pub label: String,
    // column index
    pub column: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StatementNode {
    pub id: NodeId,
    // list of associated statements in DG that help to create this statement node
    pub dgprov: DGStatementProv,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Node {
    Column(ColumnNode),
    Entity(EntityNode),
    Literal(LiteralNode),
    Statement(StatementNode),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DGStatementProv {
    // mapping from row number into list of statements in DG that constitute this statement
    // the statements should be unique per row and we enforce this during data validation
    row2stmts: Vec<Vec<NodeId>>,
    // list of statements in DG that constitute this statement between entities/literal nodes.
    // due to a statement always has a row, we do not need this.
    // pub ent_stmts: Vec<NodeId>,
}

impl Node {
    pub fn id(&self) -> NodeId {
        match self {
            Node::Column(node) => node.id,
            Node::Entity(node) => node.id,
            Node::Literal(node) => node.id,
            Node::Statement(node) => node.id,
        }
    }

    pub fn is_column(&self) -> bool {
        match self {
            Node::Column(_) => true,
            _ => false,
        }
    }

    pub fn is_statement(&self) -> bool {
        match self {
            Node::Statement(_) => true,
            _ => false,
        }
    }

    pub fn is_entity(&self) -> bool {
        match self {
            Node::Entity(_) => true,
            _ => false,
        }
    }

    pub fn is_literal(&self) -> bool {
        match self {
            Node::Literal(_) => true,
            _ => false,
        }
    }

    pub fn try_as_column(&self) -> Option<&ColumnNode> {
        match self {
            Node::Column(node) => Some(node),
            _ => None,
        }
    }

    pub fn try_as_statement(&self) -> Option<&StatementNode> {
        match self {
            Node::Statement(node) => Some(node),
            _ => None,
        }
    }

    pub fn try_as_entity(&self) -> Option<&EntityNode> {
        match self {
            Node::Entity(node) => Some(node),
            _ => None,
        }
    }

    pub fn try_as_literal(&self) -> Option<&LiteralNode> {
        match self {
            Node::Literal(node) => Some(node),
            _ => None,
        }
    }

    pub fn try_as_mut_statement(&mut self) -> Option<&mut StatementNode> {
        match self {
            Node::Statement(node) => Some(node),
            _ => None,
        }
    }
}

impl BaseNode for Node {
    fn id(&self) -> NodeId {
        match self {
            Node::Column(node) => node.id,
            Node::Entity(node) => node.id,
            Node::Literal(node) => node.id,
            Node::Statement(node) => node.id,
        }
    }

    fn set_id(&mut self, id: NodeId) {
        match self {
            Node::Column(node) => node.id = id,
            Node::Entity(node) => node.id = id,
            Node::Literal(node) => node.id = id,
            Node::Statement(node) => node.id = id,
        }
    }
}

impl DGStatementProv {
    pub fn new(nrows: usize) -> Self {
        DGStatementProv {
            row2stmts: vec![Vec::new(); nrows],
        }
    }

    pub fn track(&mut self, stmt: &DGStatementNode) {
        self.row2stmts[stmt.row].push(stmt.id);
    }

    pub fn is_valid(&self) -> bool {
        self.row2stmts
            .iter()
            .all(|lst| lst.iter().collect::<HashSet<_>>().len() == lst.len())
    }
}
