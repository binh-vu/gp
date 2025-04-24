use crate::error::into_pyerr;
use crate::models::basegraph::NodeId;
use crate::models::cangraph::{ColumnNode, Node, StatementNode};
use crate::models::datagraph::{EntityNode, LiteralNode};
use kgdata_core::pyo3helper::unsafe_update_view_lifetime_signature;
use kgdata_core::python::models::ValueView;
use kgdata_core::{pyview, pywrap};
use postcard::{from_bytes, to_allocvec};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};

// we wrap the node as python will own this object
// but for column nodes, we will return a view
pywrap!(PyCGNode(module = "gp_core.models", name = "CGNode", cls = Node, derive = (Serialize, Deserialize)) {
    f(id: NodeId),
    f(is_column: bool),
    f(is_statement: bool),
    f(is_entity: bool),
    f(is_literal: bool)
});
pyview!(PyCGColumnNode(module = "gp_core.models", name = "CGColumnNode", cls = ColumnNode) {
    c(id: NodeId),
    b(label: String),
    c(column: usize)
});
pyview!(PyCGStatementNode(module = "gp_core.models", name = "CGStatementNode", cls = StatementNode) {
    c(id: NodeId),
});
pyview!(PyCGEntityNode(module = "gp_core.models", name = "CGEntityNode", cls = EntityNode) {
    c(id: NodeId),
    b(entity_id: String),
    c(entity_prob: f64),
    c(is_in_context: bool),
});
pyview!(PyCGLiteralNode(module = "gp_core.models", name = "CGLiteralNode", cls = LiteralNode) {
    c(id: NodeId),
    c(value_prob: f64),
    c(is_in_context: bool),
});

#[pymethods]
impl PyCGNode {
    #[new]
    pub fn default() -> Self {
        PyCGNode(Node::Column(ColumnNode {
            id: NodeId(0),
            label: "".to_string(),
            column: 0,
        }))
    }

    pub fn try_as_column(&self) -> Option<PyCGColumnNode> {
        self.0.try_as_column().map(PyCGColumnNode::new)
    }

    pub fn try_as_statement(&self) -> Option<PyCGStatementNode> {
        self.0.try_as_statement().map(PyCGStatementNode::new)
    }

    pub fn try_as_entity(&self) -> Option<PyCGEntityNode> {
        self.0.try_as_entity().map(PyCGEntityNode::new)
    }

    pub fn try_as_literal(&self) -> Option<PyCGLiteralNode> {
        self.0.try_as_literal().map(PyCGLiteralNode::new)
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        let out = to_allocvec(&self).map_err(into_pyerr)?;
        Ok(PyBytes::new(py, &out))
    }

    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = from_bytes::<PyCGNode>(state.as_bytes()).map_err(into_pyerr)?;
        Ok(())
    }

    pub fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pymethods]
impl PyCGLiteralNode {
    #[getter]
    pub fn value(&self) -> ValueView {
        ValueView::new(&self.0.value)
    }
}

impl IntoPy<PyObject> for NodeId {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}
