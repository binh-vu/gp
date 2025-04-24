pub mod basegraph;
pub mod cangraph;
pub mod context;
pub mod datagraph;
pub mod db;
pub mod table;

use std::cmp::Ordering;

pub use self::context::AlgoContext;
pub use self::db::{LocalGramsDB, RemoteGramsDB};
pub use self::table::{
    CandidateEntityId, CellCandidateEntities, Column, EntityId, Link, Table, TableCells,
};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass(module = "gp_core.models", name = "MatchMethod")]
#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum MatchMethod {
    LiteralMatching,
    LinkMatching,
}

#[pyclass(module = "gp_core.models", name = "Match")]
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Match {
    pub prob: f64,
    pub method: MatchMethod,
}

impl PartialEq for Match {
    fn eq(&self, other: &Match) -> bool {
        self.prob == other.prob && self.method == other.method
    }
}

impl PartialOrd for Match {
    fn partial_cmp(&self, other: &Match) -> Option<Ordering> {
        if self.prob == other.prob {
            if self.method == other.method {
                Some(Ordering::Equal)
            } else {
                match self.method {
                    MatchMethod::LiteralMatching => Some(Ordering::Less),
                    MatchMethod::LinkMatching => Some(Ordering::Greater),
                }
            }
        } else {
            self.prob.partial_cmp(&other.prob)
        }
    }
}

impl Eq for Match {}

impl Ord for Match {
    fn cmp(&self, other: &Match) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl IntoPy<PyObject> for &Match {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.clone().into_py(py)
    }
}

impl IntoPy<PyObject> for &MatchMethod {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.clone().into_py(py)
    }
}
