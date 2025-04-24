use hashbrown::HashMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::libs::literal_matchers::ParsedTextRepr;

use super::datagraph as dgraph;

#[pyclass(module = "gp_core.models", name = "Table")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    #[pyo3(get)]
    pub id: String,
    pub links: Vec<Vec<Vec<Link>>>,
    #[pyo3(get)]
    pub columns: Vec<Column>,
    #[pyo3(get)]
    pub context: Context,
}

impl Table {
    pub fn shape(&self) -> (usize, usize) {
        if self.columns.len() == 0 {
            (0, 0)
        } else {
            (self.columns[0].values.len(), self.columns.len())
        }
    }

    #[inline]
    pub fn n_rows(&self) -> usize {
        self.links.len()
    }

    #[inline]
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Get number of cells in the table
    #[inline]
    pub fn size(&self) -> usize {
        return self.n_rows() * self.n_cols();
    }

    pub fn get_cell(&self, row: usize, col: usize) -> &str {
        &self.columns[col].values[row]
    }
}

#[pyclass(module = "gp_core.models", name = "Context")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    #[pyo3(get)]
    pub page_title: Option<String>,
    #[pyo3(get)]
    pub page_url: Option<String>,
    #[pyo3(get)]
    pub page_entities: Vec<CandidateEntityId>,
}

#[pyclass(module = "gp_core.models", name = "Link")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
    #[pyo3(get)]
    pub url: Option<String>,
    #[pyo3(get)]
    pub entities: Vec<EntityId>,
    #[pyo3(get)]
    pub candidates: Vec<CandidateEntityId>,
}

#[pyclass(module = "gp_core.models", name = "EntityId")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EntityId(pub String);

impl hashbrown::Equivalent<EntityId> for &String {
    fn equivalent(&self, key: &EntityId) -> bool {
        *self == &key.0
    }
}

#[pyclass(module = "gp_core.models", name = "CandidateEntityId")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateEntityId {
    #[pyo3(get)]
    pub id: EntityId,
    #[pyo3(get)]
    pub probability: f64,
}

#[pyclass(module = "gp_core.models", name = "Column")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub name: Option<String>,
    #[pyo3(get)]
    pub values: Vec<String>,
}

#[pyclass(module = "gp_core.models", name = "TableCells")]
#[derive(Serialize, Deserialize)]
pub struct TableCells(pub Vec<Vec<ParsedTextRepr>>);

// contains mapping from (entity id -> (probability, candidate rank)) for each cell in the table
// identify by row * ncols + column
pub struct CellCandidateEntities(Vec<HashMap<String, (f64, usize)>>);

impl CellCandidateEntities {
    pub fn new(index: Vec<HashMap<String, (f64, usize)>>) -> Self {
        Self(index)
    }

    #[inline]
    pub fn get_index(&self, dg: &dgraph::DGraph, u: &dgraph::CellNode) -> usize {
        dg.ncols * u.row + u.column
    }

    #[inline]
    pub fn get_prob(&self, cell: usize, id: &str) -> f64 {
        self.0[cell][id].0
    }

    #[inline]
    pub fn get_rank(&self, cell: usize, id: &str) -> usize {
        self.0[cell][id].1
    }

    #[inline]
    pub fn iter_candidate_entities(&self, cell: usize) -> impl Iterator<Item = (&String, &f64)> {
        self.0[cell].iter().map(|(k, v)| (k, &v.0))
    }

    #[inline]
    pub fn get_num_candidates(&self, cell: usize) -> usize {
        self.0[cell].len()
    }
}
