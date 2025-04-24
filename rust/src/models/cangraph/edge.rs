use super::super::datagraph as dgraph;
use crate::models::basegraph::{BaseEdge, EdgeId, NodeId};
use hashbrown::HashSet;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Edge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub predicate: String,
    pub is_qualifier: bool,
    // trace of edges in the data graph that constitute this edge
    // none for incoming edge of a statement
    pub dgprov: Option<DGEdgeProv>,
}

/// For tracing edges in the data graph that constitute this edge.
/// Note that in the data graph, per row can have multiple edges so this list
/// only contains the edge with the highest prob. per row if exists (greedy choice).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DGEdgeProv {
    // mapping from row number into list of edges in DG that constitute this edge
    // the edges should be unique per row and we enforce this during data validation
    row2edges: Vec<Vec<EdgeId>>,
}

impl std::fmt::Display for DGEdgeProv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tmp = self
            .row2edges
            .iter()
            .enumerate()
            .filter(|rrow| rrow.1.len() > 0)
            .fold(String::new(), |mut a, b| {
                let row = if a.len() > 0 {
                    format!(";{}:", b.0)
                } else {
                    format!("{}:", b.0)
                };

                let s = b.1.iter().fold(String::new(), |mut a, b| {
                    if a.len() > 0 {
                        a.push_str(",");
                    }
                    a.push_str(&b.to_string());
                    a
                });
                a.reserve(row.len() + s.len());
                a.push_str(&row);
                a.push_str(&s);
                a
            });
        write!(f, "{}", tmp)
    }
}

impl DGEdgeProv {
    pub fn new(nrows: usize) -> DGEdgeProv {
        DGEdgeProv {
            row2edges: vec![Vec::new(); nrows],
        }
    }

    pub fn track(&mut self, row: usize, edge: &dgraph::Edge) {
        self.row2edges[row].push(edge.id);
    }

    /// Get number of rows that have at least one edge.
    /// Complexity: O(nrows)
    pub fn get_num_matched_rows(&self) -> usize {
        self.row2edges.iter().filter(|row| row.len() > 0).count()
    }

    /// Iterate through edges of each row.
    /// Complexity: O(nrows)
    pub fn iter_all_edges<'t>(&'t self) -> impl Iterator<Item = &Vec<EdgeId>> + 't {
        self.row2edges.iter().filter(|row| row.len() > 0)
    }

    /// Iterate through edges of each row and yield the row number and the list of edges.
    /// Complexity: O(nrows)
    pub fn enumerate_all_edges<'t>(&'t self) -> impl Iterator<Item = (usize, &Vec<EdgeId>)> + 't {
        self.row2edges
            .iter()
            .enumerate()
            .filter(|(_ri, row)| row.len() > 0)
    }

    // /// Iterate over each best edge per row that constitutes to create the
    // /// edge in the candidate graph.
    // pub fn iter_best_edges<'t>(&'t self, dg: &dgraph::DGraph) -> impl Iterator<Item = EdgeId> + 't {
    //     if self.is_sorted {
    //         self.row2edges.iter().filter_map(|rowedges| {
    //             if rowedges.len() > 0 {
    //                 Some(rowedges[0])
    //             } else {
    //                 None
    //             }
    //         })
    //     } else {
    //         unimplemented!()
    //     }
    // }

    /// sort the edges so that searching will be faster.
    // pub fn optimize(
    //     &mut self,
    //     dg: &dgraph::DGraph,
    //     context: &AlgoContext,
    //     source_statement: ColumnOrEntity,
    //     cell2entscore: &CellCandidateEntities,
    // ) {
    //     for edges in self.row2edges.iter_mut() {
    //         edges.sort_by_cached_key(|eid| {
    //             let outedge = dg.graph.get_edge(*eid).unwrap();
    //             let inedge = match &source_statement {
    //                 ColumnOrEntity::Column(col) => {
    //                     let mut it = dg.graph.iter_in_edges(outedge.source).filter(|inedge| {
    //                         if let dgraph::Node::Cell(c) = dg.graph.get_node(inedge.source).unwrap()
    //                         {
    //                             if c.column == *col {
    //                                 return true;
    //                             }
    //                         }
    //                         false
    //                     });
    //                     let inedge = it.next().unwrap();
    //                     assert!(it.next().is_none());
    //                     inedge
    //                 }
    //                 ColumnOrEntity::Entity(entid) => {
    //                     let mut it = dg.graph.iter_in_edges(outedge.source).filter(|inedge| {
    //                         if let dgraph::Node::Entity(e) =
    //                             dg.graph.get_node(inedge.source).unwrap()
    //                         {
    //                             if e.entity_id == *entid {
    //                                 return true;
    //                             }
    //                         }
    //                         false
    //                     });
    //                     let inedge = it.next().unwrap();
    //                     assert!(it.next().is_none());
    //                     inedge
    //                 }
    //             };

    //             let score =
    //                 dgraph::Edge::get_full_match_prob(inedge, outedge, dg, context, &cell2entscore);

    //             return (
    //                 OrdF64(score.predicate),
    //                 OrdF64(score.source),
    //                 OrdF64(score.target),
    //             );
    //         });
    //     }
    // }

    /// check if this provenance is valid.
    pub fn is_valid(&self) -> bool {
        self.row2edges
            .iter()
            .all(|lst| lst.iter().collect::<HashSet<_>>().len() == lst.len())
    }
}

// pub struct DGEdgeProvBuilder {
//     // mapping from row number to position in DGEdgeProv's list.
//     rowmap: Vec<usize>,
// }

// impl DGEdgeProvBuilder {
//     pub fn new(table: &Table) -> Self {
//         let nrows = table.n_rows();
//         DGEdgeProvBuilder {
//             rowmap: vec![nrows + 1; nrows],
//         }
//     }
// }

// impl DGEdgeProv {
//     pub fn iter_dg_edges(&self) -> &[EdgeId] {
//         return &self.0;
//     }

//     // track this edge in the data graph.
//     pub fn track2(
//         &mut self,
//         prov_builder: &mut DGEdgeProvBuilder,
//         dg: &dgraph::DGraph,
//         dgedge: &dgraph::Edge,
//     ) {
//         // figure out the row of this edge so we can merge edges of the same row together.
//         let edgerow = self.get_edge_row(dg, dgedge);

//         // merge prov.
//         if prov_builder.rowmap[edgerow] > self.0.len() {
//             // new row, add this edge to the list.
//             prov_builder.rowmap[edgerow] = self.0.len();
//             self.0.push(dgedge.id);
//         } else {
//             // old row, merge it
//             let prevedge = self.0[prov_builder.rowmap[edgerow]];
//             if dgedge.prov.as_ref().unwrap()
//                 > dg.graph.get_edge(prevedge).unwrap().prov.as_ref().unwrap()
//             {
//                 // replace the old edge with the new one.
//                 self.0[prov_builder.rowmap[edgerow]] = dgedge.id;
//             }
//         }
//     }

// fn get_edge_row(&self, dg: &dgraph::DGraph, edge: &dgraph::Edge) -> usize {
//     if let Some(dgraph::Node::Cell(v)) = dg.graph.get_node(edge.target) {
//         v.row
//     } else if let Some(dgraph::Node::Cell(u)) = dg
//         .graph
//         .get_node(dg.graph.iter_in_edges(edge.source).next().unwrap().source)
//     {
//         u.row
//     } else {
//         unreachable!()
//     }
// }
// }

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
