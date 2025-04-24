use std::path::PathBuf;

use crate::{
    error::GramsError,
    models::{
        basegraph::{EdgeId, GraphIdMap, NodeId},
        cangraph as cgraph,
        datagraph::{self as dgraph},
        db::{BaseGramsDB, CacheRocksDBDict},
        table::CellCandidateEntities,
        AlgoContext, CandidateEntityId, Table,
    },
};
use hashbrown::{HashMap, HashSet};
use kgdata_core::{
    db::{open_class_db, open_entity_outlink_db, Map, PredefinedDB},
    models::{kgns::KnowledgeGraphNamespace, Class, Entity, EntityMetadata, EntityOutLink},
};

/// Struct to calculate features for type and relation inference
pub struct FeatureStore<'t> {
    pub table: &'t Table,
    pub context: &'t mut AlgoContext,
    pub cg: &'t cgraph::CGraph,
    pub dg: &'t dgraph::DGraph,
    pub kgns: &'t KnowledgeGraphNamespace,
}

impl<'t> FeatureStore<'t> {
    pub fn new<ED: Map<String, Entity>, EMD: Map<String, EntityMetadata>>(
        table: &'t Table,
        db: &'t BaseGramsDB<ED, EMD>,
        context: &'t mut AlgoContext,
        cg: &'t cgraph::CGraph,
        dg: &'t dgraph::DGraph,
        parallel: bool,
    ) -> Result<Self, GramsError> {
        if !context.has_called_prefetched_property() {
            context.prefetch_property(db, parallel)?;
        }

        Ok(FeatureStore {
            table,
            context,
            cg,
            dg,
            kgns: &db.0.kgns,
        })
    }

    #[inline]
    pub fn iter_cell_candidate_entities(
        &self,
        row: usize,
        col: usize,
    ) -> impl Iterator<Item = &CandidateEntityId> {
        self.table.links[row][col]
            .iter()
            .flat_map(|l| l.candidates.iter())
    }
}

/// Struct to cache necessary data needed for feature calculation.
pub struct FeatureStoreCache {
    /// for caching functions -- this is the convention for naming
    /// cacheable function and variable:
    ///
    /// - function: get__{name}: get the value from the cache, panic if not exist.
    /// - function: get_or_compute__{name}: get the value from the cache, compute if not exist.
    /// - function: compute__{name}: compute the value and store it in the cache if not exist.
    /// - variable: {name}: the variable storing the cache
    ///
    /// this is to allow us to use the cache in two phases: borrow mutable then drop them and borrow immutable
    pub(super) rel_freq: HashMap<EdgeId, f64>,
    pub(super) rel_freq_topk: HashMap<EdgeId, f64>,
    pub(super) rel_prob: HashMap<EdgeId, f64>,
    pub(super) maximum_possible_ent_links_between_two_nodes: HashMap<EdgeId, usize>,

    // /// the following cache objects are storing items that are one-to-one mapping with &[TargetRelationship]
    // /// as the feature store is supposed to be used for a single candidate graph, &[TargetRelationship] isn't
    // /// supposed to change during each call to the store cache
    // pub(super) max_num_entity_rows: Option<Vec<usize>>,
    // pub(super) max_num_pos_rels: Option<Vec<usize>>,
    // pub(super) rel_freqs: Option<Vec<f64>>,
    // pub(super) topk_rel_freqs: HashMap<usize, Vec<usize>>, // mapping from topk -> rel_freqs
    // pub(super) rel_num_unmatch_links: Vec<(f64, Vec<usize>)>, // as f64 is unhashable, we loop through the vector and manually compare

    // for caching db -- missing object detection
    pub entity_outlinks: CacheRocksDBDict<EntityOutLink>,
    pub classes: CacheRocksDBDict<Class>,

    // to get entity and literal nodes
    pub dg_idmap: GraphIdMap,
    pub cg_idmap: GraphIdMap,

    pub ncols: usize,
    // contains mapping from (entity id -> (probability, candidate rank)) for each cell in the table
    // identify by row * ncols + column
    pub index2entscore: CellCandidateEntities,
}

impl FeatureStoreCache {
    pub fn new(featstore: &FeatureStore, datadir: &PathBuf) -> Result<Self, GramsError> {
        let mut index2entscore = vec![HashMap::new(); featstore.table.size()];
        let (nrows, ncols) = featstore.table.shape();
        for ri in 0..nrows {
            for ci in 0..ncols {
                let entscore = &mut index2entscore[ri * ncols + ci];
                for link in &featstore.table.links[ri][ci] {
                    for (j, can) in link.candidates.iter().enumerate() {
                        entscore.insert(can.id.0.clone(), (can.probability, j));
                    }
                }
            }
        }

        let entity_outlinks = CacheRocksDBDict::new(open_entity_outlink_db(
            datadir
                .join(PredefinedDB::EntityOutLink.get_dbname())
                .as_os_str(),
        )?);
        let classes = CacheRocksDBDict::new(open_class_db(
            datadir.join(PredefinedDB::Class.get_dbname()).as_os_str(),
        )?);

        Ok(Self {
            rel_freq: HashMap::new(),
            rel_freq_topk: HashMap::new(),
            rel_prob: HashMap::new(),
            maximum_possible_ent_links_between_two_nodes: HashMap::new(),

            // max_num_entity_rows: None,
            // max_num_pos_rels: None,
            // rel_freqs: None,
            // topk_rel_freqs: HashMap::new(),
            // rel_num_unmatch_links: Vec::new(),
            dg_idmap: featstore.dg.into(),
            cg_idmap: featstore.cg.into(),
            ncols,
            index2entscore: CellCandidateEntities::new(index2entscore),
            entity_outlinks,
            classes,
        })
    }

    /// Find the maximum possible links between two nodes (ignore the possible predicates):
    ///
    /// Let M be the maximum possible links we want to find, N is the number of rows in the table.
    /// 1. If two nodes are not columns, M is 1 because it's entity to entity link.
    /// 2. If one node is a column, M = N - U, where U is the number of pairs that cannot have KG discovered links.
    /// A pair that cannot have KG discovered links is:
    ///     a. If both nodes are columns, and the link is
    ///         * data predicate: the source cell links to no entity.
    ///         * object predicate: the source or target cell link to no entity
    ///     b. If only one node is column, and the link is
    ///         * data predicate:
    ///             - if the source node must be an entity, then the target must be a column. U is always 0
    ///             - else then the source node must be is a column and target is a literal value: a cell in the column links to no entity
    ///         * object predicate: a cell in the column links to no entity.
    pub fn get_maximum_possible_ent_links_between_two_nodes(
        &mut self,
        store: &mut FeatureStore,
        _s: &cgraph::StatementNode,
        inedge: &cgraph::Edge,
        outedge: &cgraph::Edge,
    ) -> Result<usize, GramsError> {
        fn compute(
            store: &mut FeatureStore,
            inedge: &cgraph::Edge,
            outedge: &cgraph::Edge,
        ) -> Result<usize, GramsError> {
            let table = store.table;
            let cg = store.cg;
            let cgu = cg.graph.get_node(inedge.source).unwrap();
            let cgv = cg.graph.get_node(outedge.target).unwrap();

            if !cgu.is_column() && !cgv.is_column() {
                return Ok(1);
            }

            // instead of going through each node attach to the node in the candidate graph, we avoid by directly generating the data node ID
            let nrows = store.table.n_rows();
            let mut n_null_entities = 0;
            let is_data_predicate = store.context.props[&outedge.predicate].is_data_property();

            if is_data_predicate {
                match cgu {
                    cgraph::Node::Column(n) => {
                        for ri in 0..nrows {
                            if table.links[ri][n.column]
                                .iter()
                                .map(|l| l.candidates.len())
                                .sum::<usize>()
                                == 0
                            {
                                n_null_entities += 1;
                            }
                        }
                    }
                    cgraph::Node::Entity(_) => {
                        assert!(cgv.is_column());
                        return Ok(nrows);
                    }
                    _ => unreachable!(),
                }
            } else {
                match cgu {
                    cgraph::Node::Column(n) => match cgv {
                        cgraph::Node::Column(n2) => {
                            for ri in 0..nrows {
                                if table.links[ri][n.column]
                                    .iter()
                                    .map(|l| l.candidates.len())
                                    .sum::<usize>()
                                    + table.links[ri][n2.column]
                                        .iter()
                                        .map(|l| l.candidates.len())
                                        .sum::<usize>()
                                    == 0
                                {
                                    n_null_entities += 1;
                                }
                            }
                        }
                        _ => {
                            assert!(!cgv.is_statement());
                            for ri in 0..nrows {
                                if table.links[ri][n.column]
                                    .iter()
                                    .map(|l| l.candidates.len())
                                    .sum::<usize>()
                                    == 0
                                {
                                    n_null_entities += 1;
                                }
                            }
                        }
                    },
                    cgraph::Node::Entity(_) => {
                        let ci = cgv.try_as_column().unwrap().column;
                        for ri in 0..nrows {
                            if table.links[ri][ci]
                                .iter()
                                .map(|l| l.candidates.len())
                                .sum::<usize>()
                                == 0
                            {
                                n_null_entities += 1;
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }

            Ok(nrows - n_null_entities)
        }

        if !self
            .maximum_possible_ent_links_between_two_nodes
            .contains_key(&outedge.id)
        {
            self.maximum_possible_ent_links_between_two_nodes
                .insert(outedge.id, compute(store, inedge, outedge)?);
        }
        Ok(self.maximum_possible_ent_links_between_two_nodes[&outedge.id])
    }

    /// Get number of discovered links that don't match due to value differences. This function do not count if:
    /// - the link between two DG nodes is impossible
    /// - the property/qualifier do not exist in the entity
    pub fn get_unmatch_discovered_links(
        &self,
        store: &mut FeatureStore,
        _statement: &cgraph::StatementNode,
        inedge: &cgraph::Edge,
        outedge: &cgraph::Edge,
    ) -> Result<usize, GramsError> {
        let mut n_unmatch_links = 0;

        let uv_links = store
            .cg
            .get_dg_node_pairs(store.dg, outedge)
            .into_iter()
            .collect::<HashSet<_>>();

        let is_outpred_data_predicate = store.context.props[&outedge.predicate].is_data_property();

        let entities = &store.context.entities;
        let cgu = store.cg.graph.get_node(inedge.source).unwrap();
        let cgv = store.cg.graph.get_node(outedge.target).unwrap();

        for (dguid, dgvid) in self.iter_dg_pair(store, cgu, cgv) {
            // if has link, then we don't have to count
            if uv_links.contains(&(dguid, dgvid)) {
                continue;
            }

            let dgu = store.dg.graph.get_node(dguid).unwrap();
            let dgv = store.dg.graph.get_node(dgvid).unwrap();

            // ignore pairs that can't have any links
            if !self.dg_pair_has_possible_ent_links(store, dgu, dgv, is_outpred_data_predicate) {
                continue;
            }

            let no_data = match dgu {
                dgraph::Node::Cell(c) => store.table.links[c.row][c.column].iter().all(|l| {
                    l.candidates.iter().all(|eid| {
                        !does_ent_have_data(
                            &entities[&eid.id.0],
                            &inedge.predicate,
                            &outedge.predicate,
                            outedge.is_qualifier,
                        )
                    })
                }),
                dgraph::Node::Entity(entity) => !does_ent_have_data(
                    &entities[&entity.entity_id],
                    &inedge.predicate,
                    &outedge.predicate,
                    outedge.is_qualifier,
                ),
                _ => {
                    return Err(GramsError::IntegrityError(
                        "Source node must be cell or entity".to_owned(),
                    ))
                }
            };
            if no_data {
                continue;
            }
            n_unmatch_links += 1;
        }

        Ok(n_unmatch_links)
    }

    /// This function iterate through each pair of data graph nodes between two candidate graph nodes.
    ///
    /// If both cg nodes are entities, we only have one pair.
    /// If one or all of them are columns, the number of pairs will be the size of the table.
    /// Otherwise, not support iterating between nodes & statements.
    ///
    /// Note: this does not take into account whether there is a discovered relationship
    /// between the dg pair.       
    ///
    /// # Arguments
    ///
    /// * `u` - The source candidate graph node
    /// * `v` - The target candidate graph node
    ///
    /// # Returns
    ///
    /// This function returns an iterator of pairs of data graph nodes
    pub fn iter_dg_pair<'t>(
        &self,
        store: &'t FeatureStore,
        u: &cgraph::Node,
        v: &cgraph::Node,
    ) -> Box<dyn Iterator<Item = (NodeId, NodeId)> + 't> {
        let nrows = store.table.n_rows();
        let dg = store.dg;

        if u.is_column() && v.is_column() {
            let uci = u.try_as_column().unwrap().column;
            let vci = v.try_as_column().unwrap().column;

            return Box::new(
                (0..nrows)
                    .map(move |ri| (dg.get_cell_node_id(ri, uci), dg.get_cell_node_id(ri, vci))),
            );
        }

        if u.is_column() {
            let uci = u.try_as_column().unwrap().column;
            let vid = match v {
                cgraph::Node::Entity(v) => self.dg_idmap[&v.entity_id],
                cgraph::Node::Literal(v) => self.dg_idmap[&v.value.to_string_repr()],
                _ => unreachable!(),
            };
            return Box::new((0..nrows).map(move |i| (dg.get_cell_node_id(i, uci), vid)));
        }

        if v.is_column() {
            let uid = match u {
                cgraph::Node::Entity(u) => self.dg_idmap[&u.entity_id],
                cgraph::Node::Literal(u) => self.dg_idmap[&u.value.to_string_repr()],
                _ => unreachable!(),
            };
            let vci = v.try_as_column().unwrap().column;

            return Box::new((0..nrows).map(move |i| (uid, dg.get_cell_node_id(i, vci))));
        }

        Box::new(
            [(
                match u {
                    cgraph::Node::Entity(u) => self.dg_idmap[&u.entity_id],
                    cgraph::Node::Literal(u) => self.dg_idmap[&u.value.to_string_repr()],
                    _ => unreachable!(),
                },
                match v {
                    cgraph::Node::Entity(v) => self.dg_idmap[&v.entity_id],
                    cgraph::Node::Literal(v) => self.dg_idmap[&v.value.to_string_repr()],
                    _ => unreachable!(),
                },
            )]
            .into_iter(),
        )
    }

    pub fn dg_pair_has_possible_ent_links(
        &self,
        store: &FeatureStore,
        dgu: &dgraph::Node,
        dgv: &dgraph::Node,
        is_data_predicate: bool,
    ) -> bool {
        let is_dgu_cell = dgu.is_cell();
        let is_dgv_cell = dgv.is_cell();
        if is_dgu_cell && is_dgv_cell {
            // both are cells
            if is_data_predicate {
                let nu = dgu.as_cell();
                // data predicate: source cell must link to some entities to have possible links
                return self.has_candidates(store, nu.row, nu.column);
            } else {
                let nu = dgu.as_cell();
                let nv = dgv.as_cell();
                // object predicate: source cell and target cell must link to some entities to have possible links
                return self.has_candidates(store, nu.row, nu.column)
                    && self.has_candidates(store, nv.row, nv.column);
            }
        } else if is_dgu_cell {
            let nu = dgu.as_cell();
            // the source is cell, the target will be literal/entity value
            // we have link when source cell link to some entities, doesn't depend on type of predicate
            return self.has_candidates(store, nu.row, nu.column);
        } else if is_dgv_cell {
            // the target is cell, the source will be literal/entity value
            if is_data_predicate {
                // data predicate: always has possibe links
                return true;
            } else {
                // object predicate: have link when the target cell link to some entities
                let nv = dgv.as_cell();
                return self.has_candidates(store, nv.row, nv.column);
            }
        }
        // all cells are values, always have link due to how the link is generated in the first place
        return true;
    }

    pub fn has_candidates(&self, store: &FeatureStore, row: usize, col: usize) -> bool {
        store.table.links[row][col]
            .iter()
            .any(|l| !l.candidates.is_empty())
    }

    /// Get the corresponding source node of the statement node of a candidate graph in a data graph
    pub fn get_dg_statement_source(
        &self,
        dg: &dgraph::DGraph,
        cg: &cgraph::CGraph,
        cgsid: NodeId,
        dgsid: NodeId,
    ) -> NodeId {
        let cg_usedge = cg.graph.iter_in_edges(cgsid).next().unwrap();
        match cg.graph.get_node(cg_usedge.source).unwrap() {
            cgraph::Node::Column(u) => {
                let mut it = dg.graph.iter_in_edges(dgsid).filter(|dge| {
                    if let dgraph::Node::Cell(c) = dg.graph.get_node(dge.source).unwrap() {
                        if c.column == u.column {
                            return true;
                        }
                    }
                    return false;
                });

                let dguid = it.next().unwrap().source;
                assert!(it.next().is_none());
                return dguid;
            }
            cgraph::Node::Entity(u) => {
                let dguid = self.dg_idmap[&u.entity_id];
                assert_eq!(
                    dg.graph
                        .iter_in_edges(dgsid)
                        .filter(|dge| dge.source == dguid)
                        .count(),
                    1
                );
                return dguid;
            }
            _ => unreachable!(),
        }
    }
}

/// Test if an entity has a property/qualifier
#[inline(always)]
pub(super) fn does_ent_have_data(
    ent: &Entity,
    inedge_predicate: &str,
    outedge_predicate: &str,
    is_qualifier: bool,
) -> bool {
    if let Some(stmts) = ent.props.get(inedge_predicate) {
        !is_qualifier
            || (stmts
                .iter()
                .any(|stmt| stmt.qualifiers.contains_key(outedge_predicate)))
    } else {
        false
    }
}
