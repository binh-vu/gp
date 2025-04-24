use super::db::{BaseGramsDB, LocalGramsDB};
use crate::{error::GramsError, libs::index::ObjectHop1Index};
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use kgdata_core::db::Map;
use kgdata_core::models::{kgns::KnowledgeGraphNamespace, Entity, EntityMetadata, Property};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A context object that contains the data needed for the algorithm to run for each table.
#[derive(Serialize, Deserialize)]
pub struct AlgoContext {
    pub entity_ids_level0: Vec<String>,
    pub entities: HashMap<String, Entity>,
    pub entity_metadata: HashMap<String, EntityMetadata>,
    pub index_object1hop: Option<ObjectHop1Index>,
    pub kgns: KnowledgeGraphNamespace,
    pub props: HashMap<String, Property>,
    called_prefetched_property: bool,
    called_prefetched_transitive_entities: bool,
}

impl AlgoContext {
    pub fn new(
        entity_ids: Vec<String>,
        entities: HashMap<String, Entity>,
        entity_metadata: HashMap<String, EntityMetadata>,
        kgns: KnowledgeGraphNamespace,
    ) -> Self {
        Self {
            entity_ids_level0: entity_ids,
            entities,
            entity_metadata,
            index_object1hop: None,
            props: HashMap::new(),
            kgns,
            called_prefetched_property: false,
            called_prefetched_transitive_entities: false,
        }
    }

    pub fn get_or_fetch_property(
        &mut self,
        pid: &str,
        db: &LocalGramsDB,
    ) -> Result<&Property, GramsError> {
        if !self.props.contains_key(pid) {
            let prop =
                db.0.props
                    .get(pid)?
                    .ok_or_else(|| GramsError::DBIntegrityError(pid.to_owned()))?;
            self.props.insert(pid.to_string(), prop);
        }
        Ok(self.props.get(pid).unwrap())
    }

    /// Initialize the 1-hop index for quick lookup. Have to do this separate from get_object_1hop_index because
    /// it extends the lifetime of self mutably borrowed, preventing us from borrow self immutably again.
    ///
    /// # Arguments
    ///
    /// * `necessary` - whether to initialize the index from list of entities. If false we create
    ///                 an empty index because we do not need it (such as when allow_ent_matching = false)
    /// * `parallel` - whether to parallelize the index creation
    pub fn init_object_1hop_index(&mut self, necessary: bool, parallel: bool) {
        let has_inited = if let Some(index_object1hop) = &self.index_object1hop {
            index_object1hop.index.len() > 0 || self.entity_ids_level0.is_empty()
        } else {
            false
        };

        if !has_inited {
            if necessary {
                if parallel {
                    self.index_object1hop = Some(ObjectHop1Index::par_from_entities(
                        &self.entity_ids_level0,
                        &self.entities,
                    ));
                } else {
                    self.index_object1hop = Some(ObjectHop1Index::from_entities(
                        &self.entity_ids_level0,
                        &self.entities,
                    ));
                }
            } else {
                self.index_object1hop = Some(ObjectHop1Index {
                    index: Default::default(),
                });
            }
        }
    }

    pub fn get_object_1hop_index(&self) -> &ObjectHop1Index {
        self.index_object1hop.as_ref().unwrap()
    }

    /// Prefetch all properties used by the entities
    pub fn prefetch_property<ED: Map<String, Entity>, EMD: Map<String, EntityMetadata>>(
        &mut self,
        db: &BaseGramsDB<ED, EMD>,
        parallel: bool,
    ) -> Result<(), GramsError> {
        let all_props = self
            .entities
            .values()
            .flat_map(|ent| {
                let mut keys: HashSet<&String> = HashSet::with_capacity(ent.props.len() * 2);
                for (pid, stmts) in ent.props.iter() {
                    keys.insert(pid);
                    for stmt in stmts {
                        keys.extend(stmt.qualifiers.keys());
                    }
                }
                keys
            })
            .unique()
            .collect::<Vec<_>>();

        let newprops = if parallel {
            db.0.props.par_slice_get_exist(&all_props)?
        } else {
            db.0.props.slice_get_exist(&all_props)?
        };

        self.props
            .extend(newprops.into_iter().map(|p| (p.id.clone(), p)));

        self.called_prefetched_property = true;
        Ok(())
    }

    /// Get transitive properties used by the entities in the context
    pub fn get_transitive_props(&self) -> HashSet<&String> {
        let mut transitive_props = HashSet::new();
        for prop in self.props.values() {
            if self.kgns.is_transitive_property(prop) {
                transitive_props.insert(&prop.id);
            }
        }
        transitive_props
    }

    pub fn fetch_transitive_entities(
        &mut self,
        db: &LocalGramsDB,
        parallel: bool,
    ) -> Result<(), GramsError> {
        if !self.called_prefetched_property {
            return Err(GramsError::LogicError(
                "Must call prefetch_property first".to_owned(),
            ));
        }

        let mut transitive_props = HashSet::new();
        for prop in self.props.values() {
            if self.kgns.is_transitive_property(prop) {
                transitive_props.insert(&prop.id);
            }
        }

        // retrieve all entities are have transitive properties
        let newentids = if parallel {
            self.entities
                .par_values()
                .flat_map(|ent| {
                    let mut newentids = Vec::new();
                    for &pid in &transitive_props {
                        if !ent.props.contains_key(pid) {
                            continue;
                        }

                        for stmt in &ent.props[pid] {
                            if let Some(next_ent_id) = stmt.value.as_entity_id() {
                                if !self.entities.contains_key(&next_ent_id.id) {
                                    newentids.push(&next_ent_id.id);
                                }
                            }
                        }
                    }
                    newentids
                })
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            self.entities
                .values()
                .flat_map(|ent| {
                    let mut newentids = Vec::new();
                    for &pid in &transitive_props {
                        if !ent.props.contains_key(pid) {
                            continue;
                        }

                        for stmt in &ent.props[pid] {
                            if let Some(next_ent_id) = stmt.value.as_entity_id() {
                                if !self.entities.contains_key(&next_ent_id.id) {
                                    newentids.push(&next_ent_id.id);
                                }
                            }
                        }
                    }
                    newentids
                })
                .unique()
                .collect::<Vec<_>>()
        };

        let newents = if parallel {
            db.0.entities.par_slice_get_exist(&newentids)?
        } else {
            db.0.entities.slice_get_exist(&newentids)?
        };

        self.entities
            .extend(newents.into_iter().map(|ent| (ent.id.clone(), ent)));
        self.called_prefetched_transitive_entities = true;
        Ok(())
    }

    pub fn has_called_prefetched_property(&self) -> bool {
        self.called_prefetched_property
    }

    pub fn has_called_prefetched_transitive_entities(&self) -> bool {
        self.called_prefetched_transitive_entities
    }
}
