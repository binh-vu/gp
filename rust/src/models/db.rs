use std::path::PathBuf;

use super::{table::Table, AlgoContext};
use crate::error::GramsError;
use hashbrown::{HashMap, HashSet};
use kgdata_core::{
    db::{BaseKGDB, Map, ReadonlyRocksDBDict, RemoteKGDB, RemoteRocksDBDict, KGDB},
    error::KGDataError,
    models::{Entity, EntityMetadata, Value},
};
use rayon::prelude::*;

pub struct BaseGramsDB<ED, EMD>(pub BaseKGDB<ED, EMD>)
where
    ED: Map<String, Entity>,
    EMD: Map<String, EntityMetadata>;

pub type LocalGramsDB =
    BaseGramsDB<ReadonlyRocksDBDict<String, Entity>, ReadonlyRocksDBDict<String, EntityMetadata>>;
pub type RemoteGramsDB =
    BaseGramsDB<RemoteRocksDBDict<String, Entity>, RemoteRocksDBDict<String, EntityMetadata>>;

impl<ED, EMD> std::fmt::Debug for BaseGramsDB<ED, EMD>
where
    ED: Map<String, Entity> + Send + Sync,
    EMD: Map<String, EntityMetadata> + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GramsDB")
            .field("datadir", &self.0.datadir)
            .finish()
    }
}

impl<ED, EMD> BaseGramsDB<ED, EMD>
where
    ED: Map<String, Entity> + Send + Sync,
    EMD: Map<String, EntityMetadata> + Send + Sync,
{
    #[inline]
    pub fn get_data_dir(&self) -> &PathBuf {
        &self.0.datadir
    }

    pub fn get_algo_context(
        &self,
        table: &Table,
        n_hop: usize,
        parallel: bool,
    ) -> Result<AlgoContext, GramsError> {
        let entity_ids = self.get_table_entity_ids(table);
        let (entities, entity_metadata) = self.get_entities(&entity_ids, n_hop, parallel)?;

        Ok(AlgoContext::new(
            entity_ids,
            entities,
            entity_metadata,
            self.0.kgns.clone(),
        ))
    }

    pub fn get_table_entity_ids(&self, table: &Table) -> Vec<String> {
        let mut entity_ids = HashSet::new();
        for row in &table.links {
            for links in row {
                for link in links {
                    for candidate in &link.candidates {
                        entity_ids.insert(&candidate.id.0);
                    }
                    for entityid in &link.entities {
                        entity_ids.insert(&entityid.0);
                    }
                }
            }
        }
        for entityid in &table.context.page_entities {
            entity_ids.insert(&entityid.id.0);
        }

        entity_ids
            .into_iter()
            .map(|x| x.to_owned())
            .collect::<Vec<_>>()
    }

    pub fn get_entities(
        &self,
        entity_ids: &[String],
        n_hop: usize,
        parallel: bool,
    ) -> Result<(HashMap<String, Entity>, HashMap<String, EntityMetadata>), GramsError> {
        let mut entities = if parallel {
            self.0.entities.par_slice_get_exist_as_map(entity_ids)?
        } else {
            self.0.entities.slice_get_exist_as_map(entity_ids)?
        };

        if n_hop == 1 {
            let neighbor_ent_ids =
                self.retrieve_unfetch_neighbor_entity_ids(entity_ids, &entities, parallel);

            let entity_metadata = if parallel {
                self.0
                    .entity_metadata
                    .par_slice_get_exist_as_map(&neighbor_ent_ids)?
            } else {
                self.0
                    .entity_metadata
                    .slice_get_exist_as_map(&neighbor_ent_ids)?
            };
            return Ok((entities, entity_metadata));
        }

        let mut current_hop_entities = Vec::new();
        for i in 2..n_hop {
            let next_hop_entities = {
                let next_hop_entity_ids: Vec<&str> = if i == 2 {
                    self.retrieve_unfetch_neighbor_entity_ids(entity_ids, &entities, parallel)
                } else {
                    self.retrieve_unfetch_neighbor_entity_ids_from_ents(
                        &current_hop_entities,
                        &entities,
                        parallel,
                    )
                };

                self.0.entities.slice_get_exist(&next_hop_entity_ids)?
            };
            entities.extend(current_hop_entities.into_iter().map(|e| (e.id.clone(), e)));
            current_hop_entities = next_hop_entities;
        }

        if n_hop == 2 {
            let second_hop_entity_ids =
                self.retrieve_unfetch_neighbor_entity_ids(entity_ids, &entities, parallel);
            current_hop_entities = self.0.entities.slice_get_exist(&second_hop_entity_ids)?;
        }

        let neighbor_ent_ids = self.retrieve_unfetch_neighbor_entity_ids_from_ents(
            &current_hop_entities,
            &entities,
            parallel,
        );
        let entity_metadata = if parallel {
            self.0
                .entity_metadata
                .par_slice_get_exist_as_map(&neighbor_ent_ids)?
        } else {
            self.0
                .entity_metadata
                .slice_get_exist_as_map(&neighbor_ent_ids)?
        };

        entities.extend(current_hop_entities.into_iter().map(|e| (e.id.clone(), e)));
        Ok((entities, entity_metadata))
    }

    pub fn retrieve_unfetch_neighbor_entity_ids<
        't,
        S: std::ops::Deref<Target = str> + Sync + Send,
    >(
        &self,
        entity_ids: &[S],
        fetched_entities: &'t HashMap<String, Entity>,
        parallel: bool,
    ) -> Vec<&'t str> {
        fn init<'t>() -> HashSet<&'t str> {
            HashSet::new()
        }

        fn merge<'t>(mut c: HashSet<&'t str>, c1: HashSet<&'t str>) -> HashSet<&'t str> {
            c.extend(c1);
            c
        }

        #[inline]
        fn fold<'t>(
            entid: &str,
            collection: &mut HashSet<&'t str>,
            fetched_entities: &'t HashMap<String, Entity>,
        ) {
            for stmts in fetched_entities[entid].props.values() {
                for stmt in stmts {
                    if let Value::EntityId(eid) = &stmt.value {
                        if !fetched_entities.contains_key(&eid.id) {
                            collection.insert(&eid.id);
                        }
                    }

                    for qvals in stmt.qualifiers.values() {
                        for qval in qvals {
                            if let Value::EntityId(eid) = &qval {
                                if !fetched_entities.contains_key(&eid.id) {
                                    collection.insert(&eid.id);
                                }
                            }
                        }
                    }
                }
            }
        }

        let neighbor_entity_ids = if parallel {
            entity_ids
                .into_par_iter()
                .fold(init, |mut c, entid| {
                    fold(entid, &mut c, fetched_entities);
                    c
                })
                .reduce(init, merge)
        } else {
            let mut neighbor_entity_ids = init();
            for entid in entity_ids {
                fold(entid, &mut neighbor_entity_ids, fetched_entities);
            }
            neighbor_entity_ids
        };

        neighbor_entity_ids.into_iter().collect::<Vec<_>>()
    }

    pub fn retrieve_unfetch_neighbor_entity_ids_from_ents<'t>(
        &self,
        entities: &'t [Entity],
        fetched_entities: &'t HashMap<String, Entity>,
        parallel: bool,
    ) -> Vec<&'t str> {
        fn init<'t>() -> HashSet<&'t str> {
            HashSet::new()
        }

        fn merge<'t>(mut c: HashSet<&'t str>, c1: HashSet<&'t str>) -> HashSet<&'t str> {
            c.extend(c1);
            c
        }

        #[inline]
        fn fold<'t>(
            ent: &'t Entity,
            collection: &mut HashSet<&'t str>,
            fetched_entities: &'t HashMap<String, Entity>,
        ) {
            for stmts in ent.props.values() {
                for stmt in stmts {
                    if let Value::EntityId(eid) = &stmt.value {
                        if !fetched_entities.contains_key(&eid.id) {
                            collection.insert(&eid.id);
                        }
                    }

                    for qvals in stmt.qualifiers.values() {
                        for qval in qvals {
                            if let Value::EntityId(eid) = &qval {
                                if !fetched_entities.contains_key(&eid.id) {
                                    collection.insert(&eid.id);
                                }
                            }
                        }
                    }
                }
            }
        }

        let neighbor_entity_ids = if parallel {
            entities
                .into_par_iter()
                .fold(init, |mut c, ent| {
                    fold(ent, &mut c, fetched_entities);
                    c
                })
                .reduce(init, merge)
        } else {
            let mut neighbor_entity_ids = init();
            for ent in entities {
                fold(ent, &mut neighbor_entity_ids, fetched_entities);
            }
            neighbor_entity_ids
        };

        neighbor_entity_ids.into_iter().collect::<Vec<_>>()
    }
}

impl LocalGramsDB {
    pub fn new(datadir: &str) -> Result<Self, GramsError> {
        Ok(Self(KGDB::new(datadir)?))
    }
}

impl RemoteGramsDB {
    pub fn new<Q: AsRef<str>>(
        datadir: &str,
        entity_urls: &[Q],
        entity_metadata_urls: &[Q],
        entity_batch_size: usize,
        entity_metadata_batch_size: usize,
    ) -> Result<Self, GramsError> {
        Ok(Self(RemoteKGDB::new(
            datadir,
            entity_urls,
            entity_metadata_urls,
            entity_batch_size,
            entity_metadata_batch_size,
        )?))
    }
}

pub struct CacheRocksDBDict<V>
where
    V: Sync + Send,
{
    db: ReadonlyRocksDBDict<String, V>,
    cache: HashMap<String, Option<V>>,
}

impl<V> CacheRocksDBDict<V>
where
    V: Sync + Send,
{
    pub fn new(db: ReadonlyRocksDBDict<String, V>) -> Self {
        Self {
            db,
            cache: HashMap::new(),
        }
    }

    pub fn get(&mut self, key: &str) -> Result<Option<&V>, KGDataError> {
        if !self.cache.contains_key(key) {
            self.cache.insert(key.to_owned(), self.db.get(key)?);
        }

        Ok(self.cache[key].as_ref())
    }
}

pub trait GetId {
    fn get_id(&self) -> &str;
}

impl GetId for &str {
    #[inline]
    fn get_id(&self) -> &str {
        self
    }
}

impl GetId for Entity {
    #[inline]
    fn get_id(&self) -> &str {
        &self.id
    }
}
