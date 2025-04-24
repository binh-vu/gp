use hashbrown::{HashMap, HashSet};
use kgdata_core::models::{Entity, EntityMetadata};

use crate::models::AlgoContext;

use super::{
    object_hop1_index::{MatchedStatement, ObjectHop1Index},
    EntityTraversal, RefEntityMetadata,
};

pub struct IndexTraversal<'t> {
    pub entities: &'t HashMap<String, Entity>,
    pub entity_metadata: &'t HashMap<String, EntityMetadata>,
    pub index: &'t ObjectHop1Index,
}

impl<'t> IndexTraversal<'t> {
    pub fn from_context(context: &'t AlgoContext) -> Self {
        Self {
            entities: &context.entities,
            entity_metadata: &context.entity_metadata,
            index: context.get_object_1hop_index(),
        }
    }
}

impl<'t> EntityTraversal for IndexTraversal<'t> {
    fn get_outgoing_entity_metadata<'t1>(
        &'t1 self,
        entity_ids: &[&str],
    ) -> Vec<RefEntityMetadata<'t1>> {
        let mut found_ents = HashSet::new();
        let mut next_entities = Vec::new();
        for entid in entity_ids {
            if let Some(next_ents) = self.index.index.get(*entid) {
                for eid in next_ents.keys() {
                    if !found_ents.contains(eid) {
                        let refent = if let Some(ent) = self.entities.get(eid) {
                            RefEntityMetadata {
                                id: &ent.id,
                                label: &ent.label,
                                description: &ent.description,
                                aliases: &ent.aliases,
                            }
                        } else {
                            let ent = self.entity_metadata.get(eid).unwrap();
                            RefEntityMetadata {
                                id: &ent.id,
                                label: &ent.label,
                                description: &ent.description,
                                aliases: &ent.aliases,
                            }
                        };
                        next_entities.push(refent);
                        found_ents.insert(eid);
                    }
                }
            }
        }
        next_entities
    }

    fn iter_props_by_entity<'t1>(
        &'t1 self,
        source: &str,
        target: &str,
    ) -> core::slice::Iter<'t1, MatchedStatement> {
        if let Some(targets) = self.index.index.get(source) {
            if let Some(props) = targets.get(target) {
                return props.iter();
            }
        }
        return [].iter();
    }
}
