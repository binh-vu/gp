// mod cache_traversal;
mod index_traversal;
mod object_hop1_index;
// mod vanila_traversal;

use kgdata_core::models::{MultiLingualString, MultiLingualStringList};

pub use self::index_traversal::IndexTraversal;
pub use self::object_hop1_index::{MatchedStatement, ObjectHop1Index};

pub struct RefEntityMetadata<'t> {
    pub id: &'t String,
    pub label: &'t MultiLingualString,
    pub description: &'t MultiLingualString,
    pub aliases: &'t MultiLingualStringList,
}

pub trait EntityTraversal: Sync + Send {
    fn get_outgoing_entity_metadata<'t1>(
        &'t1 self,
        entity_ids: &[&str],
    ) -> Vec<RefEntityMetadata<'t1>>;

    fn iter_props_by_entity<'t1>(
        &'t1 self,
        source_id: &str,
        target_id: &str,
    ) -> core::slice::Iter<'t1, MatchedStatement>;
}
