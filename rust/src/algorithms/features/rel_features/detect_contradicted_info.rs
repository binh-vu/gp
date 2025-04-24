use super::super::feature_store::{does_ent_have_data, FeatureStore, FeatureStoreCache};
use crate::models::basegraph::NodeId;
use crate::{
    error::GramsError,
    models::{cangraph as cgraph, datagraph as dgraph},
};
use hashbrown::HashSet;
use kgdata_core::models::{Entity, Value};
use log::warn;
use pyo3::prelude::*;

#[pyclass(module = "gp_core.features")]
/// Represents a pair of nodes that the relationship between them is contradicted by the data
pub struct ContradictedInformation {
    #[pyo3(get)]
    pub source: NodeId,
    #[pyo3(get)]
    pub target: NodeId,
    #[pyo3(get)]
    pub inedge: String,
    #[pyo3(get)]
    pub outedge: String,
}

/// Get the **number** of DG pairs that may contain contradicted information with the relationship inedge -> s -> outedge
///
/// A pair that contain contradicted information when both:
/// (1) It does not found in data graph
/// (2) The n-ary relationships (property and (optionally) qualifier) exist in the entity.
///
/// Because of (1), the (2) says the value in KG is different from the value in the table. However, we need to be
/// careful because of missing values. To combat this, we need to distinguish when we actually have missing values.
///
/// Also, we need to be careful to not use entities that the threshold is way too small (e.g., the cell is Bob but the candidate
/// is John).
///
/// For detecting missing values, we can use some information such as if the relationship has exactly one value, or
/// we can try to detect if we can find the information in some pages.
pub fn get_contradicted_information<D: MissingInfoDetector>(
    store: &mut FeatureStore,
    storecache: &mut FeatureStoreCache,
    missing_info_detector: &D,
    inedge: &cgraph::Edge,
    outedge: &cgraph::Edge,
    correct_entity_threshold: f64,
) -> Result<Vec<ContradictedInformation>, GramsError> {
    let cgu = store.cg.graph.get_node(inedge.source).unwrap();
    let cgv = store.cg.graph.get_node(outedge.target).unwrap();

    let uv_links = store
        .cg
        .get_dg_node_pairs(&store.dg, outedge)
        .into_iter()
        .collect::<HashSet<_>>();
    // let uv_links = {
    //     context.cache.get_dg_pairs(&context.dg, s, outedge);
    //     context.cache.get_dg_pairs.get(&(s.id, outedge.id)).unwrap()
    // };

    let is_outpred_qualifier = outedge.is_qualifier;
    let is_outpred_data_predicate = store.context.props[&outedge.predicate].is_data_property();
    // let is_outpred_qualifier = inedge.predicate != outedge.predicate;
    // let is_outpred_data_predicate = context
    //     .context
    //     .get_or_fetch_property(&outedge.predicate, context.db)?
    //     .is_data_property();

    let entities = &store.context.entities;
    // let entities = &context.context.entities;

    let mut contradicted_info = Vec::new();
    // let mut n_contradicted_info = 0;

    for (dgu_id, dgv_id) in storecache.iter_dg_pair(store, cgu, cgv) {
        if uv_links.contains(&(dgu_id, dgv_id)) {
            continue;
        }

        let no_data = match store.dg.graph.get_node(dgu_id).unwrap() {
            dgraph::Node::Cell(cell) => {
                store
                    .iter_cell_candidate_entities(cell.row, cell.column)
                    .all(|canent| {
                        canent.probability < correct_entity_threshold
                            || !does_ent_have_data(
                                &entities[&canent.id.0],
                                &inedge.predicate,
                                &outedge.predicate,
                                is_outpred_qualifier,
                            )
                    })
                // cell.entity_ids.iter().all(|eid| {
                //     cell.entity_probs[eid] < correct_entity_threshold
                //         || !does_ent_have_data(
                //             &entities[eid],
                //             &inedge.predicate,
                //             &outedge.predicate,
                //             is_outpred_qualifier,
                //         )
                // })
            }
            dgraph::Node::Entity(entity) => {
                entity.entity_prob < correct_entity_threshold
                    || !does_ent_have_data(
                        &entities[&entity.entity_id],
                        &inedge.predicate,
                        &outedge.predicate,
                        is_outpred_qualifier,
                    )
            }
            _ => {
                return Err(GramsError::IntegrityError(
                    "Source node must be cell or entity".to_owned(),
                ))
            }
        };
        if no_data {
            continue;
        }

        // at this moment, it should have some info. different than the one in the table
        // but it can be due to missing values, so we check it here.
        if is_outpred_data_predicate {
            let dgv_value = match store.dg.graph.get_node(dgv_id).unwrap() {
                dgraph::Node::Cell(cell) => Some(store.table.get_cell(cell.row, cell.column)),
                dgraph::Node::Literal(literal) => match &literal.value {
                    Value::String(v) => Some(v.as_str()),
                    Value::Quantity(v) => Some(v.amount.as_str()),
                    Value::MonolingualText(v) => Some(v.text.as_str()),
                    // Value::Time(v) => &v.time,
                    // Value::GlobeCoordinate(v) => &v.globe,
                    // Value::EntityId(v) => &v.id,
                    _ => {
                        // TODO: the other types weren't handled properly
                        None
                    }
                },
                dgraph::Node::Entity(_entity) => {
                    // we do have this case that data predicate such as P2561
                    // that values link to entity value, we do not handle it for now so
                    // we set the value to None so it is skipped
                    None
                }
                _ => {
                    return Err(GramsError::IntegrityError(
                        "Target node must be cell, entity or literal".to_owned(),
                    ))
                }
            };

            let has_missing_info = if let Some(dgv_value) = dgv_value {
                match store.dg.graph.get_node(dgu_id).unwrap() {
                    dgraph::Node::Cell(cell) => store
                        .iter_cell_candidate_entities(cell.row, cell.column)
                        .any(|canent| {
                            if canent.probability < correct_entity_threshold {
                                return false;
                            }
                            let ent = &entities[&canent.id.0];
                            does_ent_have_data(
                                ent,
                                &inedge.predicate,
                                &outedge.predicate,
                                is_outpred_qualifier,
                            ) && missing_info_detector.is_missing_data_info(
                                store,
                                storecache,
                                ent,
                                &inedge.predicate,
                                &outedge.predicate,
                                dgv_value,
                            )
                        }),
                    dgraph::Node::Entity(e) => {
                        let ent = &entities[&e.entity_id];
                        // we do not need to check if this ent prob is above the threshold
                        // and if it has the data because we already checked it above (a single entity so if we do not have it, this code is not reachable)
                        missing_info_detector.is_missing_data_info(
                            store,
                            storecache,
                            ent,
                            &inedge.predicate,
                            &outedge.predicate,
                            dgv_value,
                            // context,
                        )
                    }
                    _ => false,
                }
            } else {
                false
            };

            if !has_missing_info {
                // n_contradicted_info += 1;
                contradicted_info.push(ContradictedInformation {
                    source: dgu_id,
                    target: dgv_id,
                    inedge: inedge.predicate.clone(),
                    outedge: outedge.predicate.clone(),
                });
            }
        } else {
            // object property, check an external db
            // but we need to filter out the case where we do not have the data
            let no_data = match store.dg.graph.get_node(dgv_id).unwrap() {
                dgraph::Node::Cell(dgv_cell) => store
                    .iter_cell_candidate_entities(dgv_cell.row, dgv_cell.column)
                    .all(|canent| canent.probability < correct_entity_threshold),
                dgraph::Node::Entity(dgv_entity) => {
                    dgv_entity.entity_prob < correct_entity_threshold
                }
                dgraph::Node::Literal(dgv_lit) => {
                    warn!(
                        "Found a literal value {:?} for an object property {} -> {} in one of the entities of node {:?}",
                        dgv_lit,
                        &inedge.predicate,
                        &outedge.predicate,
                        &dgu_id,
                    );
                    true
                }
                _ => {
                    return Err(GramsError::IntegrityError(
                        "Target node must be cell, entity or literal".to_owned(),
                    ))
                }
            };
            if no_data {
                continue;
            }

            let has_missing_info = match store.dg.graph.get_node(dgu_id).unwrap() {
                dgraph::Node::Cell(dgu_cell) => store
                    .iter_cell_candidate_entities(dgu_cell.row, dgu_cell.column)
                    .any(|canent| {
                        if canent.probability < correct_entity_threshold {
                            return false;
                        }
                        let ent = &entities[&canent.id.0];
                        if !does_ent_have_data(
                            ent,
                            &inedge.predicate,
                            &outedge.predicate,
                            is_outpred_qualifier,
                        ) {
                            return false;
                        }

                        match store.dg.graph.get_node(dgv_id).unwrap() {
                            dgraph::Node::Cell(dgv_cell) => store
                                .iter_cell_candidate_entities(dgv_cell.row, dgv_cell.column)
                                .any(|v_eid| {
                                    v_eid.probability >= correct_entity_threshold
                                        && missing_info_detector.is_missing_object_info(
                                            store,
                                            storecache,
                                            ent,
                                            &inedge.predicate,
                                            &outedge.predicate,
                                            &v_eid.id.0,
                                        )
                                }),
                            dgraph::Node::Entity(dgv_ent) => {
                                dgv_ent.entity_prob >= correct_entity_threshold
                                    && missing_info_detector.is_missing_object_info(
                                        store,
                                        storecache,
                                        ent,
                                        &inedge.predicate,
                                        &outedge.predicate,
                                        &dgv_ent.entity_id,
                                    )
                            }
                            _ => unreachable!(),
                        }
                    }),
                dgraph::Node::Entity(dgu_ent) => {
                    let ent = &entities[&dgu_ent.entity_id];

                    // we do not need to check if this ent prob is above the threshold
                    // and if it has the data because we already checked it above (a single entity so if we do not have it, this code is not reachable)
                    match store.dg.graph.get_node(dgv_id).unwrap() {
                        dgraph::Node::Cell(dgv_cell) => store
                            .iter_cell_candidate_entities(dgv_cell.row, dgv_cell.column)
                            .any(|v_eid| {
                                v_eid.probability >= correct_entity_threshold
                                    && missing_info_detector.is_missing_object_info(
                                        store,
                                        storecache,
                                        ent,
                                        &inedge.predicate,
                                        &outedge.predicate,
                                        &v_eid.id.0,
                                    )
                            }),
                        dgraph::Node::Entity(dgv_ent) => {
                            dgv_ent.entity_prob >= correct_entity_threshold
                                && missing_info_detector.is_missing_object_info(
                                    store,
                                    storecache,
                                    ent,
                                    &inedge.predicate,
                                    &outedge.predicate,
                                    &dgv_ent.entity_id,
                                )
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            };

            if !has_missing_info {
                // n_contradicted_info += 1;
                contradicted_info.push(ContradictedInformation {
                    source: dgu_id,
                    target: dgv_id,
                    inedge: inedge.predicate.clone(),
                    outedge: outedge.predicate.clone(),
                });
            }
        }
    }
    Ok(contradicted_info)
}

pub trait MissingInfoDetector {
    fn is_missing_data_info(
        &self,
        store: &FeatureStore,
        storecache: &mut FeatureStoreCache,
        ent: &Entity,
        property: &str,
        qualifier: &str,
        value: &str,
    ) -> bool;

    fn is_missing_object_info(
        &self,
        store: &FeatureStore,
        storecache: &mut FeatureStoreCache,
        ent: &Entity,
        property: &str,
        qualifier: &str,
        target_ent_id: &str,
    ) -> bool;
}

pub struct SimpleMissingInfoDetector;

impl MissingInfoDetector for SimpleMissingInfoDetector {
    #[allow(unused_variables)]
    fn is_missing_data_info(
        &self,
        store: &FeatureStore,
        storecache: &mut FeatureStoreCache,
        ent: &Entity,
        property: &str,
        qualifier: &str,
        value: &str,
    ) -> bool {
        // implement via infobox search
        false
    }

    #[allow(unused_variables)]
    fn is_missing_object_info(
        &self,
        store: &FeatureStore,
        storecache: &mut FeatureStoreCache,
        ent: &Entity,
        property: &str,
        qualifier: &str,
        target_ent_id: &str,
    ) -> bool {
        if let Some(link) = storecache.entity_outlinks.get(&ent.id).unwrap() {
            link.targets.contains(target_ent_id)
        } else {
            false
        }
    }
}
