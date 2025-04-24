use crate::{error::GramsError, models::AlgoContext};
use hashbrown::{HashMap, HashSet};
use kgdata_core::models::{Entity, Statement, StatementRank, Value};
use rayon::prelude::*;

/// Struct for inferring missing links in the knowledge graph.
///
/// The inferred data is added directly to the algorithm context, and trace of what has been
/// added is kept in this struct.
pub struct KGInference {
    // for tracking the new data generated (entity id => property id => statement index that the new data is started)
    newdata: HashMap<String, HashMap<String, usize>>,
}

type NewData = Vec<(
    String,
    Vec<(String, HashMap<String, (Value, StatementRank)>)>,
)>;

impl KGInference {
    pub fn new() -> Self {
        Self {
            newdata: HashMap::new(),
        }
    }

    /// Tell if a statement is not in KG
    pub fn is_inferred_statement(
        &self,
        entity_id: &str,
        predicate: &str,
        statement_index: usize,
    ) -> bool {
        if self.newdata.contains_key(entity_id) {
            let enttrace = &self.newdata[entity_id];
            if enttrace.contains_key(predicate) {
                let offset = enttrace[predicate];
                return statement_index >= offset;
            }
        }
        return false;
    }

    /// Infer new properties via sub-property of (inheritance)
    pub fn infer_subproperty(&mut self) {
        unimplemented!()
    }

    /// Infer new statements based on transitivity. From A -> B -> C, we can generate A -> C.
    ///
    /// It is unclear how qualifiers are generated using this rule so this function doesn't do that.
    pub fn infer_transitive_property<'t0: 't1, 't1>(
        &mut self,
        context: &'t0 mut AlgoContext,
        parallel: bool,
    ) -> Result<(), GramsError> {
        // generate new statements
        let newdata = if parallel {
            Self::par_find_new_transitive_statements(
                &context.entity_ids_level0,
                &context.entities,
                context.get_transitive_props(),
            )
        } else {
            Self::find_new_transitive_statements(
                &context.entity_ids_level0,
                &context.entities,
                context.get_transitive_props(),
            )
        };

        // then, we add new statements to our data and update the trace of what has been added
        self.add_new_statements(context, newdata);
        Ok(())
    }

    fn par_find_new_transitive_statements<'t1, S: AsRef<str> + Sync + Send + 't1>(
        ent_iter: &'t1 [S],
        entities: &HashMap<String, Entity>,
        transitive_props: HashSet<&String>,
    ) -> NewData {
        ent_iter
            .into_par_iter()
            .filter_map(|entid: &S| {
                let ent = &entities[entid.as_ref()];
                let mut new_ent_data = Vec::new();
                for &pid in transitive_props.iter() {
                    if !ent.props.contains_key(pid) {
                        continue;
                    }

                    // loop for the values of pid and try to gen new statements
                    let mut new_stmts = HashMap::new();
                    for stmt in &ent.props[pid] {
                        if let Some(next_ent_id) = stmt.value.as_entity_id() {
                            let nextent = &entities[&next_ent_id.id];
                            if nextent.props.contains_key(pid) {
                                for s in &nextent.props[pid] {
                                    if let Some(next_next_ent_id) = s.value.as_entity_id() {
                                        if !new_stmts.contains_key(&next_next_ent_id.id) {
                                            new_stmts.insert(
                                                next_next_ent_id.id.clone(),
                                                (s.value.clone(), s.rank.intersect(&stmt.rank)),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for stmt in &ent.props[pid] {
                        if let Some(next_ent_id) = stmt.value.as_entity_id() {
                            new_stmts.remove(&next_ent_id.id);
                        }
                    }

                    new_ent_data.push((pid.clone(), new_stmts));
                }

                if new_ent_data.len() > 0 {
                    return Some((ent.id.clone(), new_ent_data));
                } else {
                    return None;
                }
            })
            .collect::<Vec<_>>()
    }

    fn find_new_transitive_statements<
        't1,
        S: AsRef<str> + Sync + Send + 't1,
        I: IntoIterator<Item = &'t1 S> + 't1,
    >(
        ent_iter: I,
        entities: &HashMap<String, Entity>,
        transitive_props: HashSet<&String>,
    ) -> NewData {
        ent_iter
            .into_iter()
            .filter_map(|entid: &S| {
                let ent = &entities[entid.as_ref()];
                let mut new_ent_data = Vec::new();
                for &pid in transitive_props.iter() {
                    if !ent.props.contains_key(pid) {
                        continue;
                    }

                    // loop for the values of pid and try to gen new statements
                    let mut new_stmts = HashMap::new();
                    for stmt in &ent.props[pid] {
                        if let Some(next_ent_id) = stmt.value.as_entity_id() {
                            let nextent = &entities[&next_ent_id.id];
                            if nextent.props.contains_key(pid) {
                                for s in &nextent.props[pid] {
                                    if let Some(next_next_ent_id) = s.value.as_entity_id() {
                                        if !new_stmts.contains_key(&next_next_ent_id.id) {
                                            new_stmts.insert(
                                                next_next_ent_id.id.clone(),
                                                (s.value.clone(), s.rank.intersect(&stmt.rank)),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for stmt in &ent.props[pid] {
                        if let Some(next_ent_id) = stmt.value.as_entity_id() {
                            new_stmts.remove(&next_ent_id.id);
                        }
                    }

                    new_ent_data.push((pid.clone(), new_stmts));
                }

                if new_ent_data.len() > 0 {
                    return Some((ent.id.clone(), new_ent_data));
                } else {
                    return None;
                }
            })
            .collect::<Vec<_>>()
    }

    /// Add the discovered statements into the context
    fn add_new_statements(&mut self, context: &mut AlgoContext, newdata: NewData) {
        for (ent_id, p2values) in newdata {
            let ent = context.entities.get_mut(&ent_id).unwrap();

            let enttrace = self
                .newdata
                .entry(ent_id)
                .or_insert_with(|| HashMap::with_capacity(p2values.len()));
            for (pid, values) in p2values {
                let lst = ent.props.get_mut(&pid).unwrap();
                if !enttrace.contains_key(&pid) {
                    enttrace.insert(pid, lst.len());
                }
                lst.extend(values.into_values().map(|(val, rank)| Statement {
                    value: val,
                    qualifiers: HashMap::new(),
                    qualifiers_order: Vec::new(),
                    rank: rank,
                }));
            }
        }
    }
}
