use hashbrown::{HashMap, HashSet};
use kgdata_core::models::Value;

use crate::libs::literal_matchers::{LiteralMatcher, ParsedTextRepr};
use crate::models::basegraph::{EdgeId, GraphIdMap, NodeId};
use crate::models::datagraph as dg;
use crate::models::{Match, MatchMethod};
use crate::{error::GramsError, libs::index::EntityTraversal, models::AlgoContext, models::Table};

use super::data_matching::{DataMatching, Node, PotentialRelationships};
use super::kginference::KGInference;

type StmtIndex<'t> = HashMap<(usize, &'t String, &'t String, usize), NodeId>;
type Str2Node = HashMap<String, NodeId>;

/// Create a data graph.
///
/// Each statement is uniquely identified by (row number, source entiyt id, predicate, and statement index). Because
/// of that, we can have multiple outgoing edges from the same statement node containing the main property to
/// different target nodes. In addition, we can have situation where (A, B) have the same entity, and we found A -> S -> B
/// and B -> S -> A, then we will automatically have A -> S -> A even though data matching doesn't return self-matched relationships.
///
/// # Arguments
///
/// * `table` - The table to perform data matching on
/// * `table_cells` - The parsed text of the table cells
/// * `context` - The context object that contains the data needed for the algorithm to run for each table
/// * `entity_traversal` - A helper to quickly find links between entities
/// * `ignored_columns` - The columns to ignore
/// * `ignored_props` - The properties to ignore and not used during **literal** matching
/// * `allow_same_ent_search` - whether we try to discover relationships between the same entity in same row but different columns
/// * `allow_ent_matching` - whether to match relationships between entities (i.e., only do literal matching)
/// * `use_context` - whether we use context entities in the search
/// * `add_missing_property` - whether to add a new node to represent a statement's property value if the statement has both qualifier and property pointing to the same target node to give the qualifier a chance to be selected.
/// * `deterministic_order` - whether to sort the matched results so that the order is always deterministic
/// * `parallel` - whether to run the data matching algorithm in parallel
#[allow(dead_code)]
pub fn create_data_graph<ET: EntityTraversal>(
    table: &Table,
    table_cells: &Vec<Vec<ParsedTextRepr>>,
    context: &AlgoContext,
    kginference: Option<&KGInference>,
    entity_traversal: &mut ET,
    literal_matcher: &LiteralMatcher,
    ignored_columns: &[usize],
    ignored_props: &HashSet<String>,
    allow_same_ent_search: bool,
    allow_ent_matching: bool,
    use_context: bool,
    add_missing_property: bool,
    deterministic_order: bool,
    parallel: bool,
) -> Result<dg::DGraph, GramsError> {
    let mut g = dg::DGraph::new(table);

    let (nodes, relss) = DataMatching::exec(
        table,
        table_cells,
        context,
        entity_traversal,
        literal_matcher,
        ignored_columns,
        ignored_props,
        allow_same_ent_search,
        allow_ent_matching,
        use_context,
        deterministic_order,
        parallel,
    )?;

    let mut stmtindex: StmtIndex = HashMap::new();
    let mut str2node: Str2Node = HashMap::new();

    step1_from_data_matching(
        &mut g,
        context,
        kginference,
        nodes,
        &relss,
        &mut stmtindex,
        &mut str2node,
    );

    if add_missing_property {
        step2_add_missing_property(&mut g, context, &mut stmtindex, &mut str2node);
    }

    g.set_entity2nodeid(GraphIdMap::new(str2node));

    Ok(g)
}

fn step1_from_data_matching<'t>(
    g: &mut dg::DGraph,
    context: &AlgoContext,
    kginference: Option<&KGInference>,
    nodes: Vec<Node>,
    relss: &'t Vec<PotentialRelationships>,
    stmtindex: &mut StmtIndex<'t>,
    str2node: &mut Str2Node,
) {
    let mut idmap = Vec::with_capacity(nodes.len());

    for node in nodes {
        match node {
            Node::Cell(u) => {
                idmap.push(g.get_cell_node_id(u.row, u.col));
            }
            Node::Entity(u) => {
                let entid = u.entity_id.clone();
                let newnode = g.graph.add_node(dg::Node::Entity(dg::EntityNode {
                    id: NodeId(0),
                    entity_id: u.entity_id,
                    entity_prob: u.entity_prob,
                    is_in_context: true,
                }));
                let newnodeid = newnode.id();
                idmap.push(newnodeid);
                str2node.insert(entid, newnodeid);
            }
        }
    }

    for rels in relss {
        let uid = idmap[rels.source_id];
        let vid = idmap[rels.target_id];

        // get the row that the statement node is from
        let stmtrow = if let Some(dg::Node::Cell(u)) = g.graph.get_node(uid) {
            u.row
        } else if let Some(dg::Node::Cell(v)) = g.graph.get_node(vid) {
            v.row
        } else {
            unreachable!()
        };

        for rel in &rels.rels {
            for stmt in &rel.statements {
                let stmt_property = &stmt.property;
                let sid = if !stmtindex.contains_key(&(
                    stmtrow,
                    &rel.source_entity_id,
                    stmt_property,
                    stmt.statement_index,
                )) {
                    let newnode = g.graph.add_node(dg::Node::Statement(dg::StatementNode {
                        id: NodeId(0),
                        row: stmtrow,
                        entity_id: rel.source_entity_id.clone(),
                        predicate: stmt_property.clone(),
                        statement_index: stmt.statement_index,
                        is_in_kg: kginference
                            .map(|kg| {
                                !kg.is_inferred_statement(
                                    &rel.source_entity_id,
                                    &stmt_property,
                                    stmt.statement_index,
                                )
                            })
                            // by default, if kginference is None, it means we don't run
                            // inference and all statements are in KG
                            .unwrap_or(true),
                    }));
                    let newnodeid = newnode.id();
                    stmtindex.insert(
                        (
                            stmtrow,
                            &rel.source_entity_id,
                            stmt_property,
                            stmt.statement_index,
                        ),
                        newnodeid,
                    );

                    newnodeid
                } else {
                    stmtindex[&(
                        stmtrow,
                        &rel.source_entity_id,
                        stmt_property,
                        stmt.statement_index,
                    )]
                };

                // add incoming edge
                if !g.graph.has_edge_between_nodes(uid, sid, stmt_property) {
                    g.graph.add_edge(dg::Edge {
                        id: EdgeId(0),
                        source: uid,
                        target: sid,
                        predicate: stmt_property.clone(),
                        qualifier_index: None,
                        is_inferred: false,
                        prov: None,
                    });
                }

                if let Some(property_matched_score) = &stmt.matched_property {
                    // we have a matched property
                    if !g.graph.has_edge_between_nodes(sid, vid, stmt_property) {
                        g.graph.add_edge(dg::Edge {
                            id: EdgeId(0),
                            source: sid,
                            target: vid,
                            predicate: stmt_property.clone(),
                            qualifier_index: None,
                            is_inferred: false,
                            prov: Some(dg::EdgeProv(property_matched_score.clone())),
                        });
                    }
                } else {
                    // we do not have a matched property, so we need to add a new node
                    // to represent the property's value -- note that if the statement value
                    // is matched in another relationship, because we share the same statement id
                    // we will automatically have that link (so we have both qualifiers & property
                    // together). By still adding it here, it gives us another chance to say like
                    // the statement's main property may be point to an entity/value in the context
                    // rather than to another column. So this gives us more options.
                    let stmtvalue = &context.entities[&rel.source_entity_id].props[stmt_property]
                        [stmt.statement_index]
                        .value;

                    let stmtvalueprob = stmt
                        .matched_qualifiers
                        .iter()
                        .max_by(|a, b| a.score.cmp(&b.score))
                        .unwrap()
                        .score
                        .clone();
                    let main_vid = add_stmt_value_node(g, stmtvalue, str2node);

                    if let Some(svedge) =
                        g.graph
                            .get_mut_edge_between_nodes(sid, main_vid, stmt_property)
                    {
                        // the edge to the literal/entity already exists, this is due to we have
                        // another qualifier match, so we need to update its score to be the maximum
                        // of those qualifiers
                        let prov = &mut svedge.prov.as_mut().unwrap().0;
                        if prov.prob < stmtvalueprob.prob {
                            prov.prob = stmtvalueprob.prob;
                        }
                    } else {
                        g.graph.add_edge(dg::Edge {
                            id: EdgeId(0),
                            source: sid,
                            target: main_vid,
                            predicate: stmt_property.clone(),
                            qualifier_index: None,
                            is_inferred: false,
                            // we use this method because P(target) is 1.0
                            prov: Some(dg::EdgeProv(Match {
                                prob: stmtvalueprob.prob,
                                method: MatchMethod::LinkMatching,
                            })),
                        });
                    }
                }

                for qual in &stmt.matched_qualifiers {
                    let stmt_qualifier = &qual.qualifier;
                    if !g.graph.has_edge_between_nodes(sid, vid, stmt_qualifier) {
                        g.graph.add_edge(dg::Edge {
                            id: EdgeId(0),
                            source: sid,
                            target: vid,
                            predicate: stmt_qualifier.to_owned(),
                            qualifier_index: Some(qual.qualifier_index),
                            is_inferred: false,
                            prov: Some(dg::EdgeProv(qual.score.clone())),
                        });
                    };
                }
            }
        }
    }
}

/// There is a case where a statement has both qualifier and property pointing to the same node. Unless we have
/// another edge represent the property, the qualifier is unlikely to be chosen. Therefore, we add a new property node
/// with new edge to give the qualifier a chance.
fn step2_add_missing_property(
    g: &mut dg::DGraph,
    context: &AlgoContext,
    stmtindex: &mut StmtIndex,
    str2node: &mut Str2Node,
) {
    for &sid in stmtindex.values() {
        let (qual_edge_ids, stmtprop) = {
            let outedges = g.graph.out_edges(sid);
            if outedges
                .iter()
                .filter(|e| e.qualifier_index.is_none())
                .count()
                > 1
            {
                // has more than one prop
                continue;
            };
            let prop_edge = *outedges
                .iter()
                .filter(|e| e.qualifier_index.is_none())
                .next()
                .unwrap();
            let qual_edges = outedges
                .into_iter()
                .filter(|e| e.qualifier_index.is_some() && e.target == prop_edge.target)
                .map(|e| e.id)
                .collect::<Vec<_>>();
            if qual_edges.len() == 0 {
                continue;
            }

            // create a new node for the statement's value
            let stmtprop = prop_edge.predicate.clone();

            (qual_edges, stmtprop)
        };

        let (stmtvalue, stmtvalueprob) = {
            let snode = g.get_statement_node(sid).unwrap();
            let stmtvalue =
                &context.entities[&snode.entity_id].props[&stmtprop][snode.statement_index].value;
            // probability of the new statement value will be the maximum of all the qualifier's probability
            // across multiple rows
            let stmtvalueprob = qual_edge_ids
                .iter()
                .map(|&eid| g.graph.get_edge(eid).unwrap().prov.as_ref().unwrap().0.prob)
                .reduce(f64::max)
                .unwrap();
            (stmtvalue, stmtvalueprob)
        };
        let main_vid = add_stmt_value_node(g, stmtvalue, str2node);

        // create new main prop edge to the new node
        if !g.graph.has_edge_between_nodes(sid, main_vid, &stmtprop) {
            g.graph.add_edge(dg::Edge {
                id: EdgeId(0),
                source: sid,
                target: main_vid,
                predicate: stmtprop,
                qualifier_index: None,
                is_inferred: false,
                prov: Some(dg::EdgeProv(Match {
                    prob: stmtvalueprob,
                    method: MatchMethod::LinkMatching,
                })),
            });
        };
    }
}

fn add_stmt_value_node(
    g: &mut dg::DGraph,
    value: &Value,
    str2node: &mut HashMap<String, NodeId>,
) -> NodeId {
    if let Some(entid) = value.as_entity_id() {
        if let Some(nodeid) = str2node.get(&entid.id) {
            return *nodeid;
        }

        let newnode = g.graph.add_node(dg::Node::Entity(dg::EntityNode {
            id: NodeId(0),
            entity_id: entid.id.clone(),
            // 1.0 because we track the probability via the edge's provenance
            // use edge's prob here will be prob^2 which is not what we want
            entity_prob: 1.0,
            is_in_context: false,
        }));
        str2node.insert(entid.id.clone(), newnode.id());
        return newnode.id();
    } else {
        let litval = value.to_string_repr();
        if let Some(nodeid) = str2node.get(&litval) {
            return *nodeid;
        }

        let newnode = g.graph.add_node(dg::Node::Literal(dg::LiteralNode {
            id: NodeId(0),
            value: value.clone(),
            // 1.0 because we track the probability via the edge's provenance
            // use edge's prob here will be prob^2 which is not what we want
            value_prob: 1.0,
            is_in_context: false,
        }));
        str2node.insert(litval, newnode.id());
        return newnode.id();
    }
}
