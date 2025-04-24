use crate::error::GramsError;
use crate::models::basegraph::EdgeId;
use crate::models::basegraph::NodeId;
use crate::models::cangraph as cgraph;
use crate::models::datagraph as dgraph;
use crate::models::Table;

use ahash::RandomState;
use hashbrown::HashMap;
use hashbrown::HashSet;
use indexmap::IndexMap;

type AHashIndexMap<K, V> = IndexMap<K, V, RandomState>;
type StmtIndex<'t> = HashMap<(NodeId, &'t String, NodeId), NodeId>;

/// Create a candidate graph from a data graph. Assumption about the data graph:
///
/// 1. Between two nodes (cell-cell, cell-entity, or entity-cell), there can be
///    multiple statements between them (e.g., from the same entity but different
///    statement index, or from different candidate entity).
/// 2. For each statement node in the data graph, it can be uniquely identified by
///    (row number (optional if it's entity/literal value), source entity id, predicate, statement index).
///    the last three attributes are to ensure that we can trade to what value is being matched to create that
///    relationship (note that if the relationship is a qualifier, then the qualifier index
///    can be determined from the outgoing edge of the statement node). the row number is to ensure
///    no cross link between rows. for relationship between entities, the row number is None.
///
///    This assumption means that we can have multiple edges from different nodes into the same statement and
///    multiple edges from the statement to different nodes. There is no restriction that the statement must have
///    at maximum one outgoing edge which is the property of the statement.
///
///
/// First, dg edges that constitute statements' properties are grouped first (source, target, main property).
/// Then, for each statement, we add its qualifier. In data graph, because of (assumption 2), a statement
/// can have same property to different targets. But in candidate graph, we create as many statement as
/// the number of properties. The qualifiers is then duplicated for each statement.
///
/// Also, we keep some provenance to trace back the links used to construct the relationships.
/// Note that the data graph can have more than one statement for a given property between
/// the same pairs (source, target) as an entity can have multiple statements for the same property (duplicated aren't
/// merged/resolved because those statements can have different qualifiers). Therefore, when iterating over dg
/// links that are used to construct cg links to calculate the relative frequency, we may encounter multiple statements
/// of the same pairs of the same row and we need to select one statement. Once choice that we use is selecting the one
/// with maximum probability. Note that this is a greedy choice because it doesn't consider other edges of these statements
/// as a whole.
pub fn create_candidate_graph(
    table: &Table,
    dg: &dgraph::DGraph,
) -> Result<cgraph::CGraph, GramsError> {
    let mut cg = cgraph::CGraph::new(table);
    let nrows = table.n_rows();
    let mut dc_map = HashMap::new();
    let mut stmtindex: StmtIndex = HashMap::new();

    // first step is to add missing node, column nodes are already added to the graph during construction
    for node in dg.graph.iter_nodes() {
        match node {
            dgraph::Node::Entity(u) => {
                let newnode = cg.graph.add_node(cgraph::Node::Entity(u.clone()));
                dc_map.insert(u.id, newnode.id());
            }
            dgraph::Node::Literal(u) => {
                let newnode = cg.graph.add_node(cgraph::Node::Literal(u.clone()));
                dc_map.insert(u.id, newnode.id());
            }
            _ => {}
        }
    }

    // second step: add link
    for u in dg.graph.iter_nodes() {
        if u.is_statement() {
            continue;
        }
        let uid = u.id();

        // grouping by predicates, use ahash to make it deterministic
        let mut p2stmts: AHashIndexMap<&String, Vec<&dgraph::StatementNode>> =
            AHashIndexMap::default();
        for usedge in dg.graph.iter_out_edges(uid) {
            if !p2stmts.contains_key(&usedge.predicate) {
                p2stmts.insert(
                    &usedge.predicate,
                    vec![dg.get_statement_node(usedge.target).unwrap()],
                );
            } else {
                p2stmts
                    .get_mut(&usedge.predicate)
                    .unwrap()
                    .push(dg.get_statement_node(usedge.target).unwrap());
            }
        }

        let cguid = get_cg_node_id(&cg, u, &dc_map);
        for (&p, stmts) in p2stmts.iter() {
            for stmt in stmts {
                // get list of cg target nodes of the statement property
                let mut cgvids = Vec::new();
                // get list of qualifiers
                let mut qualifiers = Vec::new();

                for svedge in dg.graph.iter_out_edges(stmt.id) {
                    if svedge.predicate == *p {
                        cgvids.push((
                            svedge,
                            get_cg_node_id(&cg, dg.graph.get_node(svedge.target).unwrap(), &dc_map),
                        ));
                    } else {
                        qualifiers.push((svedge, dg.graph.get_node(svedge.target).unwrap()));
                    }
                }

                // due to how dg is constructed (each row is a sub-graph), cgvids must be a set
                assert_eq!(
                    cgvids.len(),
                    cgvids.iter().map(|x| x.1).collect::<HashSet<_>>().len()
                );

                // then for each target node, we add link to the statement's property first, and then add the qualifiers
                for (dge, cgvid) in cgvids {
                    // add statement if not exist
                    let cgsid = if !stmtindex.contains_key(&(cguid, &stmt.predicate, cgvid)) {
                        let newnode =
                            cg.graph
                                .add_node(cgraph::Node::Statement(cgraph::StatementNode {
                                    id: NodeId(0),
                                    dgprov: cgraph::DGStatementProv::new(nrows),
                                }));
                        let newnodeid = newnode.id();
                        stmtindex.insert((cguid, &stmt.predicate, cgvid), newnodeid);
                        newnodeid
                    } else {
                        stmtindex[&(cguid, &stmt.predicate, cgvid)]
                    };

                    // track the dg statement we used
                    cg.graph
                        .get_mut_node(cgsid)
                        .unwrap()
                        .try_as_mut_statement()
                        .unwrap()
                        .dgprov
                        .track(*stmt);

                    // add link encodes statement's property
                    if !cg.graph.has_edge_between_nodes(cguid, cgsid, p) {
                        cg.graph.add_edge(cgraph::Edge {
                            id: EdgeId(0),
                            source: cguid,
                            target: cgsid,
                            predicate: p.clone(),
                            is_qualifier: false,
                            dgprov: None,
                        });
                    }

                    if let Some(cgedge) = cg.graph.get_mut_edge_between_nodes(cgsid, cgvid, p) {
                        cgedge.dgprov.as_mut().unwrap().track(stmt.row, dge);
                    } else {
                        let mut dgprov = cgraph::DGEdgeProv::new(nrows);
                        dgprov.track(stmt.row, dge);
                        cg.graph.add_edge(cgraph::Edge {
                            id: EdgeId(0),
                            source: cgsid,
                            target: cgvid,
                            predicate: p.clone(),
                            is_qualifier: false,
                            dgprov: Some(dgprov),
                        });
                    }

                    // add the qualifiers
                    for (dgqe, dgqv) in &qualifiers {
                        let cgqvid = get_cg_node_id(&cg, dgqv, &dc_map);
                        if let Some(cgedge) =
                            cg.graph
                                .get_mut_edge_between_nodes(cgsid, cgqvid, &dgqe.predicate)
                        {
                            cgedge.dgprov.as_mut().unwrap().track(stmt.row, dgqe);
                        } else {
                            let mut dgprov = cgraph::DGEdgeProv::new(nrows);
                            dgprov.track(stmt.row, dgqe);
                            cg.graph.add_edge(cgraph::Edge {
                                id: EdgeId(0),
                                source: cgsid,
                                target: cgqvid,
                                predicate: dgqe.predicate.clone(),
                                is_qualifier: true,
                                dgprov: Some(dgprov),
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(cg)
}

#[inline]
fn get_cg_node_id(
    cg: &cgraph::CGraph,
    u: &dgraph::Node,
    dc_map: &HashMap<NodeId, NodeId>,
) -> NodeId {
    match u {
        dgraph::Node::Cell(u) => cg.get_column_node_id(u.column),
        dgraph::Node::Entity(u) => dc_map[&u.id],
        dgraph::Node::Literal(u) => dc_map[&u.id],
        _ => {
            unreachable!()
        }
    }
}
