from __future__ import annotations

import itertools
from copy import copy
from enum import IntEnum
from typing import Literal, Mapping, Optional, Sequence

import numpy as np
from graph.interface import NodeID
from kgdata.models import Ontology
from kgdata.models.ont_property import OntologyProperty
from rdflib import RDFS
from sm.dataset import FullTable
from sm.evaluation import sm_metrics
from sm.namespaces.namespace import KnowledgeGraphNamespace
from sm.outputs.semantic_model import (
    ClassNode,
    DataNode,
    Edge,
    LiteralNode,
    LiteralNodeDataType,
    SemanticModel,
)


class SMNodeType(IntEnum):
    Column = 0
    Class = 1
    Statement = 2
    Entity = 3
    Literal = 4


class SemanticModelHelper:
    DUMMY_CLASS_FOR_INVERSION_URI = "http://wikiba.se/ontology#DummyClassForInversion"
    ID_PROPS = {str(RDFS.label)}

    def __init__(
        self,
        entity_labels: Optional[Mapping[str, str]],
        props: Mapping[str, OntologyProperty],
        kgns: KnowledgeGraphNamespace,
    ):
        self.entity_labels = entity_labels or {}
        self.props = props
        self.kgns = kgns

    @staticmethod
    def from_ontology(ontology: Ontology) -> SemanticModelHelper:
        return SemanticModelHelper(None, ontology.props, ontology.kgns)

    def norm_sm(self, sm: SemanticModel):
        """ "Normalize the semantic model with the following modifications:
        1. Add readable label to edge and class
        2. Convert direct link (without statement) to have statement except the id props.
        """
        new_sm = sm.deep_copy()
        kgns = self.kgns

        # update readable label
        for n in new_sm.iter_nodes():
            if isinstance(n, ClassNode):
                if kgns.has_encrypted_name(n.abs_uri):
                    n.readable_label = self.get_ent_label(kgns.uri_to_id(n.abs_uri))
            elif (
                isinstance(n, LiteralNode) and n.datatype == LiteralNodeDataType.Entity
            ):
                if kgns.has_encrypted_name(n.value):
                    n.readable_label = self.get_ent_label(kgns.uri_to_id(n.value))
        for e in new_sm.iter_edges():
            if e.abs_uri not in self.ID_PROPS and kgns.has_encrypted_name(e.abs_uri):
                e.readable_label = self.get_pnode_label(kgns.uri_to_id(e.abs_uri))

        # convert direct link
        for edge in list(new_sm.iter_edges()):
            if edge.abs_uri in self.ID_PROPS:
                continue
            source = new_sm.get_node(edge.source)
            target = new_sm.get_node(edge.target)

            if (
                not isinstance(source, ClassNode)
                or source.abs_uri != self.kgns.statement_uri
            ) and (
                not isinstance(target, ClassNode)
                or target.abs_uri != self.kgns.statement_uri
            ):
                # this is direct link, we replace its edge
                # assert len(new_sm.get_edges_between_nodes(source.id, target.id)) == 1
                # new_sm.remove_edges_between_nodes(source.id, target.id)

                for edge in new_sm.get_edges_between_nodes(source.id, target.id):
                    new_sm.remove_edge(edge.id)
                    stmt = ClassNode(
                        abs_uri=self.kgns.statement_uri,
                        rel_uri=kgns.get_rel_uri(self.kgns.statement_uri),
                    )
                    new_sm.add_node(stmt)
                    new_sm.add_edge(
                        Edge(
                            source=edge.source,
                            target=stmt.id,
                            abs_uri=edge.abs_uri,
                            rel_uri=edge.rel_uri,
                            approximation=edge.approximation,
                            readable_label=edge.readable_label,
                        )
                    )
                    new_sm.add_edge(
                        Edge(
                            source=stmt.id,
                            target=edge.target,
                            abs_uri=edge.abs_uri,
                            rel_uri=edge.rel_uri,
                            approximation=edge.approximation,
                            readable_label=edge.readable_label,
                        )
                    )
        return new_sm

    def minify_sm(self, sm: SemanticModel):
        """This is a reverse function of `norm_sm`:
        1. Remove an intermediate statement if it doesn't have any qualifiers
        """
        new_sm = sm.copy()

        for n in sm.iter_nodes():
            if isinstance(n, ClassNode) and n.abs_uri == self.kgns.statement_uri:
                inedges = sm.in_edges(n.id)
                outedges = sm.out_edges(n.id)
                if len(outedges) == 1 and outedges[0].abs_uri == inedges[0].abs_uri:
                    # no qualifiers
                    new_sm.remove_node(n.id)
                    for inedge in inedges:
                        assert inedge.abs_uri == outedges[0].abs_uri
                        new_sm.add_edge(
                            Edge(
                                inedge.source,
                                outedges[0].target,
                                inedge.abs_uri,
                                inedge.rel_uri,
                                # just in case user misannotate to not include approximation in both links
                                inedge.approximation or outedges[0].approximation,
                                inedge.readable_label,
                            )
                        )
        return new_sm

    def gen_equivalent_sm(
        self,
        sm: SemanticModel,
        strict: bool = True,
        force_inversion: bool = False,
        limited_invertible_props: Optional[set[str]] = None,
        incorrect_invertible_props: Optional[set[str]] = None,
    ):
        """Given a semantic model (not being modified), generate equivalent **normalized** models by inferring inverse properties.

        Currently, we only inverse the properties, not qualifiers.



        Parameters
        ----------
        sm: the input semantic model (original)
        strict: whether to throw exception when target of an inverse property is not a class.
        force_inversion: only work when strict mode is set to false. Without force_inverse, we skip inverse properties,
                       otherwise, we generate an inverse model with a special class: wikibase:DummyClassForInversion
        limited_invertible_props: if provided, only generate inverse properties for these properties.
        incorrect_invertible_props: if provided, skip generating inverse properties for these properties.
        Returns
        -------
        """
        sm = self.norm_sm(sm)
        kgns = self.kgns

        if incorrect_invertible_props is None:
            incorrect_invertible_props = set()

        invertible_stmts: list[ClassNode] = []
        is_class_fn = lambda n1: isinstance(n1, ClassNode) or (
            isinstance(n1, LiteralNode) and n1.datatype == LiteralNodeDataType.Entity
        )

        for n in sm.iter_nodes():
            if isinstance(n, ClassNode) and n.abs_uri == self.kgns.statement_uri:
                inedges = sm.in_edges(n.id)
                outedges = sm.out_edges(n.id)
                # only has one prop
                (prop,) = list({inedge.abs_uri for inedge in inedges})
                pid = kgns.uri_to_id(prop)
                stmt_has_value = False
                for outedge in outedges:
                    if outedge.abs_uri != prop:
                        # assert len(self.wdprops[self.get_prop_id(outedge.abs_uri)].inverse_properties) == 0, "Just to make sure" \
                        #                                                                                    "that qualifiers is not invertable. Otherwise, this algorithm will missing one generated SMs"
                        # character role has an inverse property: performer. They can be used as qualifier so nothing to do here just pass
                        pass
                    else:
                        stmt_has_value = True
                if (
                    len(self.props[pid].inverse_properties) > 0
                    and pid not in incorrect_invertible_props
                    and (
                        limited_invertible_props is None
                        or pid in limited_invertible_props
                    )
                    and stmt_has_value
                ):
                    # invertible property
                    # people seem to misunderstand what inverse_property means in RDF;
                    # inverse doesn't apply to data property but only object property.
                    # so we catch the error here to detect what we should fix.
                    (outedge,) = [
                        outedge for outedge in outedges if outedge.abs_uri == prop
                    ]
                    targets_are_class = is_class_fn(sm.get_node(outedge.target))
                    if targets_are_class:
                        invertible_stmts.append(n)
                    elif strict:
                        raise Exception(f"{pid} is not invertible")
                    elif force_inversion:
                        assert isinstance(
                            sm.get_node(outedge.target), DataNode
                        ), "Clearly the model is wrong, you have an inverse property to a literal node"
                        invertible_stmts.append(n)

        # we have N statement, so we are going to have N! - 1 ways. It's literally a cartesian product
        all_choices = []
        for stmt in invertible_stmts:
            # assume that each statement only has one incoming link! fix the for loop if this assumption doesn't hold
            (inedge,) = sm.in_edges(stmt.id)
            choice: list[tuple[ClassNode, Optional[str], Optional[str]]] = [
                (stmt, None, None)
            ]
            for invprop in self.props[
                kgns.uri_to_id(inedge.abs_uri)
            ].inverse_properties:
                choice.append(
                    (
                        stmt,
                        (tmp_abs_uri := kgns.id_to_uri(invprop)),
                        kgns.get_rel_uri(tmp_abs_uri),
                    )
                )
            all_choices.append(choice)

        n_choices = np.prod([len(c) for c in all_choices]) - 1
        if n_choices > 256:
            raise sm_metrics.PermutationExplosion("Too many possible semantic models")

        all_choices_perm: list[
            tuple[tuple[ClassNode, Optional[str], Optional[str]]]
        ] = list(
            itertools.product(*all_choices)
        )  # type: ignore
        assert all(
            invprop is None for _, invprop, _ in all_choices_perm[0]
        ), "First choice is always the current semantic model"
        new_sms = [sm]
        for choice_perm in all_choices_perm[1:]:
            new_sm = sm.copy()
            # we now change the statement from original prop to use the inverse prop (change direction)
            # if the invprop is not None
            for stmt, invprop_abs_uri, invprop_rel_uri in choice_perm:
                if invprop_abs_uri is None or invprop_rel_uri is None:
                    continue
                readable_label = self.get_pnode_label(kgns.uri_to_id(invprop_abs_uri))
                # assume that each statement only has one incoming link! fix the for loop if this assumption doesn't hold
                (inedge,) = sm.in_edges(stmt.id)
                # statement must have only one property
                (outedge,) = [
                    outedge
                    for outedge in sm.out_edges(stmt.id)
                    if outedge.abs_uri == inedge.abs_uri
                ]
                assert (
                    len(new_sm.get_edges_between_nodes(inedge.source, stmt.id)) == 1
                    and len(new_sm.get_edges_between_nodes(stmt.id, outedge.target))
                    == 1
                )
                new_sm.remove_edges_between_nodes(inedge.source, stmt.id)
                new_sm.remove_edges_between_nodes(stmt.id, outedge.target)

                target = sm.get_node(outedge.target)
                if not is_class_fn(target):
                    assert isinstance(target, DataNode)
                    dummy_class_node = ClassNode(
                        abs_uri=self.DUMMY_CLASS_FOR_INVERSION_URI,
                        rel_uri=kgns.get_rel_uri(self.DUMMY_CLASS_FOR_INVERSION_URI),
                    )
                    new_sm.add_node(dummy_class_node)
                    new_sm.add_edge(
                        Edge(
                            source=dummy_class_node.id,
                            target=target.id,
                            abs_uri=str(RDFS.label),
                            rel_uri="rdfs:label",
                        )
                    )
                    outedge_target = dummy_class_node.id
                else:
                    outedge_target = outedge.target
                new_sm.add_edge(
                    Edge(
                        source=outedge_target,
                        target=stmt.id,
                        abs_uri=invprop_abs_uri,
                        rel_uri=invprop_rel_uri,
                        approximation=outedge.approximation,
                        readable_label=readable_label,
                    )
                )
                new_sm.add_edge(
                    Edge(
                        source=stmt.id,
                        target=inedge.source,
                        abs_uri=invprop_abs_uri,
                        rel_uri=invprop_rel_uri,
                        approximation=inedge.approximation,
                        readable_label=readable_label,
                    )
                )
            new_sms.append(new_sm)
        return new_sms

    def get_entity_columns(self, sm: SemanticModel) -> list[int]:
        ent_columns = []
        for dnode in sm.iter_nodes():
            if isinstance(dnode, DataNode):
                inedges = sm.in_edges(dnode.id)
                if len(inedges) == 0:
                    continue
                assert len({edge.abs_uri for edge in inedges}) == 1, inedges
                edge_abs_uri = inedges[0].abs_uri
                if edge_abs_uri in self.ID_PROPS:
                    assert len(inedges) == 1, inedges
                    source = sm.get_node(inedges[0].source)
                    assert (
                        isinstance(source, ClassNode)
                        and not source.abs_uri == self.kgns.statement_uri
                    )
                    ent_columns.append(dnode.col_index)
        return ent_columns

    @classmethod
    def is_uri_column(cls, uri: str):
        """Test if an uri is for specifying the column"""
        return uri.startswith("http://example.com/table/")

    @staticmethod
    def get_column_uri(column_index: int):
        return f"http://example.com/table/{column_index}"

    @staticmethod
    def get_column_index(uri: str):
        assert SemanticModelHelper.is_uri_column(uri)
        return int(uri.replace("http://example.com/table/", ""))

    def extract_claims(
        self, tbl: FullTable, sm: SemanticModel, allow_multiple_ent: bool = True
    ):
        """Extract claims from the table given a semantic model.

        If an entity doesn't have link, its id will be null
        """
        # norm the semantic model first
        sm = self.norm_sm(sm)
        kgns = self.kgns
        schemas = {}
        for u in sm.iter_nodes():
            if not isinstance(u, ClassNode) or kgns.statement_uri == u.abs_uri:
                continue

            schema = {"props": {}, "subject": None, "sm_node_id": u.id}
            for us_edge in sm.out_edges(u.id):
                if us_edge.abs_uri in self.ID_PROPS:
                    v = sm.get_node(us_edge.target)
                    assert isinstance(v, DataNode)
                    assert schema["subject"] is None
                    schema["subject"] = v.col_index
                    continue

                s = sm.get_node(us_edge.target)
                assert isinstance(s, ClassNode) and kgns.statement_uri == s.abs_uri
                assert kgns.is_uri_in_main_ns(us_edge.abs_uri)

                pnode = kgns.uri_to_id(us_edge.abs_uri)
                if pnode not in schema["props"]:
                    schema["props"][pnode] = []

                stmt = {
                    "index": len(schema["props"][pnode]),
                    "value": None,
                    "qualifiers": [],
                }
                schema["props"][pnode].append(stmt)
                for sv_edge in sm.out_edges(s.id):
                    v = sm.get_node(sv_edge.target)

                    assert kgns.is_uri_in_main_ns(sv_edge.abs_uri)
                    if sv_edge.abs_uri == us_edge.abs_uri:
                        assert stmt["value"] is None, "only one property"
                        # this is property
                        if isinstance(v, ClassNode):
                            stmt["value"] = {"type": "classnode", "value": v.id}
                        elif isinstance(v, DataNode):
                            stmt["value"] = {"type": "datanode", "value": v.col_index}
                        else:
                            assert isinstance(v, LiteralNode)
                            stmt["value"] = {"type": "literalnode", "value": v.value}
                    else:
                        # this is qualifier
                        if isinstance(v, ClassNode):
                            stmt["qualifiers"].append(
                                {
                                    "type": "classnode",
                                    "pnode": kgns.uri_to_id(sv_edge.abs_uri),
                                    "value": v.id,
                                }
                            )
                        elif isinstance(v, DataNode):
                            stmt["qualifiers"].append(
                                {
                                    "type": "datanode",
                                    "pnode": kgns.uri_to_id(sv_edge.abs_uri),
                                    "value": v.col_index,
                                }
                            )
                        else:
                            assert isinstance(v, LiteralNode)
                            stmt["qualifiers"].append(
                                {
                                    "type": "literalnode",
                                    "pnode": kgns.uri_to_id(sv_edge.abs_uri),
                                    "value": v.value,
                                }
                            )
            schemas[u.id] = schema

        assert all(
            c.index == ci for ci, c in enumerate(tbl.table.columns)
        ), "Cannot handle table with missing columns yet"

        records = [{} for _ in range(tbl.table.nrows())]
        node2ents = {}

        # extract data props first
        for cid, schema in schemas.items():
            ci: int = schema["subject"]
            col = tbl.table.get_column_by_index(ci)
            for ri, val in enumerate(col.values):
                # get entities
                qnode_ids = sorted(
                    {
                        entity_id
                        for link in tbl.links[ri][ci]
                        for entity_id in link.entities
                        if link.start < link.end
                    }
                )
                if len(qnode_ids) == 0:
                    # create new entity
                    ents = [
                        {
                            "id": f"{ri}-{ci}",
                            "column": ci,
                            "row": ri,
                            "uri": None,
                            "label": val,
                            "props": {},
                        }
                    ]
                else:
                    ents = [
                        {
                            "id": qnode_id,
                            "uri": kgns.id_to_uri(qnode_id),
                            "label": self.get_ent_label(qnode_id),
                            "props": {},
                        }
                        for qnode_id in qnode_ids
                    ]

                if len(ents) > 1:
                    if not allow_multiple_ent:
                        raise Exception("Encounter multiple entities")

                for prop, stmts in schema["props"].items():
                    for ent in ents:
                        assert prop not in ent["props"]
                        ent["props"][prop] = [
                            {"value": None, "qualifiers": {}} for stmt in stmts
                        ]
                        for stmt in stmts:
                            # set statement property
                            if stmt["value"]["type"] == "classnode":
                                # do it in later phase
                                pass
                            elif stmt["value"]["type"] == "datanode":
                                tci = stmt["value"]["value"]
                                ent["props"][prop][stmt["index"]]["value"] = (
                                    tbl.table.get_column_by_index(tci).values[ri]
                                )
                            else:
                                assert stmt["value"]["type"] == "literalnode"
                                ent["props"][prop][stmt["index"]]["value"] = stmt[
                                    "value"
                                ]["value"]

                            # set statement qualifiers
                            for qual in stmt["qualifiers"]:
                                if (
                                    qual["pnode"]
                                    not in ent["props"][prop][stmt["index"]][
                                        "qualifiers"
                                    ]
                                ):
                                    ent["props"][prop][stmt["index"]]["qualifiers"][
                                        qual["pnode"]
                                    ] = []
                                if qual["type"] == "classnode":
                                    # do it in later phase
                                    pass
                                elif qual["type"] == "datanode":
                                    tci = qual["value"]
                                    ent["props"][prop][stmt["index"]]["qualifiers"][
                                        qual["pnode"]
                                    ].append(
                                        tbl.table.get_column_by_index(tci).values[ri]
                                    )
                                elif qual["type"] == "literalnode":
                                    ent["props"][prop][stmt["index"]]["qualifiers"][
                                        qual["pnode"]
                                    ].append(qual["value"])

                for ent in ents:
                    assert (ent["id"], ci) not in records[ri]
                    records[ri][ent["id"], ci] = ent
                node2ents[schema["sm_node_id"], ri] = [ent for ent in ents]

        for cid, schema in schemas.items():
            ci = schema["subject"]
            col = tbl.table.get_column_by_index(ci)
            for ri in range(len(col.values)):
                ulst = node2ents[schema["sm_node_id"], ri]
                for prop, stmts in schema["props"].items():
                    for stmt in stmts:
                        if stmt["value"]["type"] == "classnode":
                            vlst = node2ents[stmt["value"]["value"], ri]
                            for u in ulst:
                                assert len(vlst) > 0
                                u["props"][prop][stmt["index"]]["value"] = vlst[0]["id"]
                                if len(vlst) > 1:
                                    # this statement must not have other qualifiers, so that v can be a list
                                    # and we can create extra statement
                                    assert len(stmt["qualifiers"]) == 0
                                    for v in vlst[1:]:
                                        u["props"][prop].append(
                                            {"value": v["id"], "qualifiers": {}}
                                        )
                        for qual in stmt["qualifiers"]:
                            if qual["type"] == "classnode":
                                for u in ulst:
                                    vlst = node2ents[qual["value"], ri]
                                    u["props"][prop][stmt["index"]]["qualifiers"][
                                        qual["pnode"]
                                    ] = [v["id"] for v in vlst]

        new_records = []
        for ri, record in enumerate(records):
            new_record = {}
            for (ent_id, ci), ent in record.items():
                if ent_id in new_record:
                    # merge the entity
                    for pid, _stmts in ent["props"].items():
                        if pid not in new_record[ent_id]["props"]:
                            new_record[ent_id]["props"][pid] = _stmts
                        else:
                            for _stmt in _stmts:
                                if not any(
                                    x == _stmt for x in new_record[ent_id]["props"][pid]
                                ):
                                    new_record[ent_id]["props"][pid].append(_stmt)
                else:
                    new_record[ent_id] = ent
            new_records.append(new_record)
        return records

    def get_ent_label(self, eid: str):
        """Get WDEntity label from id"""
        if eid not in self.entity_labels:
            return eid
        return f"{self.entity_labels[eid]} ({eid})"

    def get_pnode_label(self, pid: str):
        """Get PNode label from id"""
        if pid not in self.props:
            return pid
        return f"{self.props[pid].label} ({pid})"

    def create_sm(
        self,
        nodes: Mapping[NodeID, ClassNode | DataNode | LiteralNode],
        cpa: Sequence[tuple[NodeID, NodeID, str]],
        cta: Mapping[int, str],
        validate: bool = True,
        on_untype_source_column_node: Literal[
            "create-class", "remove-link"
        ] = "create-class",
    ) -> SemanticModel:
        """Create a semantic model from outputs of CPA and CTA tasks

        # Arguments
            nodes: mapping from node id to DataNode or LiteralNode (leaf nodes) or Statement (represent n-ary relation)
                that we can't generate automatically from the cta. Sources and targets in CPA should all be in this map.
            cpa: list of triples (source, target, predicate)
            cta: mapping from column index to class id
            validate: whether to validate the input to make sure it's correct classes and properties
        """
        if validate:
            for source, target, predicate in cpa:
                assert self.kgns.is_id(predicate)
            for col_index, ent_id in cta.items():
                assert self.kgns.is_id(ent_id)
            for node in nodes.values():
                if isinstance(node, ClassNode):
                    assert self.kgns.is_uri_in_ns(node.abs_uri)
                elif (
                    isinstance(node, LiteralNode)
                    and node.datatype == LiteralNodeDataType.Entity
                ):
                    assert self.kgns.is_uri(node.value) and self.kgns.is_uri_in_main_ns(
                        node.value
                    )

        sm = SemanticModel()
        # make a copy because when we add node into sm, we will modify the node id
        nodes = {uid: copy(u) for uid, u in nodes.items()}

        classmap = {}  # mapping from column to its class node
        # mapping from node id to sm node id, this mapping is built dynamically when iterating over cpa result.
        nodemap: dict[NodeID, int] = {}
        kgns = self.kgns

        col2id = {
            u.col_index: uid for uid, u in nodes.items() if isinstance(u, DataNode)
        }
        for col_index, ent_id in cta.items():

            # somehow, they may end-up predict multiple classes, we need to select one
            if ent_id.find(" ") != -1:
                ent_id = ent_id.split(" ")[0]
            curl = kgns.id_to_uri(ent_id)

            try:
                cnode_label = self.get_ent_label(ent_id)
            except KeyError:
                cnode_label = kgns.get_rel_uri(kgns.id_to_uri(ent_id))

            cnode = ClassNode(
                abs_uri=curl,
                rel_uri=kgns.get_rel_uri(kgns.id_to_uri(ent_id)),
                readable_label=cnode_label,
            )
            dnode = nodes[col2id[col_index]]
            sm.add_node(cnode)
            nodemap[col2id[col_index]] = sm.add_node(dnode)
            classmap[col_index] = cnode.id
            sm.add_edge(
                Edge(
                    source=cnode.id,
                    target=dnode.id,
                    abs_uri=str(RDFS.label),
                    rel_uri=kgns.get_rel_uri(RDFS.label),
                )
            )

        for source, target, predicate in cpa:
            unode = nodes[source]
            vnode = nodes[target]

            if source not in nodemap:
                nodemap[source] = sm.add_node(unode)
            if target not in nodemap:
                nodemap[target] = sm.add_node(vnode)

        # detect and handle untype source column node
        remove_rel_from_nodes = set()

        for source, target, predicate in cpa:
            unode = nodes[source]
            if isinstance(unode, DataNode):
                if unode.col_index not in classmap:
                    # discover untyped source column node
                    if on_untype_source_column_node == "create-class":
                        # this data node has an outgoing edge, but it's untyped
                        # so we create an entity class to represent its type
                        cnode_id = sm.add_node(
                            ClassNode(
                                abs_uri=self.kgns.entity_uri,
                                rel_uri=kgns.get_rel_uri(self.kgns.entity_uri),
                                readable_label=self.kgns.entity_label,
                            )
                        )
                        classmap[unode.col_index] = cnode_id
                        sm.add_edge(
                            Edge(
                                source=cnode_id,
                                target=unode.id,
                                abs_uri=str(RDFS.label),
                                rel_uri=kgns.get_rel_uri(RDFS.label),
                            )
                        )
                    else:
                        assert on_untype_source_column_node == "remove-link"
                        vnode = nodes[target]

                        if (
                            isinstance(vnode, ClassNode)
                            and vnode.abs_uri == kgns.statement_uri
                        ):
                            # this is a statement node, so we need to remove link
                            # from the statement node too because statement node
                            # can't live without the source
                            remove_rel_from_nodes.add(target)
                        remove_rel_from_nodes.add(source)

        if len(remove_rel_from_nodes) > 0:
            assert on_untype_source_column_node == "remove-link"
            cpa = [
                (source, target, predicate)
                for source, target, predicate in cpa
                if source not in remove_rel_from_nodes
            ]

        for source, target, predicate in cpa:
            unode = nodes[source]
            vnode = nodes[target]

            # if source not in nodemap:
            #     nodemap[source] = sm.add_node(unode)
            # if target not in nodemap:
            #     nodemap[target] = sm.add_node(vnode)

            if isinstance(unode, DataNode):
                # outgoing edge is from a class node instead of a data node
                assert unode.col_index in classmap
                # if unode.col_index not in classmap:
                #     # this data node has an outgoing edge, but it's untyped
                #     # so we create an entity class to represent its type
                #     curl = kgns.get_entity_abs_uri(self.ENTITY_ID)
                #     cnode_id = sm.add_node(
                #         ClassNode(
                #             abs_uri=curl,
                #             rel_uri=kgns.get_entity_rel_uri(self.ENTITY_ID),
                #             readable_label=self.kgns.entity_label,
                #         )
                #     )
                #     classmap[unode.col_index] = cnode_id
                #     sm.add_edge(
                #         Edge(
                #             source=cnode_id,
                #             target=unode.id,
                #             abs_uri=str(RDFS.label),
                #             rel_uri=kgns.get_rel_uri(RDFS.label),
                #         )
                #     )
                suid = classmap[unode.col_index]
                source = sm.get_node(suid)
            else:
                source = unode

            if isinstance(vnode, DataNode):
                # if this is an entity column, the link should map to its class
                if vnode.col_index in classmap:
                    target = sm.get_node(classmap[vnode.col_index])
                else:
                    target = vnode
            else:
                target = vnode

            sm.add_edge(
                Edge(
                    source=source.id,
                    target=target.id,
                    abs_uri=(tmp_abs_uri := kgns.id_to_uri(predicate)),
                    rel_uri=kgns.get_rel_uri(tmp_abs_uri),
                    readable_label=self.get_pnode_label(predicate),
                )
            )

        return sm

    def create_sm_from_column_rels(
        self,
        cpa: Sequence[tuple[int, int, str]],
        cta: Mapping[int, str],
        columns: list[str] | dict[int, str],
        validate: bool = True,
    ):
        nodes = {}
        if isinstance(columns, list):
            for ci, cname in enumerate(columns):
                nodes[ci] = DataNode(ci, cname)
        else:
            for ci, cname in columns.items():
                nodes[ci] = DataNode(ci, cname)
        return self.create_sm(nodes, cpa, cta, validate)

    def convert_sm_to_column_rels(
        self, sm: SemanticModel
    ) -> tuple[dict[int, str], list[tuple[int, int, str]]]:
        raise NotImplementedError()
