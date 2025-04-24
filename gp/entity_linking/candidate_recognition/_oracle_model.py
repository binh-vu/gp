from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from gp.entity_linking.candidate_recognition._interface import ICanReg
from gp.misc.evaluation.sm_wikidata import SemanticModelHelper
from kgdata.models import Ontology
from kgdata.models.ont_property import OntologyProperty
from sm.dataset import Example, FullTable
from sm.namespaces.namespace import KnowledgeGraphNamespace, OutOfNamespace
from sm.typing import InternalID


class OracleCanReg(ICanReg):
    VERSION = 102

    def __init__(
        self,
        include_should_be_entity_columns: bool,
        include_link_columns: bool,
    ):
        """
        Args:
            include_should_be_entity_columns: Whether to include should_be_entity_columns in the prediction
            include_link_columns: Whether to include columns that contain links in the prediction
        """
        self.include_should_be_entity_columns = include_should_be_entity_columns
        self.include_link_columns = include_link_columns

    def __call__(
        self, examples: Sequence[Example[FullTable]], ontology: Ontology
    ) -> list[list[int]]:
        output = []
        for example in examples:
            entity_columns = []
            for col in example.table.table.columns:
                ci = col.index
                if is_column_entity(example, ci):
                    entity_columns.append(ci)
                elif self.include_should_be_entity_columns and should_be_column_entity(
                    example, ci, ontology.props, ontology.kgns
                ):
                    entity_columns.append(ci)
                elif self.include_link_columns and column_has_links(example, ci):
                    entity_columns.append(ci)
            output.append(entity_columns)
        return output


def is_column_entity(example: Example[FullTable], ci: int):
    for sm in example.sms:
        if not sm.has_data_node(ci):
            continue
        stypes = sm.get_semantic_types_of_column(ci)
        if any(
            stype.predicate_abs_uri in SemanticModelHelper.ID_PROPS for stype in stypes
        ):
            return True
    return False


def column_has_links(example: Example[FullTable], ci: int):
    for celllinks in example.table.links[:, ci]:
        if any(len(link.entities) > 0 for link in celllinks):
            return True
    return False


def should_be_column_entity(
    example: Example[FullTable],
    ci: int,
    props: Mapping[InternalID, OntologyProperty],
    kgns: KnowledgeGraphNamespace,
):
    for sm in example.sms:
        if not sm.has_data_node(ci):
            continue
        dnode = sm.get_data_node(ci)
        for inedge in sm.in_edges(dnode.id):
            try:
                propid = kgns.uri_to_id(inedge.abs_uri)
            except OutOfNamespace:
                continue

            try:
                if props[propid].is_object_property():
                    return True
            except KeyError:
                pass
    return False
