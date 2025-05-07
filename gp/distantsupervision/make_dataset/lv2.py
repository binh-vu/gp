from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import serde.json
from gp.distantsupervision.make_dataset.interface import LabelFn
from gp.distantsupervision.make_dataset.lv1 import LabelV1, LabelV1Args
from kgdata.models.entity import Entity
from kgdata.models.ont_class import OntologyClass
from kgdata.wikidata.models.wdentity import WDEntity
from sm.dataset import FullTable
from sm.inputs.link import EntityIdWithScore
from sm.misc.funcs import assert_not_null, import_func


@dataclass
class LabelV2Args:
    base_labeler: LabelV1Args
    type_header_agreement_file: Path
    # path to the function that normalizes the column name
    norm_name_fn: str


class LabelV2(LabelFn):
    VERSION = 100

    def __init__(
        self,
        args: LabelV2Args,
        entities: Mapping[str, WDEntity | Entity],
        pagerank: Mapping[str, float],
        classes: Mapping[str, OntologyClass],
    ):
        self.args = args
        self.base_labeler = LabelV1(args.base_labeler, entities, pagerank, classes)
        self.norm_name_fn: Callable[[str], str] = import_func(args.norm_name_fn)
        self.classes = classes

        self.type_header_agreement = self.load_type_header_agreement()
        self.type_header_agreement_keys = set(self.type_header_agreement.keys())

    def label(
        self, table: FullTable, columns: list[int]
    ) -> list[list[EntityIdWithScore]]:
        return [self.label_column(table, ci) for ci in columns]

    def label_column(self, table: FullTable, ci: int) -> list[EntityIdWithScore]:
        col_name = table.table.get_column_by_index(ci).clean_multiline_name
        assert col_name is not None
        col_name = self.norm_name_fn(col_name)

        types = self.base_labeler.label_column(table.links[:, ci])
        valid_types = []
        for type in types:
            if type.id in self.type_header_agreement:
                if col_name in self.type_header_agreement[type.id]:
                    valid_types.append(type)
            else:
                cls = self.classes[type.id]
                common_ancestors = self.type_header_agreement_keys.intersection(
                    cls.ancestors.keys()
                )
                if any(
                    col_name in self.type_header_agreement[ancestor]
                    for ancestor in common_ancestors
                ):
                    valid_types.append(type)

        return valid_types

    def load_type_header_agreement(self) -> dict[str, dict[str, dict]]:
        label_parser = re.compile(r"[^(]+ \((Q\d+)\)")
        obj = {
            assert_not_null(label_parser.match(type)).group(1): headers
            for type, headers in serde.json.deser(
                self.args.type_header_agreement_file
            ).items()
        }
        return obj
