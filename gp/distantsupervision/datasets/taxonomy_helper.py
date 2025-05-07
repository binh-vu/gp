from __future__ import annotations

import re
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Mapping

import serde.textline
from gramsplus.misc.evaluation.unorganized import IndirectDictAccess, reorder2tree
from kgdata.models.ont_class import OntologyClass
from kgdata.models.ont_property import OntologyProperty


class TaxonomyFn:
    @classmethod
    def normalize_types(
        cls,
        types: list[str] | list[tuple[str, float]],
        predefined_types: dict[str, int],
        collection: Mapping[str, OntologyClass] | Mapping[str, OntologyProperty],
        skip_if_not_found=False,
    ) -> list[str] | list[tuple[str, float]]:
        """Normalize a type to find the closest most-specific one from the taxonomy. If a type
        doesn't have the closest one, return the original type.

        To identify the closest type, related types are sorted by their depth and prefer the one
        appeared in the taxonomy first.
        """
        newtypes = {}
        newtype_scores = {}
        for item in types:
            if isinstance(item, str):
                type, score = item, 0.0
            else:
                type, score = item
            ancestors = collection[type].ancestors
            foundtypes = [
                (pretype, depth)
                for pretype, depth in predefined_types.items()
                if pretype == type or pretype in ancestors
            ]
            if len(foundtypes) > 0:
                foundtype, depth = max(foundtypes, key=itemgetter(1))
                newtypes[foundtype] = max(depth, newtypes.get(foundtype, 0))
                newtype_scores[foundtype] = max(score, newtype_scores.get(foundtype, 0))
            elif not skip_if_not_found:
                newtypes[type] = 100
                newtype_scores[type] = score

        normed_types = [
            item
            for item, _ in sorted(newtypes.items(), key=lambda x: x[1], reverse=True)
        ]
        if isinstance(types[0], str):
            return normed_types

        return [(type, newtype_scores[type]) for type in normed_types]

    @classmethod
    def load_predefined_types(
        cls, high_level_concept_file: Path, classes: Mapping[str, OntologyClass]
    ) -> dict[str, int]:
        # load the predefined high-level types from disk
        predefined_types: dict[str, int] = {}
        tmp = set()
        for line in serde.textline.deser(high_level_concept_file):
            if line.find("#") != -1:
                line = line[: line.find("#")]
            line = line.strip()
            if line == "":
                continue
            tmp.add(cls.label2id(line))

        trees = reorder2tree(
            list(tmp), IndirectDictAccess(classes, attrgetter("ancestors"))
        ).trees
        for tree in trees:
            for item in tree.get_flatten_hierarchy():
                predefined_types[item.id] = max(
                    item.depth, predefined_types.get(item.id, 0)
                )
        assert len(predefined_types) == len(tmp)
        return predefined_types

    @classmethod
    def label2id(cls, label: str):
        m = re.match(r"[^(]*\(([QP]\d+)\)", label)
        assert m is not None, label
        return m.group(1)
