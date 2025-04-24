from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Mapping, Union

import orjson
from hugedict.misc import identity
from hugedict.sqlitedict import SqliteDict, SqliteDictFieldType
from kgdata.models.ont_class import OntologyClass
from kgdata.models.ont_property import OntologyProperty
from loguru import logger
from sm.evaluation.hierarchy_scoring_fn import (
    INF_DISTANCE,
    MAX_ANCESTOR_DISTANCE,
    MAX_DESCENDANT_DISTANCE,
    HierarchyScoringFn,
)
from sm.namespaces.namespace import KnowledgeGraphNamespace
from sm.namespaces.wikidata import WikidataNamespace

MAX_DISTANCE = max(MAX_ANCESTOR_DISTANCE, MAX_DESCENDANT_DISTANCE)


class ItemType(str, Enum):
    CLASS = "class"
    PROPERTY = "property"


class SqliteItemDistance(SqliteDict[str, int]):
    def __init__(
        self,
        path: Union[str, Path],
        items: Mapping[str, OntologyClass] | Mapping[str, OntologyProperty],
        kgns: KnowledgeGraphNamespace,
        timeout: float = 5.0,
    ):
        super().__init__(
            path,
            keytype=SqliteDictFieldType.str,
            ser_value=identity,
            deser_value=identity,
            valuetype=SqliteDictFieldType.int,
            timeout=timeout,
        )

        logger.warning("SqliteItemDistance is deprecated, use KGItemDistance instead")

        self._items = items
        self._cache_distance = {}
        self.uri2id = kgns.uri_to_id
        self.is_valid_uri = kgns.is_uri_in_ns
        self.is_valid_id = kgns.is_id

    def get_distance(self, pred_item: str, target_item: str) -> int:
        """Calculate distance between two items. Positive if pred_id is the ancestor of target_id, negative otherwise."""
        if pred_item == target_item:
            return 0

        if (pred_item, target_item) not in self._cache_distance:
            if self.is_valid_uri(pred_item):
                pred_id = self.uri2id(pred_item)
            else:
                assert self.is_valid_id(pred_item), pred_item
                pred_id = pred_item

            if self.is_valid_uri(target_item):
                target_id = self.uri2id(target_item)
            else:
                assert self.is_valid_id(target_item), target_item
                target_id = target_item

            key = orjson.dumps([pred_id, target_id]).decode()
            if key not in self:
                distance = self._calculate_distance(pred_id, target_id)
                self[key] = distance
            self._cache_distance[pred_item, target_item] = self[key]
        return self._cache_distance[pred_item, target_item]

    def batch_get_distance(self, items: list[tuple[str, str]]) -> list[int]:
        distances = []
        newdistances = []
        for pred_item, target_item in items:
            if pred_item == target_item:
                distances.append(0)
                continue

            if (pred_item, target_item) not in self._cache_distance:
                if self.is_valid_uri(pred_item):
                    pred_id = self.uri2id(pred_item)
                else:
                    assert self.is_valid_id(pred_item), pred_item
                    pred_id = pred_item

                if self.is_valid_uri(target_item):
                    target_id = self.uri2id(target_item)
                else:
                    assert self.is_valid_id(target_item), target_item
                    target_id = target_item

                key = orjson.dumps([pred_id, target_id]).decode()
                if key not in self:
                    distance = self._calculate_distance(pred_id, target_id)
                    newdistances.append((key, distance))
                else:
                    distance = self[key]
                self._cache_distance[pred_item, target_item] = distance

            distances.append(self._cache_distance[pred_item, target_item])
        if len(newdistances) > 0:
            self.batch_insert(newdistances)
        return distances

    def _calculate_distance(self, pred_id: str, target_id: str) -> int:
        """Calculate distance between two items. Positive if pred_id is the ancestor of target_id, negative otherwise."""
        targ_obj = self._items[target_id]

        if pred_id in targ_obj.ancestors:
            # predicted item is the ancestor of the target
            return self._calculate_distance_to_ancestors(targ_obj, pred_id)

        pred_obj = self._items[pred_id]
        if target_id in pred_obj.ancestors:
            # predicted item is the descendant of the target
            return -self._calculate_distance_to_ancestors(pred_obj, target_id)

        return INF_DISTANCE

    def _calculate_distance_to_ancestors(
        self, source: OntologyClass | OntologyProperty, target_id: str
    ) -> int:
        visited = {}
        stack: list[tuple[str, int]] = [(uid, 1) for uid in source.parents]
        while len(stack) > 0:
            uid, distance = stack.pop()
            if uid in visited and distance >= visited[uid]:
                # we have visited this node before and since last time we visit
                # the previous route is shorter, so we don't need to visit it again
                continue

            visited[uid] = distance
            u = self._items[uid]
            if target_id == uid or target_id not in u.ancestors:
                # this is a dead-end path, don't need to explore
                continue
            for parent_id in u.parents:
                if distance < MAX_DISTANCE:
                    # still within the distance limit, we can explore further
                    stack.append((parent_id, distance + 1))
        # we may have not visited the target node because its too far away (> MAX_DISTANCE).
        return visited.get(target_id, INF_DISTANCE)


class KGItemDistance:
    """Calculate distance between items from KGData"""

    def __init__(
        self,
        items: Mapping[str, OntologyClass] | Mapping[str, OntologyProperty],
        kgns: KnowledgeGraphNamespace,
    ):
        self.items = items
        self.uri2id = kgns.uri_to_id
        self.is_valid_id = kgns.is_id
        self.is_valid_uri = kgns.is_uri_in_ns

    def get_distance(self, pred_item: str, target_item: str) -> int:
        if pred_item == target_item:
            return 0

        if self.is_valid_uri(pred_item):
            pred_id = self.uri2id(pred_item)
        else:
            assert self.is_valid_id(pred_item), pred_item
            pred_id = pred_item

        if self.is_valid_uri(target_item):
            target_id = self.uri2id(target_item)
        else:
            assert self.is_valid_id(target_item), target_item
            target_id = target_item

        pred_obj = self.items[pred_id]
        target_obj = self.items[target_id]

        if pred_id in target_obj.ancestors:
            return target_obj.ancestors[pred_id]
        if target_id in pred_obj.ancestors:
            return -pred_obj.ancestors[target_id]
        return INF_DISTANCE

    def batch_get_distance(self, items: list[tuple[str, str]]) -> list[int]:
        return [
            self.get_distance(pred_item, target_item)
            for pred_item, target_item in items
        ]


def get_hierarchy_scoring_fn(
    items: Mapping[str, OntologyClass] | Mapping[str, OntologyProperty],
    kgns: KnowledgeGraphNamespace,
) -> HierarchyScoringFn:
    return HierarchyScoringFn(KGItemDistance(items, kgns))
