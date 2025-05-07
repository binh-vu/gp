from __future__ import annotations

from typing import Any, Sequence

import ray
from gp.actors.data import KGDB, KGDBArgs
from kgdata.wikidata.datasets import entity_pagerank
from ream.cache_helper import Cache, MemBackend
from sm.misc.ray_helper import (
    add_ray_actors,
    get_ray_actors,
    is_parallelizable,
    ray_actor_map,
)


class KGDBService:
    def __init__(self, kgdb: KGDB | KGDBArgs):
        if not isinstance(kgdb, KGDB):
            kgdb = KGDB(kgdb)
        self.kgdb = kgdb

    @Cache.flat_cache(backend=MemBackend(), cache_key=lambda self, key, ids: key)
    def labels(self, key: str, ids: Sequence[str]) -> list[str]:
        db = self.kgdb.pydb.entity_labels.cache()
        return [db[id] for id in ids]

    @Cache.flat_cache(backend=MemBackend(), cache_key=lambda self, key, ids: key)
    def aliases(self, key: str, ids: Sequence[str]) -> list[list[str]]:
        db = self.kgdb.pydb.entity_metadata.cache()
        return [sorted(db[id].aliases.flatten()) for id in ids]

    @Cache.flat_cache(backend=MemBackend(), cache_key=lambda self, key, ids: key)
    def pageranks(self, key: str, ids: Sequence[str]) -> list[float]:
        db = self.kgdb.pydb.entity_pagerank.cache()
        return [db[id] for id in ids]


class KGDBBatchService:
    def __init__(self, kgdb: KGDBArgs):
        self.kgdb = kgdb
        self.ns = "gp.entity_linking.populate.KGDBService:" + kgdb.get_key()

    @Cache.flat_cache(backend=MemBackend(), cache_key=lambda self, key_ids: key_ids[0])
    def batch_labels(
        self, lst_key_ids: list[tuple[str, Sequence[str]]]
    ) -> list[list[str]]:
        return ray_actor_map(
            [actor.labels.remote for actor in self.get_actors()],
            lst_key_ids,
        )

    @Cache.flat_cache(backend=MemBackend(), cache_key=lambda self, key_ids: key_ids[0])
    def batch_aliases(
        self, lst_key_ids: list[tuple[str, Sequence[str]]]
    ) -> list[list[list[str]]]:
        return ray_actor_map(
            [actor.aliases.remote for actor in self.get_actors()],
            lst_key_ids,
        )

    @Cache.flat_cache(backend=MemBackend(), cache_key=lambda self, key_ids: key_ids[0])
    def batch_pageranks(
        self, lst_key_ids: list[tuple[str, Sequence[str]]]
    ) -> list[list[float]]:
        return ray_actor_map(
            [actor.pageranks.remote for actor in self.get_actors()],
            lst_key_ids,
        )

    def get_actors(self) -> list["ray.ObjectRef[KGDBService]"]:
        actors = get_ray_actors(self.ns)
        if len(actors) == 0:
            actors = add_ray_actors(
                KGDBService, (self.kgdb,), self.ns, size=6, scope=-1
            )
        return actors
