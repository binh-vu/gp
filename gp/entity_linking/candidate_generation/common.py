from __future__ import annotations

import os
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Dict, Iterable, List, Optional, Sequence, TypeVar

import numpy as np
import orjson
from gp.actors.data import KGDB, GPExample, KGDBArgs
from gp.misc.appconfig import AppConfig
from hugedict.prelude import RocksDBDict, RocksDBOptions
from libactor.actor import Actor
from loguru import logger
from sm.dataset import Example, FullTable
from sm.misc.funcs import batch
from sm.misc.matrix import Matrix
from sm.misc.ray_helper import ray_map, ray_put
from smml.data_model_helper import NumpyDataModel
from tqdm.auto import tqdm

P = TypeVar("P")


@dataclass(slots=True)
class CanEnt:
    id: str
    score: float


class TableCanGenResult(NumpyDataModel):
    __slots__ = ["index", "ent_id", "ent_score"]

    # mapping from col_index => row_index => (start, end)
    index: dict[int, dict[int, tuple[int, int]]]
    ent_id: Annotated[np.ndarray, "String array (Object type)"]
    ent_score: Annotated[np.ndarray, "f64"]

    def __init__(
        self,
        index: dict[int, dict[int, tuple[int, int]]],
        ent_id: Annotated[np.ndarray, "String array (Object type)"],
        ent_score: Annotated[np.ndarray, "f64"],
    ):
        self.index = index
        self.ent_id = ent_id
        self.ent_score = ent_score

    @staticmethod
    def from_matrix(matrix: Matrix[list[CanEnt]]):
        index = {}
        ids = []
        scores = []

        nrows, ncols = matrix.shape()
        for ci in range(ncols):
            index[ci] = {}
            for ri in range(nrows):
                cans = matrix[ri][ci]
                if len(cans) == 0:
                    continue

                index[ci][ri] = (len(ids), len(ids) + len(cans))
                ids.extend([c.id for c in cans])
                scores.extend([c.score for c in cans])

        return TableCanGenResult(
            index=index,
            ent_id=np.asarray(ids, dtype=np.object_),
            ent_score=np.asarray(scores, dtype=np.float64),
        )

    def to_matrix(self, shp: tuple[int, int]) -> Matrix[list[CanEnt]]:
        matrix = Matrix.default(shp, list)
        for ci in self.index:
            for ri in self.index[ci]:
                start, end = self.index[ci][ri]
                matrix[ri][ci] = [
                    CanEnt(self.ent_id[i], self.ent_score[i]) for i in range(start, end)
                ]
        return matrix

    def has_cell_candidates(self, ri: int, ci: int) -> bool:
        return ci in self.index and ri in self.index[ci]

    def get_cell_candidates(self, ri: int, ci: int) -> list[CanEnt]:
        if not self.has_cell_candidates(ri, ci):
            return []
        start, end = self.index[ci][ri]
        return [CanEnt(self.ent_id[i], self.ent_score[i]) for i in range(start, end)]

    def top_k(self, k: int) -> TableCanGenResult:
        matrix = self.to_matrix(self.get_min_shape())
        # using sort instead of heapq.nlargest for stability
        return TableCanGenResult.from_matrix(
            matrix.map(lambda lst: sorted(lst, key=lambda x: x.score, reverse=True)[:k])
        )

    def get_min_shape(self) -> tuple[int, int]:
        """Get the minimum shape of the matrix"""
        ncols = max(self.index.keys()) + 1
        nrows = (
            max(max(row.keys()) if len(row) > 0 else 0 for row in self.index.values())
            + 1
        )
        return nrows, ncols


TableCanGenResult.init()


class CanGenMethod:
    """Candidate ranking method that find candidates from a cell and its surrounding context"""

    @abstractmethod
    def get_candidates(
        self,
        examples: Sequence[GPExample],
        entity_columns: Sequence[list[int]],
        verbose: bool = False,
    ) -> list[TableCanGenResult]:
        pass


class CanGenBasicMethod(CanGenMethod, Actor[P]):
    """Candidate generation method that find candidates from text"""

    def __init__(
        self,
        kgdb: KGDB,
        params: P,
        batch_size: Optional[int] = None,
        soft_limit: int = 1000,
        is_cache_enable: Optional[bool] = None,
    ):
        super().__init__(params, [])
        self.kgdb = kgdb

        self.batch_size = batch_size
        self.soft_limit = soft_limit

        self.method = self.__class__.__name__

        if is_cache_enable is None:
            is_cache_enable = AppConfig.get_instance().is_cache_enable
        if is_cache_enable:
            dbpath = self.actor_dir / "cangen_db"
            dbpath.mkdir(parents=True, exist_ok=True)

            self.kvstore = RocksDBDict(
                path=str(dbpath),
                options=RocksDBOptions(create_if_missing=True),
                deser_key=partial(str, encoding="utf-8"),
                deser_value=pickle.loads,
                ser_value=pickle.dumps,
                # allow to control if the cache should be opened in read-only mode
                # to run multiple experiments in parallel
                readonly=os.environ.get("CANGEN_ROCKSDB_READONLY", "0") == "1",
            )
        else:
            self.kvstore = None

    @abstractmethod
    def get_candidates_by_queries(
        self, queries: List[str]
    ) -> Dict[str, list[tuple[str, float]]]:
        """Generate list of candidate entities for each query."""

    def cached_get_candidates_by_queries(
        self, queries: list[str], cache_only: bool = False
    ) -> dict[str, list[tuple[str, float]]]:
        """Generate list of candidate entities for each query. This should fit to a batch"""
        if self.kvstore is None:
            return self.get_candidates_by_queries(queries)

        output = {
            q: self.kvstore[q][: self.soft_limit] for q in queries if q in self.kvstore
        }
        unk_queries = [q for q in queries if q not in output]
        if not cache_only:
            unk_query_results = self.get_candidates_by_queries(unk_queries)
        else:
            unk_query_results = {q: [] for q in unk_queries}

        for q in unk_queries:
            # sort the results by score and ids so that
            # the order is always deterministic
            res = sorted(unk_query_results[q], key=lambda x: (-x[1], x[0]))
            if not cache_only:
                self.kvstore[q] = res
            output[q] = res[: self.soft_limit]
        return output

    def get_candidates(
        self,
        examples: List[Example[FullTable]],
        entity_columns: list[list[int]],
        verbose: bool = False,
    ) -> list[TableCanGenResult]:
        if verbose:
            logger.debug("Create queries")
        queries = set()
        for ei, example in enumerate(examples):
            for ci in entity_columns[ei]:
                for cell in example.table.table.get_column_by_index(ci).values:
                    queries.add(str(cell))

        if verbose:
            logger.debug("find candidates")
        queries = list(queries)
        if self.kvstore is not None:
            original_queries = queries
            queries = [q for q in queries if q not in self.kvstore]
        else:
            original_queries = []

        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        search_results: dict[str, list[tuple[str, float]]] = {}
        with tqdm(
            total=len(queries), desc="query candidates", disable=not verbose
        ) as pbar:
            for i in range(0, len(queries), batch_size):
                subqueries = queries[i : i + batch_size]
                batch_search_results = self.get_candidates_by_queries(subqueries)
                if self.kvstore is not None:
                    for q in subqueries:
                        # sort the results by score and ids so that
                        # the order is always deterministic
                        res = sorted(
                            batch_search_results[q], key=lambda x: (-x[1], x[0])
                        )
                        self.kvstore[q] = res
                        search_results[q] = res[: self.soft_limit]
                else:
                    for q in subqueries:
                        search_results[q] = batch_search_results[q][: self.soft_limit]

                pbar.update(len(subqueries))

        if self.kvstore is not None:
            with tqdm(
                total=len(original_queries) - len(queries),
                desc="read queries from cache",
                disable=not verbose,
            ) as pbar:
                for q in original_queries:
                    if q not in search_results:
                        search_results[q] = self.kvstore[q][: self.soft_limit]
                        pbar.update(1)

        if verbose:
            logger.debug("find candidates... done! creating the output")

        output = []
        for ei, example in enumerate(examples):
            matrix = Matrix.default(example.table.table.shape(), list)
            for ci in entity_columns[ei]:
                for ri, cell in enumerate(
                    example.table.table.get_column_by_index(ci).values
                ):
                    cell = example.table.table[ri, ci]
                    query = str(cell)
                    query_res = search_results[query]

                    matrix[ri][ci] = [CanEnt(c[0], c[1]) for c in query_res]

            output.append(TableCanGenResult.from_matrix(matrix))

        if verbose:
            logger.debug("creating the output... done!")

        return output


def populate(kgdb: KGDB | KGDBArgs, ids: Iterable[str]):
    db = KGDB.get_instance(kgdb).rudb

    entid: Sequence[str] = []
    label: Sequence[str] = []
    description: Sequence[str] = []
    aliases: Sequence[str] = []
    popularity: Sequence[float] = []

    for id in ids:
        if not db.has_entity(id):
            # entity is not in the database
            newid = db.get_redirected_entity_id(id)
            if newid is None:
                continue
            id = newid
        ent = db.get_entity_metadata(id)

        entid.append(id)
        label.append(ent.label.as_lang_default())
        description.append(ent.description.as_lang_default())
        aliases.append(orjson.dumps(sorted(ent.get_all_aliases())).decode())
        popularity.append(db.get_entity_pagerank(id))

    return entid, label, description, aliases, popularity
