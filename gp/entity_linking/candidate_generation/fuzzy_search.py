from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import IO, Sequence, Union

import numpy as np
import orjson
import ray
import serde.pickle
import strsim
from gp.actors.data import KGDB, KGDBArgs
from gp.entity_linking.candidate_generation.common import CanGenBasicMethod
from hugedict.prelude import RocksDBDict, RocksDBOptions, rocksdb_load
from kgdata.db import pack_int, unpack_int
from kgdata.splitter import split_a_list
from kgdata.wikidata.config import WikidataDirCfg
from kgdata.wikidata.db import WikidataDB
from ream.workspace import ReamWorkspace
from sm.misc.funcs import assert_isinstance
from sm.misc.ray_helper import (
    add_ray_actors,
    get_ray_actors,
    is_parallelizable,
    ray_actor_map,
)
from symspellpy import SymSpell, Verbosity, helpers
from tqdm import tqdm


@dataclass
class FuzzySearchArgs:
    symspell_db_parentdir: Union[Path, str]
    max_dictionary_edit_distance: int = 2
    prefix_length: int = 7
    count_threshold: int = 1
    hard_limit: int = 1000
    sim_score_coeff: float = 0.9


class FuzzySearchImpl:

    def __init__(
        self,
        kgdb: KGDB | KGDBArgs,
        params: FuzzySearchArgs,
    ) -> None:
        if not isinstance(kgdb, KGDB):
            kgdb = KGDB(kgdb)

        WD_DIR = Path(os.environ["WD_DIR"])

        WikidataDirCfg.init(WD_DIR)
        ReamWorkspace.init(os.environ["REAM_DIR"])
        ReamWorkspace.get_instance().register_base_paths(
            DATA_DIR=Path(os.environ["DATA_DIR"])
        )

        pgrnk_stat_file = WikidataDirCfg.get_instance().entity_pagerank / "pagerank.pkl"
        pgrnk_stat = serde.pickle.deser(pgrnk_stat_file)

        self.params = params
        self.symspell = PersistentSymSpell.load(
            Path(self.params.symspell_db_parentdir)
            / f"symspell_v2_d{self.params.max_dictionary_edit_distance}_l{self.params.prefix_length}_t{self.params.count_threshold}",
        )
        self.log_pagerank = (
            float(np.log(pgrnk_stat["min"] + 1e-20)),
            float(np.log(pgrnk_stat["max"]) - np.log(pgrnk_stat["min"] + 1e-20)),
        )
        self.label2ids = assert_isinstance(kgdb.pydb, WikidataDB).label2ids.cache()
        self.chartok = strsim.CharacterTokenizer()

    def get_candidates_by_queries(
        self, queries: list[str]
    ) -> dict[str, list[tuple[str, float]]]:
        out = {}
        for query in queries:
            out[query] = self.query(query)
        return out

    def query(self, query: str) -> list[tuple[str, float]]:
        cans = {}
        query_t1 = self.chartok.tokenize(query)
        for item in self.symspell.lookup(query, Verbosity.ALL):
            label = item.term
            sim_score = strsim.levenshtein_similarity(
                query_t1, self.chartok.tokenize(label)
            )

            for id, pagerank in self.label2ids[label]:
                adjusted_score = sim_score * self.params.sim_score_coeff + (
                    1 - self.params.sim_score_coeff
                ) * self.norm_pagerank(pagerank)
                cans[id] = adjusted_score
        return sorted(cans.items(), key=lambda x: x[1], reverse=True)[
            : self.params.hard_limit
        ]

    def norm_pagerank(self, score):
        return (np.log(score + 1e-20) - self.log_pagerank[0]) / self.log_pagerank[1]


class FuzzySearch(CanGenBasicMethod[FuzzySearchArgs]):

    VERSION = 101

    def __init__(
        self,
        kgdb: KGDB,
        params: FuzzySearchArgs,
        soft_limit: int,
    ) -> None:
        super().__init__(
            kgdb,
            params,
            batch_size=1024,
            soft_limit=soft_limit,
        )
        self.num_actors = 6
        self.actor_ns = f"fuzzy_search_{self.kgdb.args.get_key()}"

    @cached_property
    def impl(self):
        return FuzzySearchImpl(self.kgdb.args, self.params)

    def get_candidates_by_queries(
        self,
        queries: list[str],
    ) -> dict[str, list[tuple[str, float]]]:
        out = {}
        if is_parallelizable():
            # construct an actor
            actors: Sequence["ray.ObjectRef[FuzzySearchImpl]"] = get_ray_actors(
                self.actor_ns
            )
            if len(actors) == 0:
                actors = add_ray_actors(
                    FuzzySearchImpl,
                    (self.kgdb.args, self.params),
                    self.actor_ns,
                    size=self.num_actors,
                )

            batch_size = 64
            resp = ray_actor_map(
                [actor.get_candidates_by_queries.remote for actor in actors],
                [
                    (queries[i : i + batch_size],)
                    for i in range(0, len(queries), batch_size)
                ],
            )
            for item in resp:
                out.update(item)
        else:
            out = self.impl.get_candidates_by_queries(queries)

        return out


class PersistentSymSpell(SymSpell):
    def _load_dictionary_stream(
        self,
        corpus_stream: IO[str],
        term_index: int,
        count_index: int,
        separator: str = " ",
    ) -> bool:
        """Loads multiple dictionary entries from a stream of word/frequency
        count pairs.

        **NOTE**: Merges with any dictionary data already loaded.

        Args:
            corpus_stream: A file object of the dictionary.
            term_index: The column position of the word.
            count_index: The column position of the frequency count.
            separator: Separator characters between term(s) and count.

        Returns:
            ``True`` after file object is loaded.
        """
        for line in tqdm(corpus_stream, total=50000000):
            parts = line.rstrip().split(separator)
            if len(parts) < 2:
                continue
            count = helpers.try_parse_int64(parts[count_index])
            if count is None:
                continue
            key = parts[term_index]
            self.create_dictionary_entry(key, count)
        return True

    def save(self, outdir: Path | str):
        """Save the symspell to a directory."""
        outdir = Path(outdir)
        dbs = PersistentSymSpell._get_databases(
            outdir, readonly=False, without_deletes=True
        )

        for name in ["words", "below_threshold_words", "bigrams"]:
            db = dbs[name]
            for key, value in getattr(self, f"_{name}").items():
                db[key] = value
            db.compact()

        PersistentSymSpell._save_deletes(self._deletes, outdir, verbose=True)

        serde.pickle.ser(
            {
                "max_length": self._max_length,
                # SymSpell settings used to generate the above
                "count_threshold": self._count_threshold,
                "max_dictionary_edit_distance": self._max_dictionary_edit_distance,
                "prefix_length": self._prefix_length,
            },
            outdir / "attrs.pkl",
        )

    @staticmethod
    def load(indir: Path | str):
        """Load the symspell from a directory."""
        indir = Path(indir)
        attrs = serde.pickle.deser(indir / "attrs.pkl")
        symspell = PersistentSymSpell(
            max_dictionary_edit_distance=attrs["max_dictionary_edit_distance"],
            prefix_length=attrs["prefix_length"],
            count_threshold=attrs["count_threshold"],
        )

        dbs = PersistentSymSpell._get_databases(indir, readonly=True)

        symspell._count_threshold = attrs["count_threshold"]
        symspell._max_dictionary_edit_distance = attrs["max_dictionary_edit_distance"]
        symspell._prefix_length = attrs["prefix_length"]
        symspell._max_length = attrs["max_length"]
        for name, db in dbs.items():
            setattr(symspell, f"_{name}", db)
        return symspell

    @staticmethod
    def _save_deletes(
        deletes: dict[str, list[str]], outdir: Path, verbose: bool = False
    ):
        dbdir = outdir / "deletes.db"
        if (dbdir / "_SUCCESS").exists():
            return

        rawdir = outdir / "deletes"
        rawdir.mkdir(parents=True, exist_ok=True)
        if not (rawdir / "_SUCCESS").exists():
            kvs = [
                orjson.dumps((key, orjson.dumps(value).decode()))
                for key, value in tqdm(
                    deletes.items(), disable=not verbose, mininterval=1
                )
            ]
            split_a_list(kvs, rawdir / "part.jl")
            (rawdir / "_SUCCESS").touch()

        rocksdb_load(
            dbpath=str(dbdir),
            dbopts=RocksDBOptions(
                create_if_missing=True,
                compression_type="lz4",
                bottommost_compression_type="zstd",
            ),
            files=[str(x) for x in rawdir.glob("part-*.jl")],
            format={
                "record_type": {"type": "tuple2", "key": None, "value": None},
                "is_sorted": False,
            },
            verbose=True,
            compact=True,
        )
        (dbdir / "_SUCCESS").touch()

    @staticmethod
    def _get_databases(outdir: Path, readonly: bool, without_deletes: bool = False):
        words: RocksDBDict[str, int] = RocksDBDict(
            path=str(outdir / "words.db"),
            options=RocksDBOptions(
                create_if_missing=True,
            ),
            deser_key=partial(str, encoding="utf-8"),
            deser_value=unpack_int,
            ser_value=pack_int,
            readonly=readonly,
        )
        below_threshold_words: RocksDBDict[str, int] = RocksDBDict(
            path=str(outdir / "below_threshold_words.db"),
            options=RocksDBOptions(
                create_if_missing=True,
            ),
            deser_key=partial(str, encoding="utf-8"),
            deser_value=unpack_int,
            ser_value=pack_int,
            readonly=readonly,
        )
        bigrams: RocksDBDict[str, int] = RocksDBDict(
            path=str(outdir / "bigrams.db"),
            options=RocksDBOptions(
                create_if_missing=True,
            ),
            deser_key=partial(str, encoding="utf-8"),
            deser_value=unpack_int,
            ser_value=pack_int,
            readonly=readonly,
        )
        out: dict = {
            "words": words,
            "below_threshold_words": below_threshold_words,
            "bigrams": bigrams,
        }

        if not without_deletes:
            deletes: RocksDBDict[str, list[str]] = RocksDBDict(
                path=str(outdir / "deletes.db"),
                options=RocksDBOptions(
                    create_if_missing=True,
                    compression_type="lz4",
                    bottommost_compression_type="zstd",
                ),
                deser_key=partial(str, encoding="utf-8"),
                deser_value=orjson.loads,
                ser_value=orjson.dumps,
                readonly=readonly,
            )
            out["deletes"] = deletes

        return out
