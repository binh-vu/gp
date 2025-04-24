from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from linecache import cache
from typing import Optional

from gp.actors.data.prelude import KGDB, GPExample, KGDBArgs
from gp.actors.el.canrank import CanRankActor
from gp.misc.appconfig import AppConfig
from gp.misc.conversions import to_rust_table
from gp.semanticmodeling import text_parser
from gp.semanticmodeling.cangraph_cfg import CanGraphExtractorCfg
from gp.semanticmodeling.text_parser import TextParser, TextParserConfigs
from gramsplus.core.algorithms import (
    CanGraphExtractedResult,
    extract_cangraph,
    par_extract_cangraphs,
)
from gramsplus.core.models import TableCells
from ream.prelude import BaseActor, Cache
from sm.dataset import Example, FullTable
from sm.misc.ray_helper import enhance_error_info, ray_map, ray_put
from tqdm import tqdm


@dataclass
class CanGraphActorArgs:
    topk: Optional[int] = field(
        default=100,
        metadata={
            "help": "keeping maximum top k candidates for each cell",
        },
    )
    text_parser: TextParserConfigs = field(
        default_factory=TextParserConfigs,
        metadata={"help": "Configuration for the text parser"},
    )
    cangraph_extractor: CanGraphExtractorCfg = field(
        default_factory=CanGraphExtractorCfg,
        metadata={"help": "Configuration for the candidate graph extractor"},
    )


class CanGraphActor(BaseActor[CanGraphActorArgs]):
    VERSION = 117

    def __init__(self, params: CanGraphActorArgs, canrank_actor: CanRankActor):
        super().__init__(params, dep_actors=[canrank_actor])

        self.canrank_actor = canrank_actor
        self.db_actor = canrank_actor.db_actor

    @cached_property
    def text_parser(self) -> TextParser:
        return TextParser(self.params.text_parser)

    @cached_property
    def ru_cangraph_extractor(self):
        return self.params.cangraph_extractor.to_rust()

    @Cache.cache(
        backend=Cache.sqlite.pickle(
            filename="cangraph", mem_persist=True, compression="lz4"
        ),
        cache_key=lambda self, example, parallel=True: example.id,
        disable=lambda self: not AppConfig.get_instance().is_cache_enable,
    )
    def __call__(self, ex: GPExample, parallel: bool = True) -> CanGraphExtractedResult:
        cans = self.get_candidate_entities([ex])[0]
        nrows, ncols = ex.table.table.shape()
        text_parser = self.text_parser

        return extract_cangraph(
            to_rust_table(ex, cans),
            TableCells(
                [
                    [
                        text_parser.parse(ex.table.table[ri, ci]).to_rust()
                        for ci in range(ncols)
                    ]
                    for ri in range(nrows)
                ]
            ),
            self.db_actor.kgdbs[ex.kgname].rudb,
            self.ru_cangraph_extractor,
            None,
            parallel=parallel,
        )

    @Cache.flat_cache(
        backend=Cache.sqlite.pickle(
            filename="cangraph", mem_persist=True, compression="lz4"
        ),
        cache_key=lambda self, example, verbose=False: example.id,
        disable=lambda self: not AppConfig.get_instance().is_cache_enable,
    )
    def batch_call(
        self, exs: list[GPExample], verbose: bool = False
    ) -> list[CanGraphExtractedResult]:
        if len(exs) == 0:
            return []

        assert all(
            ex.kgname == exs[0].kgname for ex in exs
        ), "Don't support multiple KGs in the same function call"

        ex_cans = self.get_candidate_entities(exs)
        text_parser = self.text_parser

        tables = []
        table_cells = []

        for ei, ex in tqdm(enumerate(exs), disable=not verbose):
            nrows, ncols = ex.table.table.shape()
            tbl = to_rust_table(ex, ex_cans[ei])
            cells = TableCells(
                [
                    [
                        text_parser.parse(ex.table.table[ri, ci]).to_rust()
                        for ci in range(ncols)
                    ]
                    for ri in range(nrows)
                ]
            )

            tables.append(tbl)
            table_cells.append(cells)

        return par_extract_cangraphs(
            tables,
            table_cells,
            self.db_actor.kgdbs[exs[0].kgname].rudb,
            self.ru_cangraph_extractor,
            None,
            verbose=verbose,
        )

    def get_candidate_entities(self, exs: list[GPExample]):
        ex_cans = self.canrank_actor.batch_call(exs)
        if self.params.topk is not None:
            return [cans.top_k(self.params.topk) for cans in ex_cans]
        return ex_cans
