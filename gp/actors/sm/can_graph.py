from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

from gp.actors.data import KGDB
from gp.entity_linking.candidate_generation.common import TableCanGenResult
from gp.misc.conversions import to_rust_table
from gp.semanticmodeling.cangraph_cfg import CanGraphExtractorCfg
from gp.semanticmodeling.text_parser import TextParser, TextParserConfigs
from gp_core.algorithms import (
    CanGraphExtractedResult,
    extract_cangraph,
    par_extract_cangraphs,
)
from gp_core.models import TableCells
from libactor.actor import Actor
from libactor.cache import (
    BackendFactory,
    BorrowBackend,
    FlatCacheArg,
    IdentObj,
    cache,
    flat_cache,
)
from loguru import logger
from sm.dataset import Example, FullTable
from timer import Timer
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


class CanGraphActor(Actor[CanGraphActorArgs]):
    VERSION = 117

    def __init__(self, params: CanGraphActorArgs):
        super().__init__(params, dep_actors=[])
        self.batch_invoke_cache = {}

    @cached_property
    def text_parser(self) -> TextParser:
        return TextParser(self.params.text_parser)

    @cached_property
    def ru_cangraph_extractor(self):
        return self.params.cangraph_extractor.to_rust()

    @cache(
        backend=BackendFactory.actor.sqlite.pickle(
            filename="extract_cangraph.sqlite", mem_persist=True
        ),
        cache_args=["example", "can_ent", "kgdb"],
    )
    def invoke(
        self,
        example: IdentObj[Example[FullTable]],
        can_ent: IdentObj[TableCanGenResult],
        kgdb: IdentObj[KGDB],
        parallel: bool = True,
    ) -> CanGraphExtractedResult:
        if (example.key, can_ent.key) in self.batch_invoke_cache:
            return self.batch_invoke_cache[example.key, can_ent.key]

        ex = example.value
        nrows, ncols = ex.table.table.shape()
        text_parser = self.text_parser

        return extract_cangraph(
            to_rust_table(ex, can_ent.value),
            TableCells(
                [
                    [
                        text_parser.parse(ex.table.table[ri, ci]).to_rust()
                        for ci in range(ncols)
                    ]
                    for ri in range(nrows)
                ]
            ),
            kgdb.value.rudb,
            self.ru_cangraph_extractor,
            None,
            parallel=parallel,
        )

    # @flat_cache(backend=BorrowBackend(invoke), cache_args=[])
    def batch_invoke(
        self,
        exs: list[IdentObj[Example[FullTable]]],
        can_ents: list[IdentObj[TableCanGenResult]],
        kgdb: IdentObj[KGDB],
        verbose: bool = True,
    ):
        if len(exs) == 0:
            return []

        text_parser = self.text_parser

        tables = []
        table_cells = []

        for ei, ex in tqdm(enumerate(exs), disable=not verbose):
            nrows, ncols = ex.value.table.table.shape()
            tbl = to_rust_table(ex.value, can_ents[ei].value)
            cells = TableCells(
                [
                    [
                        text_parser.parse(ex.value.table.table[ri, ci]).to_rust()
                        for ci in range(ncols)
                    ]
                    for ri in range(nrows)
                ]
            )

            tables.append(tbl)
            table_cells.append(cells)

        with Timer().watch_and_report(
            "Extracting candidate graphs", print_fn=logger.info, preprint=True
        ):
            cangraphs = par_extract_cangraphs(
                tables,
                table_cells,
                kgdb.value.rudb,
                self.ru_cangraph_extractor,
                None,
                verbose=verbose,
            )

        with Timer().watch_and_report(
            "Caching candidate graphs", print_fn=logger.info, preprint=True
        ):
            self.batch_invoke_cache = {}
            for ex, can_ent, cangraph in zip(exs, can_ents, cangraphs):
                self.batch_invoke_cache[ex.key, can_ent.key] = cangraph
                self.invoke(ex, can_ent, kgdb, False)
            self.batch_invoke_cache = {}
        return cangraphs

    # @Cache.flat_cache(
    #     backend=Cache.sqlite.pickle(
    #         filename="cangraph", mem_persist=True, compression="lz4"
    #     ),
    #     cache_key=lambda self, example, verbose=False: example.id,
    #     disable=lambda self: not AppConfig.get_instance().is_cache_enable,
    # )
    # def batch_call(
    #     self, exs: list[GPExample], verbose: bool = False
    # ) -> list[CanGraphExtractedResult]:
    #     if len(exs) == 0:
    #         return []

    #     assert all(
    #         ex.kgname == exs[0].kgname for ex in exs
    #     ), "Don't support multiple KGs in the same function call"

    #     ex_cans = self.get_candidate_entities(exs)
    #     text_parser = self.text_parser

    #     tables = []
    #     table_cells = []

    #     for ei, ex in tqdm(enumerate(exs), disable=not verbose):
    #         nrows, ncols = ex.table.table.shape()
    #         tbl = to_rust_table(ex, ex_cans[ei])
    #         cells = TableCells(
    #             [
    #                 [
    #                     text_parser.parse(ex.table.table[ri, ci]).to_rust()
    #                     for ci in range(ncols)
    #                 ]
    #                 for ri in range(nrows)
    #             ]
    #         )

    #         tables.append(tbl)
    #         table_cells.append(cells)

    #     return par_extract_cangraphs(
    #         tables,
    #         table_cells,
    #         self.db_actor.kgdbs[exs[0].kgname].rudb,
    #         self.ru_cangraph_extractor,
    #         None,
    #         verbose=verbose,
    #     )

    # def get_candidate_entities(self, exs: list[GPExample]):
    #     ex_cans = self.canrank_actor.batch_call(exs)
    #     if self.params.topk is not None:
    #         return [cans.top_k(self.params.topk) for cans in ex_cans]
    #     return ex_cans
