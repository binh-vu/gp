from __future__ import annotations

import gramsplus.core as gcore
import numpy as np
from gp.actors.data.prelude import GPExample
from gp.entity_linking.candidate_ranking.feats.feat_fn import CRFeatFn, CRFeatFnArgs
from gp.entity_linking.candidate_ranking.feats.get_candidates import CRTableCan, GetCandidatesFn
from gp.misc.conversions import to_rust_table
from gramsplus.semanticmodeling.text_parser import TextParser
from nptyping import Single
from ream.cache_helper import Cache
from ream.data_model_helper._numpy_model import SingleNumpyArray


class GetFreqMatchDataFn(CRFeatFn):
    use_args = [
        CRFeatFnArgs.text_parser_cfg,
        CRFeatFnArgs.literal_matcher_cfg,
    ]

    get_candidates: GetCandidatesFn

    REASONABLY_LARGE_NO_COLUMN = 20

    def set_args(self, args: CRFeatFnArgs):
        super().set_args(args)
        self.text_parser = TextParser(args.text_parser_cfg)
        self.literal_matcher = gcore.literal_matchers.LiteralMatcher(
            args.literal_matcher_cfg.to_rust()
        )
        return self

    @Cache.cache(
        backend=Cache.sqlite.serde(
            cls=SingleNumpyArray,
            filename=lambda slf, ex: slf.cache_file + "/data.sqlite",
            mem_persist=True,
        ),
        cache_key=lambda self, ex: ex.id,
        disable="disable",
    )
    def __call__(self, ex: GPExample):
        """Calculate how many incoming and outgoing relationships to other cells in a row of a candidate entity"""
        nrows, ncols = ex.table.table.shape()

        cans = self.get_candidates(ex)
        table = to_rust_table(ex, cans.to_table_cangen_result())
        table_cells = gcore.models.TableCells(
            [
                [
                    self.text_parser.parse(ex.table.table[ri, ci]).to_rust()
                    for ci in range(ncols)
                ]
                for ri in range(nrows)
            ]
        )

        table_res = gcore.algorithms.extract_candidate_entity_link_freqs(
            self.store.kgdb_service(ex.kgname).kgdb.rudb,
            table,
            table_cells,
            None,
            self.literal_matcher,
            ignored_columns=[],
            ignored_props=[],
            allow_same_ent_search=False,
            allow_ent_matching=True,
            use_context=True,
            deterministic_order=False,
            parallel=True,
        )
        freqs = GetFreqMatchDataFn.normalize_results(
            ex,
            cans,
            table_res,
        )
        return SingleNumpyArray(np.asarray(freqs) / self.REASONABLY_LARGE_NO_COLUMN)

    @Cache.flat_cache(
        backend=Cache.sqlite.serde(
            cls=SingleNumpyArray,
            filename=lambda slf, ex: slf.cache_file + "/data.sqlite",
            mem_persist=True,
        ),
        cache_key=lambda self, ex, verbose=False: ex.id,
        disable="disable",
    )
    def batch_call(self, exs: list[GPExample], verbose: bool = False):
        if len(exs) == 0:
            return SingleNumpyArray(np.array([]))

        in_tables = []
        in_table_cells = []

        ex_cans = self.get_candidates.batch_call(exs)

        for ei, ex in enumerate(exs):
            nrows, ncols = ex.table.table.shape()
            in_tables.append(to_rust_table(ex, ex_cans[ei].to_table_cangen_result()))
            in_table_cells.append(
                gcore.models.TableCells(
                    [
                        [
                            self.text_parser.parse(ex.table.table[ri, ci]).to_rust()
                            for ci in range(ncols)
                        ]
                        for ri in range(nrows)
                    ]
                )
            )

        table_ress = gcore.algorithms.par_extract_candidate_entity_link_freqs(
            self.store.kgdb_service(exs[0].kgname).kgdb.rudb,
            in_tables,
            in_table_cells,
            None,
            self.literal_matcher,
            ignored_columns=[[] for _ in range(len(in_tables))],
            ignored_props=[],
            allow_same_ent_search=False,
            allow_ent_matching=True,
            use_context=True,
            deterministic_order=False,
            verbose=verbose,
        )
        return [
            SingleNumpyArray(
                GetFreqMatchDataFn.normalize_results(
                    ex,
                    cans,
                    res,
                )
                / self.REASONABLY_LARGE_NO_COLUMN
            )
            for ex, cans, res in zip(exs, ex_cans, table_ress)
        ]

    @staticmethod
    def normalize_results(
        example: GPExample,
        crcan_base: CRTableCan,
        link_freqs: list[dict[str, list[int]]],
    ):
        nrows, ncols = example.table.table.shape()
        row_index = crcan_base.row_index
        col_index = crcan_base.col_index
        matched_freqs = []
        for i in range(len(crcan_base)):
            ri = row_index[i]
            ci = col_index[i]
            canid = crcan_base.ent_id[i]

            freq = link_freqs[ri * ncols + ci].get(canid, 0)
            matched_freqs.append(freq)
        return np.asarray(matched_freqs)
