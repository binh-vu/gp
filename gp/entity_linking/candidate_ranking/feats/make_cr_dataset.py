from __future__ import annotations

from functools import lru_cache
from typing import Callable, TypeVar

import numpy as np
from gp.actors.data.prelude import GPExample
from gp.actors.el.cangen import CanGenActor
from gp.entity_linking.candidate_ranking.feats.feat_fn import CRFeatFn, CRFeatFnArgs
from gp.entity_linking.candidate_ranking.feats.get_candidates import CRTableCan, GetCandidatesFn
from gp.entity_linking.candidate_ranking.feats.get_entity_desc_embedding import (
    GetEmptyEntityDescEmbeddingFn,
)
from gp.entity_linking.candidate_ranking.feats.get_freq_match import GetFreqMatchDataFn
from gp.entity_linking.candidate_ranking.feats.get_header_embedding import (
    GetEmptyHeaderEmbeddingFn,
)
from gp.entity_linking.candidate_ranking.feats.get_model_free_features import GetModelFreeFeatFn
from gp.entity_linking.populate import KGDBBatchService, KGDBService
from gp.misc.dataset import ColumnarDataset, extended_collate_fn
from ream.cache_helper import Cache, MemBackend
from ream.params_helper import NoParams
from ream.prelude import BaseActor
from sm.misc.funcs import batch
from sm.namespaces.utils import KGName
from tqdm import tqdm

Fn = TypeVar("Fn", bound=CRFeatFn)


class CRDatasetBuilder(BaseActor[NoParams]):
    VERSION = 100

    def __init__(
        self,
        cangen_actor: CanGenActor,
    ):
        super().__init__(NoParams(), [cangen_actor])

        self.cangen_actor = cangen_actor

    def __call__(
        self,
        args: CRFeatFnArgs,
        exs: list[GPExample],
        enable_fns: list[type[CRFeatFn]],
        verbose: bool = False,
    ):
        with CRFeatFn.auto_clear_mem_backend():
            funcs: list[tuple[str, CRFeatFn]] = []
            for enable_fn in enable_fns:
                funcs.append(
                    (
                        self.get_func_name(enable_fn),
                        self.get_func(enable_fn).set_args(args),
                    )
                )

            output = {fname: [] for fname, func in funcs}
            for batch_exs in tqdm(batch(64, exs), disable=not verbose):
                for fname, func in funcs:
                    output[fname].extend(func.batch_call(batch_exs))

            return output

    @lru_cache(maxsize=None)
    def get_func(self, fncls: type[Fn]) -> Fn:
        return fncls(self)

    @lru_cache(maxsize=None)
    def get_func_name(self, fncls: type[Fn]) -> str:
        return fncls.__name__

    @lru_cache(maxsize=1)
    def kgdb_service(self, kgname: KGName) -> KGDBService:
        return KGDBService(self.cangen_actor.db_actor.kgdbs[kgname])

    @lru_cache(maxsize=1)
    def kgdb_batch_service(self, kgname: KGName) -> KGDBBatchService:
        return KGDBBatchService(self.cangen_actor.db_actor.kgdbs[kgname].args)


def make_pairwise_v3_nocontext_dataset(
    store: CRDatasetBuilder,
    exs: list[GPExample],
    verbose: bool = False,
):
    out = store(
        CRFeatFnArgs(text_embedding_model="sentence-transformers/all-mpnet-base-v2"),
        exs,
        [
            GetCandidatesFn,
            GetModelFreeFeatFn,
            GetFreqMatchDataFn,
            GetEmptyHeaderEmbeddingFn,
            GetEmptyEntityDescEmbeddingFn,
        ],
        verbose=verbose,
    )

    lst_cans: list[CRTableCan] = out[store.get_func_name(GetCandidatesFn)]
    size = sum(x.cell.shape[0] for x in lst_cans)

    example_ranges = []
    cell_id = np.zeros((size,), dtype=np.int32)
    cell_id_counter = 0
    offset = 0
    for ei, ex in enumerate(exs):
        nrows, ncols = ex.table.table.shape()
        cans = lst_cans[ei]

        cell_index = cell_id_counter + cans.row_index * ncols + cans.col_index
        cell_id[offset : offset + cell_index.shape[0]] = cell_index
        example_ranges.append((offset, offset + cell_index.shape[0]))

        cell_id_counter += nrows * ncols
        offset += cell_index.shape[0]

    return ColumnarDataset(
        {
            "cell_id": cell_id,
            "cell_label": np.concatenate([x.cell for x in lst_cans]),
            "label": np.concatenate([x.is_correct for x in lst_cans]),
            "features": np.concatenate(
                [
                    np.concatenate(
                        [x.value for x in out[store.get_func_name(GetModelFreeFeatFn)]]
                    ),
                    np.concatenate(
                        [x.value for x in out[store.get_func_name(GetFreqMatchDataFn)]]
                    ).reshape(-1, 1),
                ],
                axis=1,
                dtype=np.float32,
            ),
            "header_embedding": np.concatenate(
                out[store.get_func_name(GetEmptyHeaderEmbeddingFn)]
            ),
            "entity_desc_embedding": np.concatenate(
                out[store.get_func_name(GetEmptyEntityDescEmbeddingFn)]
            ),
        },
        collate_fn=extended_collate_fn,
        references={
            "example_ranges": example_ranges,
        },
    )
