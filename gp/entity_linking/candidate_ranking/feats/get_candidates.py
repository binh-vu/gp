from __future__ import annotations

from copy import copy

import numpy as np
from gp.actors.data.prelude import GPExample
from gp.entity_linking.cangen.common import CanEnt, TableCanGenResult
from gp.entity_linking.cangen.oracle_model import CanGenOracleMethod
from gp.entity_linking.candidate_ranking.feats.feat_fn import CRFeatFn, CRFeatFnArgs
from nptyping import Bool, Float64, Int32, NDArray, Object, Shape
from ream.cache_helper import Cache, CacheableFn
from ream.data_model_helper._numpy_model import NumpyDataModel


class CRTableCan(NumpyDataModel):
    __slots__ = [
        "index",
        "cell",
        "col_index",
        "row_index",
        "ent_id",
        "ent_retrieve_score",
        "is_correct",
    ]
    # mapping from col_index => row_index => (start, end)
    index: dict[int, dict[int, tuple[int, int]]]
    cell: NDArray[Shape["*"], Object]  # content of the cell
    col_index: NDArray[Shape["*"], Int32]  # column index of the cell
    row_index: NDArray[Shape["*"], Int32]  # row index of the cell
    ent_id: NDArray[Shape["*"], Object]
    ent_retrieve_score: NDArray[Shape["*"], Float64]  # retrieval score of the entity
    is_correct: NDArray[Shape["*"], Bool]  # whether the cell is correct

    def to_table_cangen_result(self) -> TableCanGenResult:
        return TableCanGenResult(
            index=self.index,
            ent_id=self.ent_id,
            ent_score=self.ent_retrieve_score,
        )


CRTableCan.init()


class GetCandidatesFn(CRFeatFn):
    use_args = [CRFeatFnArgs.add_missing_gold, CRFeatFnArgs.remove_nil_entity]

    @Cache.cache(
        backend=Cache.sqlite.serde(
            cls=CRTableCan,
            filename=lambda slf, ex: slf.cache_file + "/data.sqlite",
            mem_persist=True,
        ),
        cache_key=lambda self, ex: ex.id,
        disable="disable",
    )
    def __call__(self, ex: GPExample) -> CRTableCan:
        cans = self.cangen_actor(ex)
        gold_cans = CanGenOracleMethod().get_candidates(
            [ex], [list(range(ex.table.ncols()))]
        )[0]

        if self.args.add_missing_gold:
            # add missing gold entities
            shp = ex.table.table.shape()
            mcan = cans.to_matrix(shp)
            mgoldcan = gold_cans.to_matrix(shp)

            new_mcan = mcan.map_with_index(
                lambda ri, ci, cans: GetCandidatesFn.merge_gold_cans(
                    cans, mgoldcan[ri][ci], self.args.remove_nil_entity
                )
            )
            cans = TableCanGenResult.from_matrix(new_mcan)
        else:
            assert (
                not self.args.remove_nil_entity
            ), "remove_nil_entity is only valid when add_missing_gold is True"

        cell = []
        col_index = []
        row_index = []
        is_correct = []

        for ci in cans.index:
            for ri, (start, end) in cans.index[ci].items():
                text = ex.table.table[ri, ci]

                cell.extend((text for _ in range(start, end)))
                col_index.extend((ci for _ in range(start, end)))
                row_index.extend((ri for _ in range(start, end)))

                gold_ids = {x.id for x in gold_cans.get_cell_candidates(ri, ci)}
                for j in range(start, end):
                    is_correct.append(cans.ent_id[j] in gold_ids)

        return CRTableCan(
            index=cans.index,
            cell=np.asarray(cell, dtype=np.object_),
            col_index=np.asarray(col_index, dtype=np.int64),
            row_index=np.asarray(row_index, dtype=np.int64),
            ent_id=cans.ent_id,
            ent_retrieve_score=cans.ent_score,
            is_correct=np.asarray(is_correct, dtype=np.bool_),
        )

    @staticmethod
    def merge_gold_cans(
        cans: list[CanEnt], gold_cans: list[CanEnt], remove_nil_entity: bool
    ):
        # remove candidates with NIL entity -- i.e., cells without gold entity
        if len(gold_cans) == 0 and remove_nil_entity:
            return []

        can_ids = {can.id for can in cans}
        gold_cans = [can for can in gold_cans if can.id not in can_ids]

        if len(gold_cans) > 0:
            # although `to_matrix` function create a new copy, this function still
            # copy because we cannot guarantee that usages in which the cans is already a temp copy
            cans = copy(cans)

            for gold_can in gold_cans:
                cans.append(gold_can)

        return cans
