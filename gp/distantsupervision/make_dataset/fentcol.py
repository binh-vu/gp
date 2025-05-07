from __future__ import annotations

from typing import Optional

from gp.distantsupervision.make_dataset.utils import FilterMixin, StrPath
from gp.entity_linking.candidate_recognition._heuristic_model import HeuristicCanReg
from kgdata.models import Ontology
from sm.dataset import Example, FullTable
from sm.misc.fn_cache import CacheMethod


class FilterNotEntCol(FilterMixin):
    VERSION = 100

    def __init__(self, log_file: Optional[StrPath] = None):
        super().__init__(log_file)
        self.canreg = HeuristicCanReg()

    def is_noisy(self, table: FullTable, ci: int) -> tuple[bool, str]:
        if ci not in self.reg_ent_cols(table):
            return True, "The column is not recognized as entity column"
        return False, ""

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def reg_ent_cols(self, table: FullTable) -> set[int]:
        # set the ontology to None because HeuristicCanReg does not use it
        # and we don't want to pass it to the function
        ontology: Ontology = None  # type: ignore
        return set(
            self.canreg(
                [Example(id=table.table.table_id, sms=[], table=table)], ontology
            )[0]
        )
