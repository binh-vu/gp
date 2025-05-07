from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from gp.distantsupervision.make_dataset.fbhct import FilterByHeaderColTypeArgs
from gp.distantsupervision.make_dataset.fbt import FilterByEntTypeArgs
from gp.distantsupervision.make_dataset.freg import FilterRegexArgs
from gp.distantsupervision.make_dataset.utils import FilterMixin, StrPath
from sm.dataset import FullTable


@dataclass
class CombinedFilterArgs:
    regex: FilterRegexArgs = field(
        default_factory=FilterRegexArgs,
    )
    ignore_types: FilterByEntTypeArgs = field(
        default_factory=FilterByEntTypeArgs,
    )
    header_col_type: Optional[FilterByHeaderColTypeArgs] = field(default=None)


class CombinedFilter(FilterMixin):
    VERSION = 100

    def __init__(
        self, filters: Sequence[FilterMixin], log_file: Optional[StrPath] = None
    ):
        super().__init__(log_file)
        self.filters = filters

    def is_noisy(self, table: FullTable, ci: int) -> tuple[bool, str]:
        for f in self.filters:
            is_noisy, reason = f.is_noisy(table, ci)
            if is_noisy:
                return True, reason
        return False, ""
