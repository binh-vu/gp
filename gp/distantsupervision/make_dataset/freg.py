from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from gp.distantsupervision.make_dataset.utils import FilterMixin, StrPath
from sm.dataset import FullTable

# fmt: off
DEFAULT_FILTER_PATTERNS = [
    r"\d+",      # number year 1920, etc.
    r"\d+-\d+",  # number range 1920-1925 or year-month 1920-01, etc.
    r"\[\d+\]",  # references [1], [2], etc.
]
# fmt: on


@dataclass
class FilterRegexArgs:
    noisy_threshold: float = 0.5
    patterns: Sequence[str] = field(default_factory=lambda: DEFAULT_FILTER_PATTERNS)


class FilterRegex(FilterMixin):
    VERSION = 101

    def __init__(self, args: FilterRegexArgs, log_file: Optional[StrPath] = None):
        super().__init__(log_file)
        self.patterns = args.patterns
        self.regexes = [re.compile(p) for p in self.patterns]

    def is_noisy(self, table: FullTable, ci: int) -> tuple[bool, str]:
        for cell in table.table.get_column_by_index(ci).values:
            for regex in self.regexes:
                if regex.match(cell):
                    return True, f"cell {cell} matches {regex.pattern}"
        return False, ""
