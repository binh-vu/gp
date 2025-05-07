from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

from gp.distantsupervision.make_dataset.interface import FilterFn
from loguru import logger
from ream.helper import _logger_formatter
from sm.dataset import FullTable

StrPath = Union[str, Path]


class LoggerMixin:
    def __init__(self, log_file: Optional[StrPath] = None):
        self.logger = logger.bind(name=self.__class__.__name__)
        if log_file is not None:
            self.logger.add(log_file, format=_logger_formatter)


class FilterMixin(FilterFn, LoggerMixin):
    def filter(self, table: FullTable, columns: list[int]) -> list[int]:
        new_cols = []
        for ci in columns:
            noisy, reason = self.is_noisy(table, ci)
            if not noisy:
                new_cols.append(ci)
            else:
                self.logger.debug(
                    f"Filter column {ci} in table {table.table.table_id}: {reason}"
                )
        return new_cols

    @abstractmethod
    def is_noisy(self, table: FullTable, ci: int) -> tuple[bool, str]:
        pass
