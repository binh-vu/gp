from __future__ import annotations

from abc import ABC, abstractmethod

from sm.dataset import FullTable
from sm.inputs.link import EntityIdWithScore


class FilterFn(ABC):
    @abstractmethod
    def filter(self, table: FullTable, columns: list[int]) -> list[int]:
        ...


class TransformFn(ABC):
    @abstractmethod
    def transform(self, table: FullTable, columns: list[int]) -> FullTable:
        ...


class LabelFn(ABC):
    @abstractmethod
    def label(
        self, table: FullTable, columns: list[int]
    ) -> list[list[EntityIdWithScore]]:
        ...


class NoFilter(FilterFn):
    def filter(self, table: FullTable, columns: list[int]) -> list[int]:
        return columns


class NoTransform(TransformFn):
    def transform(self, table: FullTable, columns: list[int]) -> FullTable:
        return table
