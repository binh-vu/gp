from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

from kgdata.models import Ontology
from sm.dataset import Example, FullTable


class ICanReg:
    @abstractmethod
    def __call__(
        self, examples: Sequence[Example[FullTable]], ontology: Ontology
    ) -> list[list[int]]:
        pass
