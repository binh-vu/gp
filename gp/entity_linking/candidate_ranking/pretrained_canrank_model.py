from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, Protocol, Sequence, TypeVar

import torch
from gp.actors.data import GPExample
from gp.actors.el.cangen import CanGenActor
from gp.entity_linking.candidate_ranking.common import TableCanGenUpdateScores
from gp.entity_linking.candidate_ranking.feats.make_cr_dataset import CRDatasetBuilder
from sm.misc.funcs import import_attr


@dataclass
class PretrainedCanRankModelArgs:
    model_class: str
    model_file: Path | str


class PretrainedCanRankModel:
    VERSION = 100

    def __init__(self, params: PretrainedCanRankModelArgs, cangen_actor: CanGenActor):
        self.params = params
        self.cangen_actor = cangen_actor
        self.dataset_builder = CRDatasetBuilder(cangen_actor)

    def batch_rank_candidates(
        self, exs: Sequence[GPExample], verbose: bool = False
    ) -> list[TableCanGenUpdateScores]:
        return self.method.rank_dataset(self.dataset_builder, exs, verbose)

    @cached_property
    def method(self):
        # load the model
        self.model_class: type[IPretrainedModel] = import_attr(self.params.model_class)
        if torch.cuda.is_available():
            map_location = None
        elif torch.backends.mps.is_available():
            map_location = "mps"
        else:
            map_location = "cpu"
        return self.model_class.load_from_checkpoint(
            self.params.model_file, map_location=map_location
        )


C = TypeVar("C")
A = TypeVar("A")


class IPretrainedModel(Protocol):
    @classmethod
    def load_from_checkpoint(
        cls: type[C], checkpoint: Path | str, map_location: Optional[str] = None
    ) -> C: ...

    def rank_dataset(
        self, store: CRDatasetBuilder, exs: Sequence[GPExample], verbose: bool = False
    ) -> list[TableCanGenUpdateScores]: ...
