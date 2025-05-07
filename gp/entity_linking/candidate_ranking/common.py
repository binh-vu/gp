from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, Sequence, TypeVar

import lightning.pytorch as pl
import torch
from gp.actors.data import GPExample
from gp.entity_linking.candidate_ranking.feats.make_cr_dataset import CRDatasetBuilder
from gp.misc.dataset import ColumnarDataset
from smml.data_model_helper import SingleNumpyArray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

TableCanGenUpdateScores = SingleNumpyArray


class CanRankMethod:

    @abstractmethod
    def batch_rank_candidates(
        self,
        examples: Sequence[GPExample],
        verbose: bool = False,
    ) -> list[TableCanGenUpdateScores]:
        pass


C = TypeVar("C")
A = TypeVar("A")


@dataclass
class BaseTorchCanRankOutput:
    loss: Optional[torch.Tensor]
    probs: torch.Tensor


class BaseTorchCanRankModel(ABC, Generic[A], pl.LightningModule):
    EVAL_BATCH_SIZE = 100
    EXPECTED_EVAL_ARGS = set()

    @classmethod
    @abstractmethod
    def from_args(cls: type[C], args: A) -> C: ...

    @abstractmethod
    def make_dataset(
        self,
        store: CRDatasetBuilder,
        exs: Sequence[GPExample],
        verbose: bool = False,
    ) -> ColumnarDataset: ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> BaseTorchCanRankOutput: ...

    def rank_dataset(
        self,
        store: CRDatasetBuilder,
        exs: Sequence[GPExample],
        verbose: bool = False,
    ) -> list[TableCanGenUpdateScores]:
        self.eval()

        params = next(self.parameters())
        device = params.device

        dataset = self.make_dataset(store, exs, verbose)
        dloader = DataLoader(
            dataset,
            batch_size=self.EVAL_BATCH_SIZE,
            shuffle=False,
            pin_memory=params.is_cuda,
            collate_fn=dataset.collate_fn,
        )

        with torch.no_grad():
            probs = []
            for batch in tqdm(
                dloader,
                total=len(dloader),
                desc="ranking candidates in a dataset",
                disable=not verbose,
            ):
                kwargs = {}
                for arg in self.EXPECTED_EVAL_ARGS:
                    kwargs[arg] = batch[arg].to(device)
                output = self.forward(**kwargs)
                probs.append(output.probs.cpu())

            probs = torch.cat(probs).numpy()

        example_ranges = dataset.references["example_ranges"]
        return [
            TableCanGenUpdateScores(
                probs[example_ranges[ei][0] : example_ranges[ei][1]]
            )
            for ei, ex in enumerate(exs)
        ]
