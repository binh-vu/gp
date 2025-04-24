from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from ream.data_model_helper import (
    EncodedSingleMasked2DNumpyArray,
    EncodedSingleNumpyArray,
)
from sm.misc.funcs import get_decoder
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map


class ColumnarDataset(Dataset):
    """A columnar dataset"""

    def __init__(
        self,
        columns: dict[str, np.ndarray | list | Feat],
        dtypes: Optional[dict[str, Any]] = None,
        collate_fn: Optional[Callable] = None,
        references: Optional[dict] = None,
        name: str = "",
    ):
        self.name = name
        self.collate_fn = collate_fn
        self.columns = columns
        self.size = len(next(iter(self.columns.values())))
        self.dtypes = dtypes
        self.references = references or {}

        if dtypes is not None:
            for name, feat in self.columns.items():
                if name in dtypes and isinstance(feat, np.ndarray):
                    self.columns[name] = feat.astype(dtypes[name])
                    dtypes.pop(name)

    def update_size(self):
        self.size = len(next(iter(self.columns.values())))

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int | slice):
        obj = {name: feat[idx] for name, feat in self.columns.items()}
        if isinstance(idx, slice):
            return ColumnarDataset(
                obj,
                dtypes=self.dtypes,
                collate_fn=self.collate_fn,
                references=self.references,
                name=self.name,
            )
        return obj

    def to_df(self):
        cols = {}
        for name, feat in self.columns.items():
            if not isinstance(feat, np.ndarray):
                raise ValueError(
                    f"Column {name} is not a numpy array, cannot convert dataset to dataframe"
                )
            if len(feat.shape) > 2:
                raise ValueError(
                    f"Column {name} has more than 2 dimensions, cannot convert dataset to dataframe"
                )
            if len(feat.shape) == 2:
                for i in range(feat.shape[1]):
                    cols[f"{name}_{i}"] = feat[:, i]
            else:
                cols[name] = feat
        return pd.DataFrame(cols)


@dataclass
class Feat:
    def __getitem__(self, idx: int | slice):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


@dataclass
class EmbeddingFeat(Feat):
    """For nd-array, this code works for 3d array as well.

    Example:
        >>> a = np.arange(16).reshape((4,4))
        >>> idx = [[0, 1, 3], [1, 3, 2]]
        >>> a[idx]

    """

    value: np.ndarray | list
    embs: np.ndarray
    decoder: Optional[list[str]] = None

    def __getitem__(self, idx: int | slice):
        if not isinstance(idx, int):
            return EmbeddingFeat(self.value[idx], self.embs, self.decoder)
        return self.embs[self.value[idx]]

    def __len__(self):
        return len(self.value)

    @staticmethod
    def from_encoded_single_numpy_array(
        array: EncodedSingleNumpyArray | EncodedSingleMasked2DNumpyArray,
        emb_fn: Callable[[list[str]], np.ndarray],
    ):
        return EmbeddingFeat(array.value, emb_fn(array.decoder), array.decoder)

    def to_array(self):
        return self.embs[self.value]

    def merge(self, other: EmbeddingFeat):
        # haven't handle non-decoder yet
        assert self.decoder is not None and other.decoder is not None
        self_encoder = {v: i for i, v in enumerate(self.decoder)}

        other_valuemap = [10000000 for _ in other.decoder]
        new_embs = []

        for i, v in enumerate(other.decoder):
            if v not in self_encoder:
                self_encoder[v] = len(self_encoder)
                new_embs.append(other.embs[i])
            other_valuemap[i] = self_encoder[v]

        value = np.concatenate(
            [self.value, [other_valuemap[v] for v in other.value]], axis=0
        )
        if len(new_embs) > 0:
            embs = np.concatenate([self.embs, np.stack(new_embs, axis=0)], axis=0)
        else:
            embs = self.embs
        return EmbeddingFeat(value, embs, get_decoder(self_encoder))


@dataclass
class DynSize:
    value: Any

    @staticmethod
    def collate_fn(batch, *, collate_fn_map=None):
        return torch.stack([torch.as_tensor(b.value) for b in batch])


@dataclass
class SmallDynSize:
    value: Any

    @staticmethod
    def collate_fn(batch, *, collate_fn_map=None):
        size = max(len(b.value) for b in batch)
        return [
            torch.as_tensor(np.pad(b.value, [(0, size - b.value.shape[0]), (0, 0)]))
            for b in batch
        ]


collate_fn_map = default_collate_fn_map.copy()
collate_fn_map[DynSize] = DynSize.collate_fn
collate_fn_map[SmallDynSize] = SmallDynSize.collate_fn


def extended_collate_fn(batch):
    return collate(batch, collate_fn_map=collate_fn_map)
