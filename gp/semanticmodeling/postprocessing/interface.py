from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias

from gp.semanticmodeling.postprocessing.cgraph import CGEdgeTriple


@dataclass
class PostprocessingResult:
    edges: Sequence[CGEdgeTriple]
    nodes: Mapping[int, str]


EdgeProb: TypeAlias = dict[CGEdgeTriple, float]
# mapping from column to each type and its probability
NodeProb: TypeAlias = dict[int, dict[str, float]]


class PostprocessingFn:
    @abstractmethod
    def __call__(
        self,
        node_probs: NodeProb,
        edge_probs: EdgeProb,
    ) -> PostprocessingResult: ...
