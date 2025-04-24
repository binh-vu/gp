from __future__ import annotations

from collections import defaultdict
from operator import itemgetter
from typing import Sequence

import sm.misc as M
from graph.retworkx.api import digraph_all_simple_paths
from sm.dataset import FullTable
from sm.misc.funcs import assert_not_null

from gramsplus.semanticmodeling.postprocessing.cgraph import CGraph
from gramsplus.semanticmodeling.postprocessing.interface import (
    EdgeProb,
    NodeProb,
    PostprocessingFn,
    PostprocessingResult,
)


class GreedyKnownTargetsFn(PostprocessingFn):
    """Post-proessing the select the binary relationships and types between given columns"""

    def __init__(
        self,
        cg: CGraph,
        edge_targets: Sequence[tuple[int, int]],
        type_targets: Sequence[int],
        cpa_threshold: float,
        cta_threshold: float,
    ):
        self.cg = cg
        self.edge_targets = edge_targets
        self.type_targets = set(type_targets)
        self.cpa_threshold = cpa_threshold
        self.cta_threshold = cta_threshold

    def __call__(
        self,
        node_probs: NodeProb,
        edge_probs: EdgeProb,
    ) -> PostprocessingResult:
        nodes = {}
        for colindex, utypes in node_probs.items():
            if colindex in self.type_targets:
                ctype, cscore = max(utypes.items(), key=itemgetter(1))
                if cscore >= self.cta_threshold:
                    nodes[colindex] = ctype

        column2node = {
            assert_not_null(u.column_index): u.id
            for u in self.cg.iter_nodes()
            if u.is_column_node
        }
        selected_edges = []

        for sci, tci in self.edge_targets:
            uid = column2node[sci]
            vid = column2node[tci]

            if not self.cg.has_node(uid) or not self.cg.has_node(vid):
                continue
            paths = digraph_all_simple_paths(self.cg, uid, vid, cutoff=2)
            paths = [
                (inedge, outedge)
                for inedge, outedge in paths
                if inedge.key == outedge.key
                and edge_probs[inedge.source, inedge.target, inedge.key]
                >= self.cpa_threshold
                and edge_probs[outedge.source, outedge.target, outedge.key]
                >= self.cta_threshold
            ]
            if len(paths) == 0:
                continue

            best_path = max(
                paths,
                key=lambda p: sum(edge_probs[e.source, e.target, e.key] for e in p),
            )
            for edge in best_path:
                selected_edges.append((edge.source, edge.target, edge.key))

        return PostprocessingResult(selected_edges, nodes)
