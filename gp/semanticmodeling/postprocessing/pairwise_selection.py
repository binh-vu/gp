from typing import Dict

from graph.retworkx.api import digraph_all_simple_paths
from sm.dataset import FullTable

from gramsplus.semanticmodeling.postprocessing.cgraph import (
    CGEdgeTriple,
    CGraph,
)
from gramsplus.semanticmodeling.postprocessing.common import (
    add_context,
    ensure_valid_statements,
)
from gramsplus.semanticmodeling.postprocessing.config import PostprocessingConfig


class PairwiseSelection:
    def __init__(
        self,
        table: FullTable,
        cg: CGraph,
        edge_probs: Dict[CGEdgeTriple, float],
        threshold: float,
    ):
        self.table = table
        self.cg = cg
        self.edge_probs = edge_probs
        self.threshold = threshold

    def get_result(self) -> CGraph:
        """Select the highest score relationship between two nodes"""
        edge_probs = {e: p for e, p in self.edge_probs.items() if p >= self.threshold}

        # first step is to remove dangling statements and standalone nodes
        # even if we don't need this step
        subcg = self.cg.subgraph_from_edge_triples(edge_probs.keys())
        subcg.remove_dangling_statement()
        subcg.remove_standalone_nodes()

        compared_nodes = set()
        for u in subcg.nodes():
            if u.is_column_node:
                compared_nodes.add(u.id)
            elif (u.is_entity_node or u.is_literal_node) and u.is_in_context:
                compared_nodes.add(u.id)

        select_edges = set()
        for uid in compared_nodes:
            for vid in compared_nodes:
                paths = [
                    (
                        e1,
                        e2,
                        edge_probs.get((e1.source, e1.target, e1.key), 0.0)
                        + edge_probs.get((e2.source, e2.target, e2.key), 0.0),
                    )
                    for e1, e2 in digraph_all_simple_paths(subcg, uid, vid, cutoff=2)
                ]
                if len(paths) == 0:
                    continue

                e1, e2, score = max(paths, key=lambda x: x[2])
                if score == 0:
                    continue

                select_edges.add(e1.id)
                select_edges.add(e2.id)

        predcg = subcg.subgraph_from_edges(select_edges)
        predcg.remove_dangling_statement()
        predcg.remove_standalone_nodes()

        if PostprocessingConfig.INCLUDE_CONTEXT:
            add_context(subcg, predcg, edge_probs)

        # add back statement property if missing into to ensure a correct model
        ensure_valid_statements(subcg, predcg, create_if_not_exists=True)

        return predcg
