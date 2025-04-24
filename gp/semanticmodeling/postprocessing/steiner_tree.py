from __future__ import annotations

from collections import defaultdict
from copy import copy
from operator import itemgetter
from typing import Literal, Optional, cast

from gramsplus.semanticmodeling.postprocessing.cgraph import (
    CGEdge,
    CGEdgeTriple,
    CGNode,
    CGraph,
)
from gramsplus.semanticmodeling.postprocessing.common import (
    add_context,
    ensure_valid_statements,
)
from gramsplus.semanticmodeling.postprocessing.config import PostprocessingConfig
from gramsplus.semanticmodeling.postprocessing.interface import (
    EdgeProb,
    NodeProb,
    PostprocessingFn,
    PostprocessingResult,
)
from graph.retworkx.api import dag_longest_path, has_cycle
from sm.dataset import FullTable
from steiner_tree.bank.solver import PSEUDO_ROOT_ID, BankSolver
from steiner_tree.bank.struct import (
    BankEdge,
    BankGraph,
    BankNode,
    NoSingleRootException,
    Solution,
    UpwardTraversal,
)


class SteinerTreeFn(PostprocessingFn):
    def __init__(
        self,
        table: FullTable,
        cg: CGraph,
        cpa_threshold: float,
        cta_threshold: float,
        cta_score_offset: float,
    ):
        self.table = table
        self.cpa_threshold = cpa_threshold
        self.cta_threshold = cta_threshold + cta_score_offset
        self.cg = cg

    def __call__(
        self, node_probs: NodeProb, edge_probs: EdgeProb
    ) -> PostprocessingResult:
        st = SteinerTree(
            self.table,
            self.cg,
            edge_probs,
            self.cpa_threshold,
            additional_terminal_nodes=None,
        )
        pred_cg = st.get_result()

        edges = [(e.source, e.target, e.key) for e in pred_cg.iter_edges()]
        nodes = {}
        for colindex, utypes in node_probs.items():
            ctype, cscore = max(utypes.items(), key=itemgetter(1))
            if cscore >= self.cta_threshold:
                nodes[colindex] = ctype
        return PostprocessingResult(edges=edges, nodes=nodes)


class SteinerTree:
    def __init__(
        self,
        table: FullTable,
        cg: CGraph,
        edge_probs: dict[CGEdgeTriple, float],
        threshold: float,
        additional_terminal_nodes: Optional[list[str]] = None,
    ):
        self.table = table
        self.cg = cg
        self.edge_probs = edge_probs
        self.threshold = threshold

        self.top_k_st = 50
        self.top_k_path = 50

        # extra terminal nodes that the tree should have, usually used in
        # interactive modeling where users add some entity nodes in their model
        self.additional_terminal_nodes = additional_terminal_nodes

    def get_result(self) -> CGraph:
        """Select edges that forms a tree"""
        edge_probs = {e: p for e, p in self.edge_probs.items() if p >= self.threshold}

        # first step is to remove dangling statements and standalone nodes
        subcg = self.cg.subgraph_from_edge_triples(edge_probs.keys())
        subcg.remove_dangling_statement()
        subcg.remove_standalone_nodes()

        # terminal nodes are columns node, entity nodes are adding later
        terminal_nodes = {u.id for u in subcg.iter_nodes() if u.is_column_node}
        if self.additional_terminal_nodes is not None:
            terminal_nodes.update(self.additional_terminal_nodes)
        if len(terminal_nodes) == 0:
            # if the tree does not have any column, we essentially don't predict anything
            # so we return an empty graph
            return CGraph()

        # add pseudo root
        edge_weights = {e: 1.0 / p for e, p in edge_probs.items()}  # p > 0.5

        solver = BankSteinerTree(
            original_graph=subcg,
            terminal_nodes=terminal_nodes,
            top_k_st=self.top_k_st,
            top_k_path=self.top_k_path,
            weight_fn=lambda e: edge_weights[e.source, e.target, e.key],
            solution_cmp_fn=self.compare_solutions,
            invalid_roots={u.id for u in subcg.nodes() if u.is_statement_node},
            allow_shorten_graph=False,
        )
        trees, _solutions = solver.run()
        if len(trees) == 0:
            return CGraph()

        tree = cast(CGraph, trees[0])
        tree.remove_dangling_statement()

        if PostprocessingConfig.INCLUDE_CONTEXT:
            add_context(subcg, tree, edge_probs)

        # add back statement property if missing into to ensure a correct model
        ensure_valid_statements(subcg, tree, create_if_not_exists=False)

        # fmt: off
        # from graph.viz.graphviz import draw
        # draw(graph=tree, filename="/tmp/graphviz/st204.png", **CGGraph.graphviz_props())
        # draw(graph=tree, filename="/tmp/graphviz/g25.png", **CGGraph.graphviz_props())
        # fmt: on

        return tree

    def compare_solutions(self, a: Solution, b: Solution) -> int:
        """Comparing two solutions, -1 (smaller) means a better solution -- we are solving minimum steiner tree"""
        a_weight = a.weight / max(a.num_edges, 1)
        b_weight = b.weight / max(b.num_edges, 1)

        if a_weight < b_weight:
            return -1
        if a_weight > b_weight:
            return 1
        # equal weight, prefer the one with shorter depth
        # TODO: there is a case where a statement has a qualifier
        # it makes the tree without source node shorter than the one with source node
        # however, the one with the source node is better...
        # so the steiner tree should have an option to say like if we pick this path, we must
        # select the source node to avoid these issue...

        # for a dirty fix, perhaps we can say, the one with a dangling statement is infinitly bad
        if not hasattr(a, "has_no_source_statement"):
            has_no_source_statement = False
            for u in a.graph.iter_nodes():
                if u.id == PSEUDO_ROOT_ID:
                    continue
                cgu = self.cg.get_node(u.id)
                if cgu.is_statement_node and a.graph.in_degree(u.id) == 0:
                    has_no_source_statement = True
                    break
            setattr(a, "has_no_source_statement", has_no_source_statement)

        if not hasattr(b, "has_no_source_statement"):
            has_no_source_statement = False
            for u in b.graph.iter_nodes():
                if u.id == PSEUDO_ROOT_ID:
                    continue
                cgu = self.cg.get_node(u.id)
                if cgu.is_statement_node and b.graph.in_degree(u.id) == 0:
                    has_no_source_statement = True
                    break
            setattr(b, "has_no_source_statement", has_no_source_statement)

        a_has_no_source = getattr(a, "has_no_source_statement")
        b_has_no_source = getattr(b, "has_no_source_statement")
        if not a_has_no_source and b_has_no_source:
            return -1
        if a_has_no_source and not b_has_no_source:
            return 1

        if not hasattr(a, "depth"):
            setattr(a, "depth", len(dag_longest_path(a.graph)))
        if not hasattr(b, "depth"):
            setattr(b, "depth", len(dag_longest_path(b.graph)))
        return getattr(a, "depth") - getattr(b, "depth")


class BankSteinerTree(BankSolver[CGNode, CGEdge]):
    ADD_MISSING_STATEMENT_PROPS = True

    def _solve(
        self, g: BankGraph, terminal_nodes: set[str], top_k_st: int, top_k_path: int
    ):
        """Override the main algorithm to handle statement"""
        roots = {u.id for u in g.iter_nodes() if u.id not in self.invalid_roots}

        attr_visit_hists: list[tuple[str, UpwardTraversal]] = []
        # to ensure the order
        for uid in list(sorted(terminal_nodes)):
            visit_hist = UpwardTraversal.top_k_beamsearch(g, uid, top_k_path)
            roots = roots.intersection(visit_hist.paths.keys())
            attr_visit_hists.append((uid, visit_hist))

        if len(roots) == 0:
            raise NoSingleRootException()

        # to ensure the order again & remove randomness
        roots = sorted(roots)

        # merge the paths using beam search
        results = []
        for root in roots:
            current_states = []
            uid, visit_hist = attr_visit_hists[0]
            for path in visit_hist.paths[root]:
                pg = BankGraph()
                if len(path.path) > 0:
                    assert uid == path.path[0].target
                pg.add_node(BankNode(uid))
                for e in path.path:
                    pg.add_node(BankNode(e.source))
                    pg.add_edge(e.clone())

                self.add_missing_statement(pg)
                current_states.append(pg)

            if len(current_states) > top_k_st:
                current_states = [
                    _s.graph for _s in self._sort_solutions(current_states)[:top_k_st]
                ]

            for uid, visit_hist in attr_visit_hists[1:]:
                next_states = []
                for state in current_states:
                    for path in visit_hist.paths[root]:
                        pg = state.copy()
                        if len(path.path) > 0:
                            assert uid == path.path[0].target
                        if not pg.has_node(uid):
                            pg.add_node(BankNode(uid))
                        for e in path.path:
                            if not pg.has_node(e.source):
                                pg.add_node(BankNode(id=e.source))
                            # TODO: here we don't check by edge_key because we may create another edge of different key
                            # hope this new path has been exploited before.
                            if not pg.has_edges_between_nodes(e.source, e.target):
                                pg.add_edge(e.clone())
                        self.add_missing_statement(pg)

                        # if there are more than path between two nodes within
                        # two hop, we'll select one
                        update_graph = False
                        for n in pg.iter_nodes():
                            if pg.in_degree(n.id) >= 2:
                                grand_parents: dict[
                                    str, list[tuple[BankEdge, ...]]
                                ] = defaultdict(list)
                                for inedge in pg.in_edges(n.id):
                                    grand_parents[inedge.source].append((inedge,))
                                    for grand_inedge in pg.in_edges(inedge.source):
                                        grand_parents[grand_inedge.source].append(
                                            (grand_inedge, inedge)
                                        )

                                for grand_parent, edges in grand_parents.items():
                                    if len(edges) > 1:
                                        # we need to select one path from this grand parent to the rest
                                        # they have the same length, so we select the one has smaller weight
                                        edges = sorted(
                                            edges,
                                            key=lambda x: x[0].weight + x[1].weight
                                            if len(x) == 2
                                            else x[0].weight * 2,
                                        )

                                        for lst in edges[1:]:
                                            for edge in lst:
                                                # TODO: handle removing edges multiple times
                                                try:
                                                    pg.remove_edge(edge.id)
                                                except:
                                                    continue
                                        update_graph = True
                        if update_graph:
                            for n in pg.nodes():
                                if pg.in_degree(n.id) == 0 and pg.out_degree(n.id) == 0:
                                    pg.remove_node(n.id)
                        # after add a path to the graph, it can create new cycle, detect and fix it
                        if has_cycle(pg):
                            # we can show that if the graph contain cycle, there is a better path
                            # so no need to try to break cycles as below
                            # cycles_iter = [(uid, vid) for uid, vid, eid, orien in cycles_iter]
                            # for _g in self._break_cycles(root, pg, cycles_iter):
                            #     next_states.append(_g)
                            pass
                        else:
                            next_states.append(pg)

                        # the output graph should not have parallel edges
                        assert not pg.has_parallel_edges()
                if len(next_states) > top_k_st:
                    next_states = [
                        _s.graph for _s in self._sort_solutions(next_states)[:top_k_st]
                    ]
                # assert all(_.check_integrity() for _ in next_states)
                current_states = next_states
                # cgs = [g for g in next_states if len(list(nx.simple_cycles(g))) > 0]
                # nx.draw_networkx(cgs[0]); plt.show()
                # nx.draw(cgs[0]); plt.show()
            results += current_states

        return self._sort_solutions(results)

    def add_missing_statement(self, g: BankGraph):
        if not self.ADD_MISSING_STATEMENT_PROPS:
            return

        cg = cast(CGraph, self.original_graph)
        for uprime in g.iter_nodes():
            if uprime.id == PSEUDO_ROOT_ID:
                continue
            u = cg.get_node(uprime.id)
            if u.is_statement_node:
                (inedge,) = cg.in_edges(u.id)
                if all(outedge.key != inedge.key for outedge in g.out_edges(u.id)):
                    # no statement property, add it back
                    # sometimes the baseline don't have main statement
                    tmp = [
                        outedge
                        for outedge in self.graph.out_edges(u.id)
                        if outedge.key == inedge.key
                    ]
                    if len(tmp) == 0:
                        continue
                    (main_prop,) = tmp
                    if not g.has_node(main_prop.target):
                        target_id = g.add_node(
                            copy(self.graph.get_node(main_prop.target))
                        )
                        assert target_id == main_prop.target
                    g.add_edge(main_prop.clone())
