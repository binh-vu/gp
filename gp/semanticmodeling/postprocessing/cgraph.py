from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, TypeAlias

from gp_core.algorithms import CanGraphExtractedResult
from graph.retworkx import BaseEdge, BaseNode, RetworkXStrDiGraph
from sm.inputs.table import ColumnBasedTable
from sm.misc.funcs import assert_not_null
from sm.namespaces.namespace import KnowledgeGraphNamespace
from sm.outputs.semantic_model import LiteralNodeDataType
from sm.prelude import O


@dataclass
class CGNode(BaseNode[str]):
    id: str  # it is number but converted into a string

    is_statement_node: bool
    is_column_node: bool
    is_entity_node: bool
    is_literal_node: bool
    is_in_context: bool

    column_index: Optional[int]  # not null if is_column_node=True

    def to_sm_node(
        self,
        table: ColumnBasedTable,
        kgns: KnowledgeGraphNamespace,
        id2literal: Callable[[str], str],
    ):
        """Convert a node in the candidate graph to a node in the semantic model"""
        if self.is_statement_node:
            return O.ClassNode(
                abs_uri=kgns.statement_uri,
                rel_uri=kgns.get_rel_uri(kgns.statement_uri),
                approximation=False,
                readable_label="Statement",
            )
        if self.is_column_node:
            assert self.column_index is not None
            return O.DataNode(
                col_index=self.column_index,
                label=assert_not_null(
                    table.get_column_by_index(self.column_index).clean_multiline_name
                ),
            )
        if self.is_entity_node:
            entid = id2literal(self.id)
            return O.LiteralNode(
                value=kgns.uri_to_id(entid),
                datatype=LiteralNodeDataType.Entity,
            )
        if self.is_literal_node:
            litval = id2literal(self.id)
            return O.LiteralNode(
                value=litval,
                datatype=LiteralNodeDataType.String,
            )
        raise RuntimeError("Unknown node type")


CGEdgeTriple: TypeAlias = tuple[str, str, str]
CGEdge: TypeAlias = BaseEdge[str, str]


class CGraph(RetworkXStrDiGraph[str, CGNode, BaseEdge[str, str]]):
    """A temporary copied version of the candidate graph in Rust for running post-processing"""

    @staticmethod
    def from_rust(cangraph: CanGraphExtractedResult) -> CGraph:
        g = CGraph()
        for u in cangraph.nodes:
            uid = u.id()
            is_in_context = False
            is_column = u.is_column()
            is_entity = u.is_entity()
            is_literal = u.is_literal()
            column_index = None
            if is_entity:
                is_in_context = assert_not_null(u.try_as_entity()).is_in_context
            if is_literal:
                is_in_context = assert_not_null(u.try_as_literal()).is_in_context
            if is_column:
                column_index = assert_not_null(u.try_as_column()).column

            g.add_node(
                CGNode(
                    str(uid),
                    u.is_statement(),
                    is_column,
                    is_entity,
                    is_literal,
                    is_in_context,
                    column_index,
                )
            )
        for row in cangraph.edgedf.iter_rows(named=True):
            g.add_edge(
                BaseEdge(
                    -1,
                    str(row["source"]),
                    str(row["statement"]),
                    cangraph.edges[row["inedge"]],
                )
            )
            g.add_edge(
                BaseEdge(
                    -1,
                    str(row["statement"]),
                    str(row["target"]),
                    cangraph.edges[row["outedge"]],
                )
            )
        return g

    def remove_dangling_statement(self):
        """Remove statement nodes that have no incoming nodes or no outgoing edges"""
        for s in self.nodes():
            if s.is_statement_node and (
                self.in_degree(s.id) == 0 or self.out_degree(s.id) == 0
            ):
                self.remove_node(s.id)

    def remove_standalone_nodes(self):
        """Remove nodes that do not any incoming / outgoing edges"""
        for u in self.nodes():
            if self.degree(u.id) == 0:
                self.remove_node(u.id)
