from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from operator import itemgetter
from typing import Optional, Sequence

from gp.misc.evaluation.sm_wikidata import SemanticModelHelper
from gp.misc.itemdistance import get_hierarchy_scoring_fn
from graph.retworkx.api import digraph_all_simple_paths
from kgdata.models import Ontology
from loguru import logger
from sm.dataset import Example, FullTable
from sm.evaluation.cpa_cta_metrics import _cta_real, _get_cta
from sm.evaluation.prelude import (
    CTAEvalOutput,
    PrecisionRecallF1,
    _cpa_transformation,
    cta,
    sm_metrics,
)
from sm.evaluation.sm_metrics import PermutationExplosion
from sm.misc.funcs import assert_isinstance
from sm.outputs.semantic_model import ClassNode, DataNode
from sm.outputs.semantic_model import Edge as SMEdge
from sm.outputs.semantic_model import LiteralNode, SemanticModel


@dataclass
class WrappedSemanticModel:
    sm: SemanticModel
    is_normalized: bool = False
    is_cpa_transformed: bool = False


@dataclass
class CPAEvalRes:
    result: PrecisionRecallF1
    gold_sm: SemanticModel
    cpa_gold_sm: SemanticModel
    cpa_pred_sm: SemanticModel


@dataclass
class CTAEvalRes:
    result: CTAEvalOutput
    # mapping from column index (converted to string) to class id
    cta_gold: dict[str, str]
    # mapping from column index (converted to string) to class id
    cta_pred: dict[str, str]


class Evaluator:
    def __init__(
        self,
        ontology: Ontology,
        entity_labels: Optional[Mapping[str, str]] = None,
    ):
        self.sm_helper = SemanticModelHelper(
            entity_labels, ontology.props, ontology.kgns
        )
        self.kgns = self.sm_helper.kgns
        self.class_scoring_fn = get_hierarchy_scoring_fn(ontology.classes, self.kgns)
        self.prop_scoring_fn = get_hierarchy_scoring_fn(ontology.props, self.kgns)

    def avg_cpa(
        self,
        examples: Sequence[Example[FullTable]],
        sms: Sequence[SemanticModel],
        extra_cols: tuple[str, str] = ("", ""),
    ) -> list[CPAEvalRes]:
        cpas: list[CPAEvalRes] = []
        for ex, pred_sm in zip(examples, sms):
            cpas.append(self.cpa(ex, pred_sm))

        avg_cpa = PrecisionRecallF1.avg([cpa.result for cpa in cpas])
        logger.info(
            "for copying...\n{}\tcpa-p\tcpa-r\tcpa-f1\n{}",
            (extra_cols[0] + "\t") if extra_cols[0] else "",
            ",".join(
                ([extra_cols[1]] if extra_cols[1] else [])
                + [
                    "%.2f" % (round(float(x) * 100, 2))
                    for x in [avg_cpa.precision, avg_cpa.recall, avg_cpa.f1]
                ],
            ),
        )
        # fmt: on
        return cpas

    def avg_cta(
        self,
        examples: Sequence[Example[FullTable]],
        sms: Sequence[
            SemanticModel | Mapping[str, str] | Mapping[str, dict[str, float]]
        ],
        extra_cols: tuple[str, str] = ("", ""),
    ) -> list[CTAEvalOutput]:
        ctas: list[CTAEvalOutput] = []
        for ex, pred_sm in zip(examples, sms):
            ctas.append(self.cta(ex, pred_sm).result)
        avg_cta = PrecisionRecallF1.avg(ctas)
        logger.info(
            "for copying...\n{}\tcta-p\tcta-r\tcta-f1\n{}",
            (extra_cols[0] + "\t") if extra_cols[0] else "",
            ",".join(
                ([extra_cols[1]] if extra_cols[1] else [])
                + [
                    "%.2f" % (round(float(x) * 100, 2))
                    for x in [avg_cta.precision, avg_cta.recall, avg_cta.f1]
                ],
            ),
        )
        # fmt: on
        return ctas

    def cpa(
        self,
        example: Example[FullTable],
        sm: SemanticModel,
    ) -> CPAEvalRes:
        """Calculate the CPA score. The code is borrowed from: sm.evaluation.cpa_cta_metrics.cpa to adapt with this class API that uses WrappedSemanticModel"""
        gold_sms = self.get_example_gold_sms(example)
        cpa_pred_sm = self.convert_sm_for_cpa(self.norm_sm(sm))
        cpa_gold_sms = [self.convert_sm_for_cpa(gold_sm) for gold_sm in gold_sms]
        return self._cpa_real(
            cpa_pred_sm,
            cpa_gold_sms,
            gold_sms,
        )

    def _cpa_real(
        self,
        cpa_pred_sm: WrappedSemanticModel,
        cpa_gold_sms: list[WrappedSemanticModel],
        gold_sms: list[WrappedSemanticModel],
    ) -> CPAEvalRes:
        """Calculate the CPA score. The code is borrowed from: sm.evaluation.cpa_cta_metrics.cpa to adapt with this class API that uses WrappedSemanticModel"""
        assert cpa_pred_sm.is_normalized and cpa_pred_sm.is_cpa_transformed
        assert all(
            cpa_gold_sm.is_normalized and cpa_gold_sm.is_cpa_transformed
            for cpa_gold_sm in cpa_gold_sms
        )

        output = None
        best_cpa_gold_sm = None
        best_gold_sm = None

        for i, cpa_gold_sm in enumerate(cpa_gold_sms):
            try:
                res = sm_metrics.precision_recall_f1(
                    gold_sm=cpa_gold_sm.sm,
                    pred_sm=cpa_pred_sm.sm,
                    scoring_fn=self.prop_scoring_fn,
                    # debug_dir="/tmp",
                    hard_permutation_threshold=100000,
                )
            except PermutationExplosion:
                res = sm_metrics.precision_recall_f1(
                    gold_sm=self.uniquify_statement(cpa_gold_sm.sm),
                    pred_sm=self.uniquify_statement(cpa_pred_sm.sm),
                    scoring_fn=self.prop_scoring_fn,
                    debug_dir="/tmp",
                    hard_permutation_threshold=100000,
                )

            if output is None or res.f1 > output.f1:
                output = res
                best_cpa_gold_sm = cpa_gold_sm
                best_gold_sm = gold_sms[i]

        assert (
            output is not None
            and best_cpa_gold_sm is not None
            and best_gold_sm is not None
        )

        return CPAEvalRes(
            result=output,
            gold_sm=best_gold_sm.sm,
            cpa_gold_sm=best_cpa_gold_sm.sm,
            cpa_pred_sm=cpa_pred_sm.sm,
        )

    def cta(
        self,
        example: Example[FullTable],
        pred_sm: SemanticModel | Mapping[str, str] | Mapping[str, dict[str, float]],
    ) -> CTAEvalRes:
        if isinstance(pred_sm, Mapping):
            pred_cta = {}
            for k, v in pred_sm.items():
                if isinstance(v, dict):
                    curi = max(v.items(), key=itemgetter(1))[0]
                else:
                    curi = v

                if not self.kgns.is_uri(curi):
                    assert self.kgns.is_id(curi)
                    curi = self.kgns.id_to_uri(curi)
                pred_cta[k] = curi

            gold_cta, cta_output = max(
                [
                    (
                        (gold_cta := _get_cta(gold_sm.sm, self.sm_helper.ID_PROPS)),
                        _cta_real(
                            gold_cta,
                            pred_cta,
                            self.class_scoring_fn,
                        ),
                    )
                    for gold_sm in self.get_example_gold_sms(example)
                ],
                key=lambda x: x[1].f1,
            )
        else:
            gold_sms = self.get_example_gold_sms(example)
            wrapped_pred_sm = self.norm_sm(pred_sm)
            gold_sm, cta_output = max(
                [
                    (
                        gold_sm,
                        cta(
                            gold_sm.sm,
                            wrapped_pred_sm.sm,
                            self.sm_helper.ID_PROPS,
                            self.class_scoring_fn,
                        ),
                    )
                    for gold_sm in gold_sms
                ],
                key=lambda x: x[1].f1,
            )

            pred_cta = _get_cta(wrapped_pred_sm.sm, self.sm_helper.ID_PROPS)
            gold_cta = _get_cta(gold_sm.sm, self.sm_helper.ID_PROPS)
        return CTAEvalRes(cta_output, gold_cta, pred_cta)

    # @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_example_gold_sms(
        self, example: Example[FullTable]
    ) -> list[WrappedSemanticModel]:
        return self.get_equiv_sms(example.sms)

    # @CacheMethod.cache(CacheMethod.single_object_arg)
    def norm_sm(self, sm: SemanticModel) -> WrappedSemanticModel:
        return WrappedSemanticModel(self.sm_helper.norm_sm(sm), is_normalized=True)

    # @CacheMethod.cache(CacheMethod.single_object_arg)
    def convert_sm_for_cpa(
        self, wrapped_sm: WrappedSemanticModel
    ) -> WrappedSemanticModel:
        """Convert a semantic model to another model for evaluating the CPA task:
        - SemModelTransformation.replace_class_nodes_by_subject_columns(sm, id_props)
        - SemModelTransformation.remove_isolated_nodes(sm)
        """
        assert wrapped_sm.is_normalized
        if wrapped_sm.is_cpa_transformed:
            return wrapped_sm

        sm = wrapped_sm.sm
        cpa_sm = sm.deep_copy()
        _cpa_transformation(cpa_sm, self.sm_helper.ID_PROPS)

        return WrappedSemanticModel(cpa_sm, is_normalized=True, is_cpa_transformed=True)

    def get_equiv_sms(self, sms: list[SemanticModel]) -> list[WrappedSemanticModel]:
        return [
            WrappedSemanticModel(equiv_sm, is_normalized=True)
            for sm in sms
            for equiv_sm in self.sm_helper.gen_equivalent_sm(
                sm, strict=False, incorrect_invertible_props={"P571", "P582"}
            )
        ]

    def uniquify_statement(self, sm: SemanticModel) -> SemanticModel:
        """For handle permutation issue when evaluating precision/recall/f1"""
        newsm = sm.deep_copy()
        for node in newsm.iter_nodes():
            if isinstance(node, ClassNode):
                if node.abs_uri == self.sm_helper.kgns.statement_uri:
                    inedges = newsm.in_edges(node.id)
                    outedges = newsm.out_edges(node.id)

                    if len(inedges) == 1 and len(outedges) <= 1:
                        postfix = "_" + inedges[0].rel_uri
                        node.abs_uri = self.sm_helper.kgns.statement_uri + postfix
                        node.rel_uri = self.sm_helper.kgns.statement_uri + postfix
        return newsm

    def get_cpa_between_columns(
        self, sms: list[WrappedSemanticModel]
    ) -> set[tuple[int, int, str, str]]:
        """Get relationships between columns in a form of (source column, target column, property, qualifier).

        Note: This function will NOT generate relationships to/from entity or literal nodes.
        """
        rels: set[tuple[int, int, str, str]] = set()
        for sm in sms:
            cpa_sm = self.convert_sm_for_cpa(sm)
            for snode in cpa_sm.sm.iter_nodes():
                if not (
                    isinstance(snode, ClassNode)
                    and snode.abs_uri == self.kgns.statement_uri
                ):
                    continue

                (usedge,) = cpa_sm.sm.in_edges(snode.id)
                u = cpa_sm.sm.get_node(usedge.source)
                if not isinstance(u, DataNode):
                    continue
                sourcecol = u.col_index

                for svedge in cpa_sm.sm.out_edges(snode.id):
                    v = cpa_sm.sm.get_node(svedge.target)
                    if not isinstance(v, DataNode):
                        continue
                    targetcol = v.col_index
                    rels.add(
                        (
                            sourcecol,
                            targetcol,
                            self.kgns.uri_to_id(usedge.abs_uri),
                            self.kgns.uri_to_id(svedge.abs_uri),
                        )
                    )
        return rels

    def remove_incorrect_relationships(
        self, pred_sm: SemanticModel, gold_sms: list[WrappedSemanticModel]
    ) -> WrappedSemanticModel:
        """Remove easily detected incorrect relationships from the predicted semantic model to measure
        the maximum recall of the predicted semantic model. This is often used to measure the quality of
        candidate graphs.

        The returned result is a normalized CPA model.
        """
        gold_rels: set[tuple[int, int, str, str]] = self.get_cpa_between_columns(
            gold_sms
        )

        # remove relationships that aren't in the ground truth.
        newsm = self.convert_sm_for_cpa(self.norm_sm(pred_sm))
        for snode in newsm.sm.nodes():
            if (
                isinstance(snode, ClassNode)
                and snode.abs_uri == self.kgns.statement_uri
            ):
                (usedge,) = newsm.sm.in_edges(snode.id)
                u = newsm.sm.get_node(usedge.source)
                assert isinstance(u, DataNode)
                sourcecol = u.col_index

                for svedge in newsm.sm.out_edges(snode.id):
                    v = newsm.sm.get_node(svedge.target)
                    assert isinstance(v, DataNode)
                    targetcol = v.col_index
                    if (
                        sourcecol,
                        targetcol,
                        self.kgns.uri_to_id(usedge.abs_uri),
                        self.kgns.uri_to_id(svedge.abs_uri),
                    ) not in gold_rels:
                        newsm.sm.remove_edge(svedge.id)

                if newsm.sm.out_degree(snode.id) == 0:
                    newsm.sm.remove_node(snode.id)

        # remove duplicate edges
        dup_edges = defaultdict(list)
        for snode in newsm.sm.nodes():
            if not (
                isinstance(snode, ClassNode)
                and snode.abs_uri == self.kgns.statement_uri
                and newsm.sm.out_degree(snode.id) == 1
            ):
                continue

            (usedge,) = newsm.sm.in_edges(snode.id)
            u = newsm.sm.get_node(usedge.source)
            assert isinstance(u, DataNode)
            sourcecol = u.col_index

            (svedge,) = newsm.sm.out_edges(snode.id)
            v = newsm.sm.get_node(svedge.target)
            assert isinstance(v, DataNode)
            targetcol = v.col_index

            key = (
                sourcecol,
                targetcol,
                usedge.abs_uri,
                svedge.abs_uri,
            )
            dup_edges[key].append(snode.id)

        for key, snode_ids in dup_edges.items():
            if len(snode_ids) > 1:
                for snode_id in snode_ids[1:]:
                    newsm.sm.remove_node(snode_id)

        return newsm

    def keep_targets(
        self, sm: SemanticModel, cpa: Sequence[tuple[int, int]], cta: Sequence[int]
    ) -> SemanticModel:
        wsm = self.norm_sm(sm)
        wsm = WrappedSemanticModel(
            wsm.sm.copy(),
            is_normalized=wsm.is_normalized,
            is_cpa_transformed=wsm.is_cpa_transformed,
        )
        # first of all, we need to validate the targets, if a column is not an entity column, it cannot participate in the CPA
        entcols = set(cta)
        for ci, cj in cpa:
            if ci not in entcols:
                raise ValueError(
                    f"Column {ci} is not an entity column but it has outgoing relationships"
                )

        # remove literal nodes such as entity nodes
        for v in list(wsm.sm.iter_nodes()):
            if isinstance(v, LiteralNode):
                wsm.sm.remove_node(v.id)
        for v in list(wsm.sm.iter_nodes()):
            if (
                isinstance(v, ClassNode)
                and v.abs_uri == self.kgns.statement_uri
                and (wsm.sm.in_degree(v.id) == 0 or wsm.sm.out_degree(v.id) == 0)
            ):
                wsm.sm.remove_node(v.id)

        # remove columns that are tagged as entity column but not in the target list first -- don't need to remove standalone data nodes
        for v in wsm.sm.iter_data_nodes():
            if (
                not wsm.sm.is_entity_column(v.col_index, self.sm_helper.ID_PROPS)
                or v.col_index in cta
            ):
                continue

            # find the class node <u> that is connected to this column and remove it
            (uvedge,) = wsm.sm.in_edges(v.id)
            u = wsm.sm.get_node(uvedge.source)
            assert isinstance(u, ClassNode) and u.abs_uri != self.kgns.statement_uri

            # now to remove <u>, we need to remove the statement nodes as well
            for usedge in wsm.sm.out_edges(u.id):
                s = wsm.sm.get_node(usedge.target)
                if isinstance(s, ClassNode):
                    assert (
                        s.abs_uri == self.kgns.statement_uri
                        and wsm.sm.in_degree(usedge.target) == 1
                    )
                    wsm.sm.remove_node(usedge.target)
                else:
                    assert isinstance(s, DataNode) and s.col_index == v.col_index
            # we need to rewire the edge to the column node before removing <u>
            for suedge in wsm.sm.in_edges(u.id):
                wsm.sm.add_edge(
                    SMEdge(
                        source=suedge.source,
                        target=v.id,
                        abs_uri=suedge.abs_uri,
                        rel_uri=suedge.rel_uri,
                        approximation=suedge.approximation,
                        readable_label=suedge.readable_label,
                    )
                )
            wsm.sm.remove_node(u.id)

        # remove relationships that are not in the cpa
        cpa_wsm = self.convert_sm_for_cpa(wsm)
        relcols = set(cpa)
        remove_rels: set[tuple[int, int]] = set()

        for s in cpa_wsm.sm.iter_nodes():
            if isinstance(s, ClassNode):
                assert (
                    s.abs_uri == self.kgns.statement_uri
                ), "The cpa model only contains statement and data nodes"
                (uvedge,) = cpa_wsm.sm.in_edges(s.id)
                for svedge in cpa_wsm.sm.out_edges(s.id):
                    u = assert_isinstance(cpa_wsm.sm.get_node(uvedge.source), DataNode)
                    v = assert_isinstance(cpa_wsm.sm.get_node(svedge.target), DataNode)
                    if (u.col_index, v.col_index) not in relcols:
                        remove_rels.add(
                            (
                                u.col_index,
                                v.col_index,
                            )
                        )

        for ci, cj in remove_rels:
            # we need to find the correct source & target nodes to remove the relationship
            if wsm.sm.is_entity_column(ci, self.sm_helper.ID_PROPS):
                (inedge,) = wsm.sm.in_edges(wsm.sm.get_data_node(ci).id)
                u = wsm.sm.get_node(inedge.source)
                # class node connects with entity columns must not be the statement node (normalization exclude id props)
                assert isinstance(u, ClassNode) and u.abs_uri != self.kgns.statement_uri
            else:
                u = wsm.sm.get_data_node(ci)

            if wsm.sm.is_entity_column(cj, self.sm_helper.ID_PROPS):
                (inedge,) = wsm.sm.in_edges(wsm.sm.get_data_node(cj).id)
                v = wsm.sm.get_node(inedge.source)
                # class node connects with entity columns must not be the statement node (normalization exclude id props)
                assert isinstance(v, ClassNode) and v.abs_uri != self.kgns.statement_uri
            else:
                v = wsm.sm.get_data_node(cj)

            paths = digraph_all_simple_paths(wsm.sm, u.id, v.id, cutoff=2)
            for path in paths:
                inedge, outedge = path
                wsm.sm.remove_edge(inedge.id)
                wsm.sm.remove_edge(outedge.id)

                s = wsm.sm.get_node(inedge.target)
                assert isinstance(s, ClassNode) and s.abs_uri == self.kgns.statement_uri
                if wsm.sm.degree(inedge.target) == 0:
                    wsm.sm.remove_node(inedge.target)

        return wsm.sm
