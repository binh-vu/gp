from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

from gp.actors.data.prelude import KGDB, DataActor, GPExample, KGDBArgs
from gp.actors.sm.can_graph import CanGraphActor
from gp.entity_linking.cangen.common import TableCanGenResult
from gp.misc.appconfig import AppConfig
from gp.misc.evaluation.el_osin_mixin import EntityLinkingOsinActorMixin
from gp.misc.evaluation.sm_osin_mixin import SMOsinActorMixin
from gp.misc.evaluation.sm_wikidata import SemanticModelHelper
from gp.semanticmodeling.models.pretrained_models import (
    PretrainedCPAModel,
    PretrainedCTAModel,
)
from gp.semanticmodeling.postprocessing.cgraph import CGraph
from gp.semanticmodeling.postprocessing.greedy_known_targets import GreedyKnownTargetsFn
from gp.semanticmodeling.postprocessing.interface import EdgeProb, NodeProb
from gp.semanticmodeling.postprocessing.steiner_tree import SteinerTreeFn
from gramsplus.actors.data import PredictionTargets
from gramsplus.core.algorithms import CanGraphExtractedResult
from ream.cache_helper import Cache, FileBackend, MemBackend
from ream.prelude import BaseActor
from sm.dataset import Example, FullTable
from sm.misc.ray_helper import get_instance, ray_map, ray_put
from sm.namespaces.wikidata import WikidataNamespace
from sm.outputs.semantic_model import SemanticModel
from sm.prelude import O

PostprocessingMethod = Literal["greedy_known_targets", "steiner_tree"]
OnUntypeSourceColumnNode = Literal["create-class", "remove-link"]


@dataclass
class GraphInferenceActorArgs:
    postprocessing: PostprocessingMethod = field(
        default="steiner_tree",
        metadata={
            "help": "The post-processing method to use",
        },
    )
    on_untype_source_column_node: OnUntypeSourceColumnNode = field(
        default="create-class",
        metadata={
            "help": "How to handle untyped source node",
        },
    )


class GraphInferenceActor(
    SMOsinActorMixin[GraphInferenceActorArgs],
    EntityLinkingOsinActorMixin[GraphInferenceActorArgs],
    BaseActor[GraphInferenceActorArgs],
):
    """Graph Inference Actor"""

    VERSION = 101
    EXP_NAME = "Graph Inference"
    EXP_VERSION = 100

    def __init__(
        self,
        params: GraphInferenceActorArgs,
        cangraph_actor: CanGraphActor,
        pretrained_cpa_model: PretrainedCPAModel,
        pretrained_cta_model: PretrainedCTAModel,
    ):
        super().__init__(
            params,
            dep_actors=[
                cangraph_actor,
                pretrained_cpa_model,
                pretrained_cta_model,
            ],
        )

        self.pretrained_cpa_model = pretrained_cpa_model
        self.pretrained_cta_model = pretrained_cta_model

        self.cangraph_actor = cangraph_actor
        self.canrank_actor = cangraph_actor.canrank_actor
        self.db_actor = self.canrank_actor.db_actor

    # @Cache.cache(
    #     backend=MemBackend(),
    #     # backend=Cache.file.pickle(mem_persist=True),
    #     cache_ser_args={
    #         "cta_threshold": lambda x: str(x),
    #         "cpa_threshold": lambda x: str(x),
    #     },
    #     cache_args=["dataset", "cpa_threshold", "cta_threshold"],
    # )
    def batch_call(
        self,
        exs: list[GPExample],
        cpa_threshold: float = 0.5,
        cta_threshold: float = 0.5,
        known_targets: Optional[list[PredictionTargets]] = None,
    ):
        ex_cangraphs = self.cangraph_actor.batch_call(exs)
        edge_probs = self.pretrained_cpa_model.batch_call(exs)
        node_probs = self.pretrained_cta_model.batch_call(exs)

        if self.params.postprocessing == "greedy_known_targets":
            assert known_targets is not None
            targets = known_targets
            assert len(targets) == len(exs)
            postprocessing_args = []
            for target in targets:
                postprocessing_args.append(
                    {
                        "edge_targets": target.cpa,
                        "type_targets": target.cta,
                        "cpa_threshold": cpa_threshold,
                        "cta_threshold": cta_threshold,
                    }
                )
        else:
            postprocessing_args = [
                {
                    "cpa_threshold": cpa_threshold,
                    "cta_threshold": cta_threshold,
                    "cta_score_offset": self.pretrained_cta_model.cta_score_offset(),
                }
                for _ in range(len(exs))
            ]

        pred_sms = self.predict_sm(
            self.db_actor.kgdbs[exs[0].kgname],
            exs,
            ex_cangraphs,
            edge_probs,
            node_probs,
            self.params.postprocessing,
            postprocessing_args,
            self.params.on_untype_source_column_node,
        )
        return pred_sms

    def get_semantic_models(
        self, exs: list[GPExample], **kwargs
    ) -> list[SemanticModel]:
        return self.batch_call(exs, **kwargs)

    def get_candidate_entities(self, exs: list[GPExample]) -> list[TableCanGenResult]:
        return self.cangraph_actor.get_candidate_entities(exs)

    def get_edge_probs(self, exs: list[GPExample], **kwargs) -> list[EdgeProb]:
        db = self.db_actor.kgdbs[exs[0].kgname]
        props = db.pydb.props.cache()

        out: list[EdgeProb] = []
        ex_cangraphs = self.cangraph_actor.batch_call(exs)
        for cangraph, edge_probs in zip(
            ex_cangraphs, self.pretrained_cpa_model.batch_call(exs, True)
        ):
            new_edge_probs = {}
            for (source, target, edge), eprob in edge_probs.items():
                source = cangraph.nodes[int(source)]
                target = cangraph.nodes[int(target)]

                if source.is_column():
                    scol = source.try_as_column()
                    assert scol is not None
                    slbl = f"{scol.label} ({scol.column})"
                elif source.is_entity():
                    sent = source.try_as_entity()
                    assert sent is not None
                    slbl = f"ent:{sent.entity_id}"
                elif source.is_literal():
                    slit = source.try_as_literal()
                    assert slit is not None
                    slbl = f"literal:{slit.value.to_string_repr()}"
                else:
                    assert source.is_statement()
                    sstm = source.try_as_statement()
                    assert sstm is not None
                    slbl = f"statement:{sstm.id}"

                if target.is_column():
                    scol = target.try_as_column()
                    assert scol is not None
                    tlbl = f"{scol.label} ({scol.column})"
                elif target.is_entity():
                    sent = target.try_as_entity()
                    assert sent is not None
                    tlbl = f"ent:{sent.entity_id}"
                elif target.is_literal():
                    slit = target.try_as_literal()
                    assert slit is not None
                    tlbl = f"literal:{slit.value.to_string_repr()}"
                else:
                    assert target.is_statement()
                    sstm = target.try_as_statement()
                    assert sstm is not None
                    tlbl = f"statement:{sstm.id}"

                new_edge_probs[(slbl, tlbl, str(props[edge]))] = eprob
            out.append(new_edge_probs)

        return out

    def get_node_probs(self, exs: list[GPExample], **kwargs):
        return self.pretrained_cta_model.batch_call(exs, True)

    def predict_sm(
        self,
        kgdb: KGDB,
        examples: list[GPExample],
        cangraphs: list[CanGraphExtractedResult],
        edge_probs: list[EdgeProb],
        node_probs: list[NodeProb],
        postprocessing_method: PostprocessingMethod = "steiner_tree",
        postprocessing_args: Optional[Sequence[dict]] = None,
        on_untype_source_column_node: OnUntypeSourceColumnNode = "create-class",
    ):
        using_ray = len(examples) > 1 and not AppConfig.get_instance().is_ray_disable
        dbargs = ray_put(kgdb.args, using_ray)
        using_ray = False
        verbose = False

        return ray_map(
            postprocessing,
            [
                (
                    dbargs,
                    ex.table,
                    cangraphs[ei],
                    edge_probs[ei],
                    node_probs[ei],
                    postprocessing_method,
                    postprocessing_args[ei] if postprocessing_args is not None else {},
                    on_untype_source_column_node,
                )
                for ei, ex in enumerate(examples)
            ],
            desc="predict sm postprocessing",
            verbose=verbose,
            using_ray=using_ray,
            is_func_remote=False,
        )


def postprocessing(
    kgdb: KGDB | KGDBArgs,
    table: FullTable,
    cangraph: CanGraphExtractedResult,
    edge_probs: EdgeProb,
    node_probs: NodeProb,
    postprocessing_method: PostprocessingMethod,
    postprocessing_extraargs: dict,
    on_untype_source_column_node: OnUntypeSourceColumnNode,
):
    db = KGDB.get_instance(kgdb).pydb
    sm_helper = get_instance(
        lambda: SemanticModelHelper(
            db.entity_labels,
            db.props,
            WikidataNamespace.create(),
        ),
        __name__ + ".sm_helper",
    )

    if postprocessing_method == "steiner_tree":
        pp = SteinerTreeFn(
            table, CGraph.from_rust(cangraph), **postprocessing_extraargs
        )
    elif postprocessing_method == "greedy_known_targets":
        pp = GreedyKnownTargetsFn(
            CGraph.from_rust(cangraph), **postprocessing_extraargs
        )
    else:
        raise NotImplementedError(postprocessing_method)

    ppres = pp(node_probs, edge_probs)

    # convert cgraph into nodes/edges
    nodes = {}
    cpa = [(int(source), int(target), key) for source, target, key in ppres.edges]
    cta = ppres.nodes

    # gather nodes that will be in the semantic model
    uids = set()
    for source, target, prop_id in cpa:
        uids.add(source)
        uids.add(target)

    for u in cangraph.nodes:
        if u.is_column() and u.id not in uids:
            u = u.try_as_column()
            assert u is not None
            if u.column in cta:
                uids.add(u.id)

    for uid in uids:
        assert uid not in nodes
        u = cangraph.nodes[uid]
        if u.is_column():
            u = u.try_as_column()
            assert u is not None
            uprime = O.DataNode(
                col_index=u.column,
                label=str(u.label),
            )
        elif u.is_entity():
            u = u.try_as_entity()
            assert u is not None
            uprime = O.LiteralNode(
                value=sm_helper.kgns.id_to_uri(u.entity_id),
                readable_label=sm_helper.get_ent_label(u.entity_id),
                is_in_context=u.is_in_context,
                datatype=O.LiteralNodeDataType.Entity,
            )
        elif u.is_literal():
            u = u.try_as_literal()
            assert u is not None
            # TODO: how to ensure to_string_repr for the same value is the same?
            uprime = O.LiteralNode(
                value=u.value.to_string_repr(),
                readable_label=None,
                is_in_context=u.is_in_context,
                datatype=O.LiteralNodeDataType.String,
            )
        else:
            assert u.is_statement()
            u = u.try_as_statement()
            assert u is not None
            uprime = O.ClassNode(
                abs_uri=sm_helper.kgns.statement_uri,
                rel_uri=sm_helper.kgns.get_rel_uri(sm_helper.kgns.statement_uri),
                approximation=False,
                readable_label="wikibase:Statement",
            )
        nodes[uid] = uprime

    return sm_helper.create_sm(
        nodes, cpa, cta, on_untype_source_column_node=on_untype_source_column_node
    )
