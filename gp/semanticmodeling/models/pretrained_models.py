from __future__ import annotations

from abc import abstractmethod
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from operator import itemgetter
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from gramsplus.actors.candidate_graph import CanGraphActor
from gramsplus.actors.candidate_ranking import CanRankActor
from gramsplus.misc.dataset import ColumnarDataset
from gramsplus.semanticmodeling.postprocessing.interface import EdgeProb, NodeProb
from ream.actors.base import BaseActor
from ream.cache_helper import Cache, MemBackend
from ream.data_model_helper import NumpyDataModelHelper
from ream.dataset_helper import DatasetList
from ream.helper import import_attr
from scripts.www24.cpa_helper import make_dataset
from scripts.www24.cta_helper import predict_column_prob, predict_cta
from scripts.www24.stores.cpa_store import CPAStore
from scripts.www24.stores.cpa_store_fn import CPAStoreFn, CPAStoreFnArgs
from sm.misc.funcs import import_attr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class CPAStoreInvokeArgs:
    args: CPAStoreFnArgs
    funcs: list[str]

    @cached_property
    def enable_fns(self) -> list[type[CPAStoreFn]]:
        return [import_attr(fn) for fn in self.funcs]


@dataclass
class PretrainedCPAModelArgs:
    model_class: str
    model_file: Path
    store_invoke_args: CPAStoreInvokeArgs


class PretrainedCPAModel(BaseActor[PretrainedCPAModelArgs]):
    VERSION = 100

    def __init__(
        self,
        params: PretrainedCPAModelArgs,
        store: CPAStore,
    ):
        super().__init__(params, [store])
        self.store = store

    @Cache.cache(backend=MemBackend())
    def __call__(self, dsquery: str, verbose: bool) -> DatasetList[EdgeProb]:
        args = copy(self.params.store_invoke_args.args)
        args.dsquery = dsquery
        return self.method.predict_dataset(
            self.store,
            CPAStoreInvokeArgs(args, self.params.store_invoke_args.funcs),
            verbose,
        )

    @cached_property
    def method(self):
        self.model_class: type[BaseTorchCPAModel] = import_attr(self.params.model_class)
        if torch.cuda.is_available():
            map_location = "cuda"
        else:
            map_location = "cpu"
        return self.model_class.load_from_checkpoint(
            self.params.model_file, map_location=map_location
        )


class BaseTorchCPAModel(pl.LightningModule):
    EVAL_BATCH_SIZE = 100
    EXPECTED_EVAL_ARGS = set()

    def make_dataset(
        self, store: CPAStore, store_invoke_args: CPAStoreInvokeArgs
    ) -> ColumnarDataset:
        ds = store(store_invoke_args.args, store_invoke_args.enable_fns)
        return make_dataset(ds, is_train=False)

    def predict_dataset(
        self, store: CPAStore, store_invoke_args: CPAStoreInvokeArgs, verbose: bool
    ) -> DatasetList[EdgeProb]:
        self.eval()

        params = next(self.parameters())
        device = params.device

        dsquery = store_invoke_args.args.dsquery

        dataset = self.make_dataset(store, store_invoke_args)
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
                desc="predicting cpa",
                disable=not verbose,
            ):
                kwargs = {}
                for arg in self.EXPECTED_EVAL_ARGS:
                    kwargs[arg] = batch[arg].to(device)
                output = self.forward(**kwargs)
                probs.append(output.prob.cpu())

            probs = torch.cat(probs).numpy()

        examples = store.data_actor(dsquery)
        cangraphs = store.cangraph_actor(dsquery)
        index = NumpyDataModelHelper.create_simple_index(
            [ex.table.table.table_id for ex in examples],
            [len(c.edgedf) for c in cangraphs],
        )

        output = []
        for ex, cangraph in zip(examples, cangraphs):
            edge_probs = {}
            start, end = index[ex.table.table.table_id]
            for ri, row in enumerate(cangraph.edgedf.iter_rows(named=True)):
                ri = ri + start
                source = str(row["source"])
                target = str(row["target"])
                statement = str(row["statement"])
                inedge = cangraph.edges[row["inedge"]]
                outedge = cangraph.edges[row["outedge"]]

                edge_probs[statement, target, outedge] = probs[ri]
                if outedge == inedge:
                    edge_probs[source, statement, inedge] = probs[ri]
            output.append(edge_probs)
        return DatasetList(examples.name, output)


@dataclass
class PretrainedCTAModelArgs:
    model_class: str
    model_args: dict


class PretrainedCTAModel(BaseActor[PretrainedCTAModelArgs]):
    VERSION = 101

    MOD_SELECTED_TYPE_SCORE = 100

    def __init__(self, params: PretrainedCTAModelArgs, cangraph_actor: CanGraphActor):
        super().__init__(params, [cangraph_actor])
        self.cangraph_actor = cangraph_actor

    @Cache.cache(backend=MemBackend())
    def __call__(self, dsquery: str, verbose: bool) -> DatasetList[NodeProb]:
        return self.method.predict_dataset(
            self.cangraph_actor, dsquery, self.params.model_args, verbose
        )

    @cached_property
    def method(self):
        self.model_class: type[HeuristicELCTAModel] = import_attr(
            self.params.model_class
        )
        return self.model_class()

    def cta_score_offset(self) -> float:
        return self.method.cta_score_offset(self.params.model_args)


class HeuristicELCTAModel:
    def predict_dataset(
        self, actor: CanGraphActor, dsquery: str, args: dict, verbose: bool
    ) -> DatasetList[NodeProb]:
        examples = actor.data_actor(dsquery)
        candidates = actor.canrank_actor.get_candidate_entities(dsquery)

        if args.get("topk", None) is not None:
            candidates = candidates.top_k_candidates(args["topk"])

        ress = predict_cta(
            actor.data_actor.get_kgdb(dsquery),
            examples,
            candidates,
            predict_column_prob,
            {"max_extend_distance": args["max_extend_distance"]},
        )

        output = []
        for res in ress:
            nodeprob = {}
            for ci, ctypes in res.col2type_freq.items():
                nodeprob[ci] = {ctype: cscore for ctype, cscore in ctypes}

                if args["modify_selected_type_score"]:
                    # changing the score of the correct type to be the highest one + 100
                    max_score = max(ctypes, key=itemgetter(1))[1]
                    assert max_score < 100
                    type, score = res.col2type[ci]
                    score = score + 100
                    nodeprob[ci][type] = score
            output.append(nodeprob)
        return DatasetList(examples.name, output)

    def cta_score_offset(self, args: dict) -> float:
        if args["modify_selected_type_score"]:
            return 100
        return 0.0
