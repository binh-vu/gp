from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from operator import itemgetter
from pathlib import Path
from typing import Mapping, Optional, Sequence

import serde.csv
from gp.actors.data._db import KGDB, DBActor

# from gp.actors.data.data_autolabel import AutoLabeledTable
from gp.misc.evaluation.sm_wikidata import SemanticModelHelper
from libactor.actor import Actor
from libactor.cache import BackendFactory, IdentObj, cache
from rdflib import RDFS
from sm.dataset import Dataset, Example, FullTable
from sm.namespaces.namespace import KnowledgeGraphNamespace
from sm.namespaces.utils import KGName, get_kgns
from sm.prelude import M, O
from smml.dataset_helper import DatasetList, DatasetQuery

try:
    from sm_datasets.datasets import Datasets
except ImportError:
    Datasets = None


@dataclass
class DataActorArgs:
    skip_unk_ont_ent: bool = field(
        default=False,
        metadata={
            "help": "Skip examples with unknown ontology entities (e.g., Q1234567)"
        },
    )
    skip_no_sm: bool = field(
        default=False,
        metadata={"help": "Skip examples without semantic models"},
    )
    include_dataset_name: bool = field(
        default=False,
        metadata={
            "help": "Include the dataset name in the example id to make it unique"
        },
    )


@dataclass
class PredictionTargets:
    cea: list[tuple[int, int]]
    cta: list[int]
    cpa: list[tuple[int, int]]


@dataclass
class GPExample(Example[FullTable]):
    kgname: KGName


class DataActor(Actor[DataActorArgs]):
    VERSION = 110

    def __init__(
        self,
        params: DataActorArgs,
        db_actor: (
            DBActor | tuple[DBActor]
        ),  # for __reduce__ & consistent with the super().__init__
    ):
        if isinstance(db_actor, Sequence):
            db_actor = db_actor[0]
        super().__init__(params, [db_actor])
        self.db_actor = db_actor

    @cache(backend=BackendFactory.actor.mem)
    def get_prediction_targets(self, dsquery: str) -> DatasetList[PredictionTargets]:
        """Get the prediction targets for the given dataset query. The prediction targets are
        the columns (single/pair) that we want to predict the type and relation for.
        """
        dataset = DatasetQuery.from_string(dsquery).dataset

        assert dataset.startswith("semtab")
        loc = Datasets().get_dataset(dataset).location

        cea_target_file = loc / "CEA_targets.csv"
        cpa_target_file = loc / "CPA_targets.csv"
        cta_target_file = loc / "CTA_targets.csv"

        cea_lst = M.group_by(serde.csv.deser(cea_target_file), itemgetter(0))
        cpa_lst = M.group_by(serde.csv.deser(cpa_target_file), itemgetter(0))
        cta_lst = M.group_by(serde.csv.deser(cta_target_file), itemgetter(0))

        examples = self.load_dataset(dsquery)
        output = []
        for ex in examples:
            cea = [
                (int(ri) - 1, int(ci))
                for tid, ri, ci in cea_lst.get(ex.table.table.table_id, [])
            ]
            assert all(ri >= 0 for ri, ci in cea)

            tar = PredictionTargets(
                cea=cea,
                cta=[int(ci) for tid, ci in cta_lst.get(ex.table.table.table_id, [])],
                cpa=[
                    (int(ci), int(cj))
                    for tid, ci, cj in cpa_lst.get(ex.table.table.table_id, [])
                ],
            )

            output.append(tar)
        return DatasetList(examples.name, output, examples.provenance)

    @cache(
        backend=BackendFactory.actor.sqlite.serde(
            cls=DatasetList,
            mem_persist=True,
            compression="lz4",
            log_serde_time="DataActor.load_dataset",
        )
    )
    def load_dataset(self, dsquery: str) -> DatasetList[GPExample]:
        parsed_dsquery = DatasetQuery.from_string(dsquery)
        sm_examples = self._load_entire_dataset(parsed_dsquery.dataset)
        ds = parsed_dsquery.select_list(sm_examples)

        if self.params.include_dataset_name:
            for ex in ds:
                assert ex.id == ex.table.table.table_id
                ex.id = f"{ds.name}__{ex.id}"
                ex.table.table.table_id = ex.id

        if parsed_dsquery.postprocessing is not None:
            examples = ds
            for pp in parsed_dsquery.postprocessing:
                if pp == "no-unk-col":
                    fn = ignore_unk_columns
                else:
                    raise NotImplementedError(pp)
                examples = [r for ex in examples if (r := fn(ex)) is not None]
            return DatasetList(ds.name, examples, ds.provenance)
        return ds

    def forward(self, dsquery: str) -> IdentObj[DatasetList[GPExample]]:
        return IdentObj(key=f"{self.key}:{dsquery}", value=self.load_dataset(dsquery))

    @cache(
        backend=BackendFactory.actor.sqlite.pickle(mem_persist=True, compression="lz4")
    )
    def _load_entire_dataset(self, dataset: str) -> list[GPExample]:
        """
        Load dataset either by name or from disk.

        Args:
            dataset: The name of the dataset or the absolute path to the dataset file/folder.

        """
        if dataset.startswith("/"):
            # load dataset from disk
            assert Path(dataset).exists()
            examples = Dataset(Path(dataset)).load()
        else:
            ds = Datasets()
            db = self.get_kgdb(dataset).pydb
            entity_labels = db.entity_labels.cache()
            props = db.props.cache()
            redirections = db.entity_redirections.cache()

            m = re.match(r"(\w+)__v\d+", dataset)
            if m is not None:
                dataset = m.group(1)

            examples = ds.get_dataset(dataset).load()
            examples = ds.fix_redirection(
                examples,
                entity_labels,
                props,
                redirections,
                self.get_kgdb(dataset).kgns,
                skip_unk_ont_ent=self.params.skip_unk_ont_ent,
                skip_no_sm=self.params.skip_no_sm,
            )

        kgname = self.get_kgname(dataset)
        return [
            GPExample(id=ex.id, sms=ex.sms, table=ex.table, kgname=kgname)
            for ex in examples
        ]

    @classmethod
    @lru_cache()
    def get_kgname(cls, dsquery: str) -> KGName:
        dataset = DatasetQuery.from_string(dsquery).dataset
        if (
            dataset.startswith("wt")
            or dataset.startswith("semtab2022_r1")
            or dataset.startswith("semtab2023_r1")
            or dataset.startswith("semtab2024_wikitables")
            or dataset.find("wikidata") != -1
        ):
            return KGName.Wikidata
        if dataset.find("dbpedia") != -1 or dataset.startswith("t2dv2"):
            return KGName.DBpedia
        return KGName.Generic

    def get_kgdb(self, dsquery: str) -> KGDB:
        kgname = self.get_kgname(dsquery)
        return self.db_actor.kgdbs[kgname]

    @cache(backend=BackendFactory.actor.mem)
    def get_sm_helper(self, dsquery: str):
        db = self.get_kgdb(dsquery).pydb
        return SemanticModelHelper(
            db.entity_labels.cache(),
            db.props.cache(),
            get_kgns(self.get_kgname(dsquery)),
        )


# class ExtendedDataActor(DataActor):
#     """An extended data actor that can return datasets generated auto-labeled data actor.
#     Mainly used to manually run cangraph or other actors on the auto-labeled datasets.
#     """

#     VERSION = 100

#     def __init__(
#         self, params: DataActorArgs, autolabel_data_actor: AutoLabeledDataActor
#     ):
#         super(DataActor, self).__init__(params, [autolabel_data_actor])
#         self.db_actor = autolabel_data_actor.db_actor
#         self.autolabel_data_actor = autolabel_data_actor

#     def load_dataset(self, dsquery: str) -> DatasetList[GPExample]:
#         if self.autolabel_data_actor.is_autolabel_dataset(dsquery):
#             ds = self.autolabel_data_actor(dsquery)
#             kgdb = self.get_kgdb(dsquery)
#             kgns = kgdb.kgns
#             entity_label = kgdb.pydb.entity_labels.cache()
#             return DatasetList(
#                 ds.name,
#                 [
#                     from_auto_labeled_table(tbl, kgns, kgdb.kgname, entity_label)
#                     for tbl in ds
#                 ],
#                 ds.provenance,
#             )
#         else:
#             return super().load_dataset(dsquery)


# def from_auto_labeled_table(
#     tbl: AutoLabeledTable,
#     kgns: KnowledgeGraphNamespace,
#     kgname: KGName,
#     entity_label: Mapping[str, str],
# ) -> GPExample:
#     """To convert to the desired format, this method need to convert column types into a
#     semantic model. To support multiple types per column, we create as much classes as
#     needed. For example, column A has two types C1 and C2, then we will have C1 -> A and
#     C2 -> A in the same semantic model.
#     """
#     sm = O.SemanticModel()

#     colids = []
#     for ci, col in enumerate(tbl.table.table.columns):
#         assert col.index == ci
#         cname = col.clean_multiline_name
#         assert cname is not None
#         colids.append(sm.add_node(O.DataNode(ci, cname)))

#     for ci, ctypes in zip(tbl.entity_columns, tbl.entity_column_types):
#         cname = tbl.table.table.get_column_by_index(ci).clean_multiline_name
#         assert cname is not None
#         uid = colids[ci]
#         for ctype in ctypes:
#             assert ctype.id.belong_to(kgns)

#             cid = sm.add_node(
#                 O.ClassNode(
#                     (ctype_uri := kgns.id_to_uri(ctype.id)),
#                     kgns.get_rel_uri(ctype_uri),
#                     readable_label=f"{entity_label[ctype.id]} {ctype.id}",
#                 )
#             )
#             sm.add_edge(O.Edge(cid, uid, str(RDFS.label), "rdfs:label"))

#     return GPExample(
#         id=tbl.table.table.table_id, sms=[sm], table=tbl.table, kgname=kgname
#     )


def ignore_unk_columns(ex: GPExample) -> Optional[GPExample]:
    """Remove unannotated columns in an example."""
    keep_cols = []
    ignore_cols = []

    for col in ex.table.table.columns:
        if all(len(sm.get_semantic_types_of_column(col.index)) > 0 for sm in ex.sms):
            keep_cols.append(col.index)
        else:
            ignore_cols.append(col.index)

    keep_cols = sorted(keep_cols)
    if len(keep_cols) == 0:
        return None

    table = ex.table.keep_columns(keep_cols)
    sms = ex.sms

    if len(ignore_cols) > 0 and ex.sms is not None:
        new_sms = []
        for sm in ex.sms:
            newsm = sm.deep_copy()
            for ci in ignore_cols:
                if newsm.has_data_node(ci):
                    newsm.remove_node(newsm.get_data_node(ci).id)
            new_sms.append(newsm)
        sms = new_sms

    return GPExample(id=ex.id, sms=sms, table=table, kgname=ex.kgname)
