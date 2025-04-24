from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import ray
from libraries.gp.gp.actors.data._db import KGDB, DBActor, DBActorArgs
from gramsplus.distantsupervision.make_dataset.prelude import (
    CombinedFilter,
    CombinedFilterArgs,
    EntityRecognitionV1,
    EntityRecognitionV1Args,
    FilterByEntType,
    FilterByEntTypeArgs,
    FilterByHeaderColType,
    FilterByHeaderColTypeArgs,
    FilterFn,
    FilterNotEntCol,
    FilterRegex,
    FilterRegexArgs,
    FilterV1,
    FilterV1Args,
    LabelFn,
    LabelV1,
    LabelV1Args,
    LabelV2,
    LabelV2Args,
    NoFilter,
    NoTransform,
    TransformFn,
    TransformV1,
    TransformV1Args,
    TransformV2,
)
from gramsplus.entity_linking.candidate_recognition.heuristic_model import (
    HeuristicCanRegArgs,
)
from gramsplus.semanticmodeling.text_parser import TextParser
from ream.cache_helper import MemBackend
from ream.dataset_helper import DatasetList, DatasetQuery
from ream.params_helper import NoParams
from ream.prelude import BaseActor, Cache
from ream.workspace import ReamWorkspace
from sm.dataset import Dataset, Example, FullTable
from sm.inputs.prelude import EntityIdWithScore
from sm.misc.ray_helper import get_instance, ray_map, ray_put
from sm.namespaces.utils import KGName
from sm_datasets.datasets import Datasets


@dataclass
class AutoLabelDataActorArgs:
    dataset_dir: Path = field(
        metadata={
            "help": "Path to the directory containing datasets of linked tables",
        }
    )
    skip_non_unique_mention: bool = field(
        default=False,
        metadata={
            "help": "Skip tables with non-unique mention",
        },
    )
    skip_column_with_no_type: bool = field(
        default=False,
        metadata={
            "help": "Column that auto-labeler cannot find any entity type will be skipped"
        },
    )
    recog_method: Literal["recog_v1"] = field(
        default="recog_v1",
        metadata={
            "help": "Entity recognition method to use",
            "variants": {"recog_v1": EntityRecognitionV1},
        },
    )
    recog_v1: EntityRecognitionV1Args = field(
        default_factory=EntityRecognitionV1Args,
        metadata={"help": "Entity recognition v1 arguments"},
    )
    filter_method: Literal[
        "no_filter",
        "filter_non_ent_col",
        "filter_regex",
        "filter_v1",
        "filter_ent_type",
        "filter_header_col_type",
        "filter_combined",
    ] = field(
        default="filter_regex",
        metadata={
            "help": "Filter method to use",
            "variants": {
                "no_filter": NoFilter,
                "filter_non_ent_col": FilterNotEntCol,
                "filter_regex": FilterRegex,
                "filter_v1": FilterV1,
                "filter_ent_type": FilterByEntType,
                "filter_header_col_type": FilterByHeaderColType,
                "filter_combined": CombinedFilter,
            },
        },
    )
    filter_regex: FilterRegexArgs = field(
        default_factory=FilterRegexArgs,
        metadata={"help": "Filter regex arguments"},
    )
    filter_ent_type: FilterByEntTypeArgs = field(
        default_factory=FilterByEntTypeArgs,
        metadata={"help": "Filter by entity type arguments"},
    )
    filter_not_ent_col: NoParams = field(
        default_factory=NoParams,
        metadata={"help": "Filter not entity column arguments"},
    )
    filter_header_col_type: Optional[FilterByHeaderColTypeArgs] = field(
        default=None,
        metadata={"help": "Filter by header col type arguments"},
    )
    filter_combined: CombinedFilterArgs = field(
        default_factory=CombinedFilterArgs,
        metadata={"help": "Combined filter arguments"},
    )
    filter_v1: FilterV1Args = field(
        default_factory=FilterV1Args,
        metadata={"help": "Filter v1 arguments"},
    )
    transform_method: Literal["transform_v1", "transform_v2", "no_transform"] = field(
        default="transform_v1",
        metadata={
            "help": "Transformation method to use",
            "variants": {
                "transform_v1": TransformV1,
                "no_transform": NoTransform,
                "transform_v2": TransformV2,
            },
        },
    )
    transform_v1: Optional[TransformV1Args] = field(
        default=None,
        metadata={"help": "Transformation v1 arguments"},
    )
    transform_v2: Optional[TransformV1Args] = field(
        default=None,
        metadata={"help": "Transformation v2 arguments"},
    )
    label_method: Literal["label_v1", "label_v2"] = field(
        default="label_v1",
        metadata={
            "help": "Label method to use",
            "variants": {"label_v1": LabelV1, "label_v2": LabelV2},
        },
    )
    label_v1: Optional[LabelV1Args] = None
    label_v2: Optional[LabelV2Args] = None


@dataclass
class AutoLabeledTable:
    table: FullTable
    entity_columns: list[int]
    entity_column_types: list[list[EntityIdWithScore]]

    def to_dict(self):
        return {
            "table": self.table.to_dict(),
            "entity_columns": self.entity_columns,
            "entity_column_types": [
                [e.to_dict() for e in coltypes] for coltypes in self.entity_column_types
            ],
        }

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(
            table=FullTable.from_dict(obj["table"]),
            entity_columns=obj["entity_columns"],
            entity_column_types=[
                [EntityIdWithScore.from_dict(e) for e in coltypes]
                for coltypes in obj["entity_column_types"]
            ],
        )


class AutoLabeledDataActor(BaseActor[AutoLabelDataActorArgs]):
    VERSION = 116

    def __init__(self, params: AutoLabelDataActorArgs, db_actor: DBActor):
        super().__init__(params, [db_actor])
        self.db_actor = db_actor

    @Cache.cache(backend=MemBackend())
    def __call__(self, dsquery: str) -> DatasetList[AutoLabeledTable]:
        parsed_dsquery = DatasetQuery.from_string(dsquery)
        tables = self.process_dataset(parsed_dsquery.dataset)
        return parsed_dsquery.select_list(tables)

    def is_autolabel_dataset(self, dsquery: str) -> bool:
        parsed_dsquery = DatasetQuery.from_string(dsquery)
        return (self.params.dataset_dir / parsed_dsquery.dataset).exists()

    @Cache.cache(
        backend=Cache.cls.dir(
            cls=DatasetList,
            dirname="process_{dataset}",
            compression="lz4",
            mem_persist=True,
            log_serde_time=True,
        ),
    )
    def process_dataset(self, dataset: str) -> DatasetList[AutoLabeledTable]:
        examples = Dataset(self.params.dataset_dir / dataset).load()

        # when we change the databases, sometimes the entities are not the same.
        kgdb = self.get_kgdb(dataset)
        entity_labels = kgdb.pydb.entity_labels.cache()
        props = kgdb.pydb.props.cache()
        redirections = kgdb.pydb.entity_redirections.cache()
        examples = Datasets().fix_redirection(
            examples,
            entity_labels,
            props,
            redirections,
            kgdb.kgns,
        )

        if len(examples) > 1000:
            selfargs = ray_put(
                (self.get_working_fs().root, self.params, self.db_actor.params)
            )

            def ray_process_example(
                selfargs, dataset: str, ream_args: dict, ex: Example[FullTable]
            ):
                ReamWorkspace.init_from_dict(ream_args)
                self = AutoLabeledDataActor.get_instance(selfargs)
                return self.process_example(
                    ex,
                    self.get_er(),
                    self.get_filter(dataset),
                    self.get_transformation(dataset),
                    self.get_label(dataset),
                    TextParser.default(),
                )

            ream_args_ref = ray.put(ReamWorkspace.get_instance().to_dict())
            output = ray_map(
                ray_process_example,
                [(selfargs, dataset, ream_args_ref, ex) for ex in examples],
                verbose=True,
                desc="generate auto-label dataset",
                auto_shutdown=True,
                is_func_remote=False,
            )
            autolabel_tables = [x for x in output if x is not None]
        else:
            er = self.get_er()
            fil = self.get_filter(dataset)
            map = self.get_transformation(dataset)
            labeler = self.get_label(dataset)

            text_parser = TextParser.default()
            autolabel_tables = []
            for ex in examples:
                newtable = self.process_example(ex, er, fil, map, labeler, text_parser)
                if newtable is not None:
                    autolabel_tables.append(newtable)
        return DatasetList(dataset, autolabel_tables)

    def process_example(
        self,
        ex: Example[FullTable],
        er: EntityRecognitionV1,
        fil: FilterFn,
        map: TransformFn,
        labeler: LabelFn,
        text_parser: TextParser,
    ) -> Optional[AutoLabeledTable]:
        table = ex.table
        if self.params.skip_non_unique_mention and has_non_unique_mention(table):
            return None

        # recognize entity columns
        entity_columns = er.recognize(table)
        entity_columns = fil.filter(table, entity_columns)

        table = map.transform(table, entity_columns)
        entity_column_types = labeler.label(table, entity_columns)

        if self.params.skip_column_with_no_type:
            valid_cols = [
                i for i in range(len(entity_columns)) if len(entity_column_types[i]) > 0
            ]
            entity_columns = [entity_columns[i] for i in valid_cols]
            entity_column_types = [entity_column_types[i] for i in valid_cols]

        if len(entity_columns) == 0:
            return None

        newtable = normalize_table(ex.table, text_parser)
        return AutoLabeledTable(newtable, entity_columns, entity_column_types)

    @Cache.cache(backend=MemBackend())
    def get_er(self):
        if self.params.recog_method == "recog_v1":
            return EntityRecognitionV1(self.params.recog_v1)
        raise NotImplementedError()

    @Cache.cache(backend=MemBackend())
    def get_filter(self, dataset: str):
        logfile = self.get_working_fs().root / f"logs/{dataset}/filter.log"
        kgdb = self.get_kgdb(dataset)

        if self.params.filter_method == "filter_regex":
            return FilterRegex(self.params.filter_regex, logfile)
        if self.params.filter_method == "filter_v1":
            return FilterV1(
                self.params.filter_v1,
                kgdb.pydb.entities.cache(),
                logfile,
            )
        if self.params.filter_method == "filter_combined":
            filters = [
                FilterRegex(self.params.filter_combined.regex, logfile),
                FilterByEntType(
                    self.params.filter_combined.ignore_types,
                    kgdb.pydb.entities.cache(),
                    kgdb.pydb.classes.cache(),
                    logfile,
                ),
            ]
            if self.params.filter_combined.header_col_type is not None:
                filters.append(
                    FilterByHeaderColType(
                        self.params.filter_combined.header_col_type,
                        self.get_label(dataset),
                        logfile,
                    )
                )
            return CombinedFilter(
                filters,
                logfile,
            )
        if self.params.filter_method == "no_filter":
            return NoFilter()
        if self.params.filter_method == "filter_non_ent_col":
            return FilterNotEntCol(HeuristicCanRegArgs(), logfile)
        raise NotImplementedError()

    @Cache.cache(backend=MemBackend())
    def get_transformation(self, dataset: str):
        logfile = self.get_working_fs().root / f"logs/{dataset}/transform.log"
        kgdb = self.get_kgdb(dataset)
        if self.params.transform_method == "transform_v1":
            assert self.params.transform_v1 is not None
            return TransformV1(
                self.params.transform_v1,
                kgdb.pydb.entities.cache(),
                kgdb.pydb.classes.cache(),
                logfile,
            )
        if self.params.transform_method == "transform_v2":
            assert self.params.transform_v2 is not None
            return TransformV2(
                self.params.transform_v2,
                kgdb.pydb.entities.cache(),
                kgdb.pydb.classes.cache(),
                logfile,
            )

        if self.params.transform_method == "no_transform":
            return NoTransform()
        raise NotImplementedError()

    @Cache.cache(backend=MemBackend())
    def get_label(self, dataset: str) -> LabelFn:
        kgdb = self.get_kgdb(dataset)
        if self.params.label_method == "label_v1":
            assert self.params.label_v1 is not None
            return LabelV1(
                self.params.label_v1,
                kgdb.pydb.entities.cache(),
                kgdb.pydb.entity_pagerank.cache(),
                kgdb.pydb.classes.cache(),
            )
        if self.params.label_method == "label_v2":
            assert self.params.label_v2 is not None
            return LabelV2(
                self.params.label_v2,
                kgdb.pydb.entities.cache(),
                kgdb.pydb.entity_pagerank.cache(),
                kgdb.pydb.classes.cache(),
            )
        raise NotImplementedError()

    @Cache.cache(backend=MemBackend())
    def get_kgname(self, dataset: str):
        if dataset.startswith("wt"):
            return KGName.Wikidata
        raise NotImplementedError(dataset)

    def get_kgdb(self, dataset: str) -> KGDB:
        kgname = self.get_kgname(dataset)
        return self.db_actor.kgdbs[kgname]

    @staticmethod
    def get_instance(
        args: AutoLabeledDataActor | tuple[str, AutoLabelDataActorArgs, DBActorArgs]
    ):
        if isinstance(args, AutoLabeledDataActor):
            return args
        return get_instance(
            lambda: AutoLabeledDataActor(args[1], DBActor(args[2])),
            f"autolabeled_actor:{args[0]}",
        )


def has_non_unique_mention(table: FullTable) -> bool:
    """Check if the example table has the same mention at different cells linked to different entities"""
    col2mention = defaultdict(lambda: defaultdict(set))

    for ri, ci, links in table.links.enumerate_flat_iter():
        if len(links) == 0:
            continue

        text = table.table[ri, ci]
        assert isinstance(text, str), text

        for link in links:
            mention = text[link.start : link.end]
            if len(mention) > 0:
                col2mention[ci][mention].update(link.entities)
                if len(col2mention[ci][mention]) > 1:
                    return True

    return False


def normalize_table(oldtable: FullTable, text_parser: TextParser) -> FullTable:
    table = deepcopy(oldtable)
    for col in table.table.columns:
        assert col.name is not None
        col.name = text_parser._norm_string(col.name)

    # normalize cells and links
    for ci, col in enumerate(table.table.columns):
        for ri, cell in enumerate(col.values):
            if isinstance(cell, str):
                newcell = text_parser._norm_string(cell)
                col.values[ri] = newcell

                if newcell != cell:
                    # adjust the links
                    for link in table.links[ri, ci]:
                        if (
                            cell[link.start : link.end]
                            != newcell[link.start : link.end]
                        ):
                            # the new mention is different from the old mention
                            before = text_parser._norm_nostrip_string(
                                cell[: link.start]
                            ).lstrip()
                            mention = text_parser._norm_nostrip_string(
                                cell[link.start : link.end]
                            )
                            if len(before) == 0 and mention.lstrip() != mention:
                                mention = mention.lstrip()
                            after = text_parser._norm_nostrip_string(
                                cell[link.end :]
                            ).rstrip()
                            if len(after) == 0 and mention.rstrip() != mention:
                                mention = mention.rstrip()
                            if before + mention + after != newcell:
                                raise NotImplementedError(
                                    f"Haven't implemented fixing where part of the mention has been changed. Recovered string: `{before+mention+after}` - transformed string: `{newcell}`"
                                )
                            link.start = len(before)
                            link.end = len(before) + len(mention)
    return table
