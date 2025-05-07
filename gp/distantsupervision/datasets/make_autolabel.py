from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import orjson
from gp.actors.data._db import KGDB, KGDBArgs
from gp.distantsupervision.make_dataset.prelude import (
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
from gp.semanticmodeling.text_parser import TextParser
from libactor.cache import BackendFactory, IdentObj, cache
from libactor.misc import get_parallel_executor, orjson_dumps, typed_delayed
from ream.params_helper import NoParams
from sm.dataset import Example, FullTable
from sm.inputs.prelude import EntityIdWithScore
from sm.misc.funcs import batch


@dataclass
class AutoLabelDataActorArgs:
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

    def get_key(self):
        return orjson_dumps(asdict(self)).decode()


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


def make_autolabel(
    exs: list[Example[FullTable]],
    kgdb: KGDB,
    args: AutoLabelDataActorArgs,
    workdir: Path,
    n_jobs: int = -1,
) -> list[AutoLabeledTable]:
    batch_exs = batch(64, exs)
    return [
        x
        for arr in get_parallel_executor(n_jobs=n_jobs, return_as="generator")(
            typed_delayed(wrap_process_example)(lst, kgdb.args, args, workdir)
            for lst in batch_exs
        )
        for x in arr
    ]


def wrap_process_example(
    exs: list[Example[FullTable]],
    kgdb_args: KGDBArgs,
    args: AutoLabelDataActorArgs,
    workdir: Path,
) -> list[AutoLabeledTable]:
    kgdb = KGDB.get_instance(kgdb_args)
    ikgdb = IdentObj(kgdb.args.get_key(), kgdb)
    iargs = IdentObj(args.get_key(), args)

    er = get_er(iargs)
    filter = get_filter(iargs, ikgdb, workdir)
    map = get_transformation(iargs, ikgdb, workdir)
    labeler = get_label(iargs, ikgdb)
    text_parser = TextParser.default()

    return [
        result
        for ex in exs
        if (result := process_example(ex, er, filter, map, labeler, text_parser, args))
        is not None
    ]


def process_example(
    ex: Example[FullTable],
    er: EntityRecognitionV1,
    fil: FilterFn,
    map: TransformFn,
    labeler: LabelFn,
    text_parser: TextParser,
    args: AutoLabelDataActorArgs,
) -> Optional[AutoLabeledTable]:
    table = ex.table
    if args.skip_non_unique_mention and has_non_unique_mention(table):
        return None

    # recognize entity columns
    entity_columns = er.recognize(table)
    entity_columns = fil.filter(table, entity_columns)

    table = map.transform(table, entity_columns)
    entity_column_types = labeler.label(table, entity_columns)

    if args.skip_column_with_no_type:
        valid_cols = [
            i for i in range(len(entity_columns)) if len(entity_column_types[i]) > 0
        ]
        entity_columns = [entity_columns[i] for i in valid_cols]
        entity_column_types = [entity_column_types[i] for i in valid_cols]

    if len(entity_columns) == 0:
        return None

    newtable = normalize_table(ex.table, text_parser)
    return AutoLabeledTable(newtable, entity_columns, entity_column_types)


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


@cache(backend=BackendFactory.func.mem)
def get_er(args: IdentObj[AutoLabelDataActorArgs]):
    if args.value.recog_method == "recog_v1":
        return EntityRecognitionV1(args.value.recog_v1)
    raise NotImplementedError()


@cache(backend=BackendFactory.func.mem)
def get_filter(
    args: IdentObj[AutoLabelDataActorArgs], kgdb: IdentObj[KGDB], workdir: Path
):
    logfile = workdir / "filter.log"

    vargs = args.value
    vkgdb = kgdb.value

    if vargs.filter_method == "filter_regex":
        return FilterRegex(vargs.filter_regex, logfile)
    if vargs.filter_method == "filter_v1":
        return FilterV1(
            vargs.filter_v1,
            vkgdb.pydb.entities.cache(),
            logfile,
        )
    if vargs.filter_method == "filter_combined":
        filters = [
            FilterRegex(vargs.filter_combined.regex, logfile),
            FilterByEntType(
                vargs.filter_combined.ignore_types,
                vkgdb.pydb.entities.cache(),
                vkgdb.pydb.classes.cache(),
                logfile,
            ),
        ]
        if vargs.filter_combined.header_col_type is not None:
            filters.append(
                FilterByHeaderColType(
                    vargs.filter_combined.header_col_type,
                    get_label(args, kgdb),
                    logfile,
                )
            )
        return CombinedFilter(
            filters,
            logfile,
        )
    if vargs.filter_method == "no_filter":
        return NoFilter()
    if vargs.filter_method == "filter_non_ent_col":
        return FilterNotEntCol(logfile)
    raise NotImplementedError()


@cache(backend=BackendFactory.func.mem)
def get_label(args: IdentObj[AutoLabelDataActorArgs], kgdb: IdentObj[KGDB]) -> LabelFn:
    vargs = args.value
    vkgdb = kgdb.value
    if vargs.label_method == "label_v1":
        assert vargs.label_v1 is not None
        return LabelV1(
            vargs.label_v1,
            vkgdb.pydb.entities.cache(),
            vkgdb.pydb.entity_pagerank.cache(),
            vkgdb.pydb.classes.cache(),
        )
    if vargs.label_method == "label_v2":
        assert vargs.label_v2 is not None
        return LabelV2(
            vargs.label_v2,
            vkgdb.pydb.entities.cache(),
            vkgdb.pydb.entity_pagerank.cache(),
            vkgdb.pydb.classes.cache(),
        )
    raise NotImplementedError()


@cache(backend=BackendFactory.func.mem)
def get_transformation(
    args: IdentObj[AutoLabelDataActorArgs], kgdb: IdentObj[KGDB], workdir: Path
):
    vargs = args.value
    vkgdb = kgdb.value

    logfile = workdir / f"transform.log"
    if vargs.transform_method == "transform_v1":
        assert vargs.transform_v1 is not None
        return TransformV1(
            vargs.transform_v1,
            vkgdb.pydb.entities.cache(),
            vkgdb.pydb.classes.cache(),
            logfile,
        )
    if vargs.transform_method == "transform_v2":
        assert vargs.transform_v2 is not None
        return TransformV2(
            vargs.transform_v2,
            vkgdb.pydb.entities.cache(),
            vkgdb.pydb.classes.cache(),
            logfile,
        )

    if vargs.transform_method == "no_transform":
        return NoTransform()
    raise NotImplementedError()
