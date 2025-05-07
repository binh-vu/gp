"""
Attempt to fix incorrect auto links in the Wikipedia tables:

1. Detect linking to temporal entity and updating it.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

from gp.distantsupervision.make_dataset.interface import TransformFn
from gp.distantsupervision.make_dataset.utils import LoggerMixin, StrPath
from kgdata.models.entity import Entity
from kgdata.models.ont_class import OntologyClass
from kgdata.wikidata.models.wdclass import WDClass
from kgdata.wikidata.models.wdentity import WDEntity
from loguru import logger
from sm.dataset import FullTable
from sm.inputs.link import EntityId, Link
from sm.misc.funcs import assert_not_null


class ChangeStatus(enum.IntEnum):
    NoChange = 0
    Update = 1
    Remove = 2


DEFAULT_TEMPORAL_CLASSES = {
    "Q46135307",  # nation at sport competition
    "Q27020041",  # sports season
    "Q29791211",  # sport in a geographic region
}
DEFAULT_SURGERIES = [
    "Q46135307:P17",  # nation at sport competition -> country via P17 (country)
    "Q46135307:P1532",  # nation at sport competition -> country for sport (P1532) -- noisy
    "Q1539532:P5138",  # sports season of a sports club -> club via P5138 (season of club or team)
    "Q27020041:P3450",  # sports season (Q27020041) -> sports season of league or competition (P3450) -- noisy
    "Q29791211:P17",  # sport in a geographic region (Q29791211) -> country (P17)
]


@dataclass
class TransformV1Args:
    noisy_threshold: float = field(
        default=0.5,
        metadata={
            "help": "When the freq of noisy cells in the column exceed the threshold, the transformation on that column is triggered."
        },
    )
    temporal_classes: set[str] = field(
        default_factory=lambda: DEFAULT_TEMPORAL_CLASSES,
        metadata={
            "help": "Temporal classes that are used to detect temporal entities."
        },
    )
    known_surgeries: list[str] = field(
        default_factory=lambda: DEFAULT_SURGERIES,
        metadata={
            "help": "Known surgeries of <classid>:[<prop>] or <classid>:<prop> where [] denote optional"
        },
    )


class TransformV1(TransformFn, LoggerMixin):
    VERSION = 104

    def __init__(
        self,
        args: TransformV1Args,
        entities: Mapping[str, WDEntity | Entity],
        classes: Mapping[str, OntologyClass],
        log_file: Optional[StrPath] = None,
    ):
        super().__init__(log_file)
        self.args = args
        self.entities = entities
        self.classes = classes
        self.temporal_classes = args.temporal_classes

        regex = re.compile(r"(\w+):(\w+)")
        self.known_surgeries: list[Surgery] = [
            Surgery(
                steps=[
                    SurgeryStep(
                        (m := assert_not_null(regex.match(sur))).group(1),
                        m.group(2),
                    )
                    for sur in surpath.split(" -> ")
                ],
                optional=surpath.startswith("[") and surpath.endswith("]"),
            )
            for surpath in args.known_surgeries
        ]

    def transform(self, table: FullTable, columns: list[int]) -> FullTable:
        nrows, ncols = table.table.shape()
        newlinks = None
        for ci in columns:
            collinks = self.transform_column(table, ci)
            if collinks is not None:
                if newlinks is None:
                    newlinks = table.links.shallow_copy()
                for ri in range(nrows):
                    newlinks[ri, ci] = collinks[ri]
        if newlinks is None:
            return table

        self.logger.debug("Transformed table {}", table.table.table_id)
        return FullTable(table.table, table.context, newlinks)

    def transform_column(
        self, table: FullTable, column: int
    ) -> Optional[list[list[Link]]]:
        nrows, ncols = table.table.shape()
        n_transformed = 0
        collinks = []
        for ri in range(nrows):
            newlinks = self.transform_cell(table, ri, column)
            if newlinks is not None:
                collinks.append(newlinks)
                n_transformed += 1
            else:
                collinks.append(table.links[ri, column])
        if n_transformed / nrows >= self.args.noisy_threshold:
            return collinks
        return None

    def transform_cell(
        self, table: FullTable, row: int, column: int
    ) -> Optional[list[Link]]:
        """Transform a cell if possible:

        1. empty mentions are ignored even if they are linked (e.g., country flag).
        2. entity that are links to temporal entities are replaced with non-temporal entity.
        """
        non_empty_links = [
            link for link in table.links[row, column] if link.end > link.start
        ]

        modified = len(non_empty_links) != len(table.links[row, column])
        newlinks = []
        for link in non_empty_links:
            newents = []
            for entid in link.entities:
                ent = self.entities[entid]
                newentid, status = self.transform_entity(ent)
                modified = modified or status != ChangeStatus.NoChange
                if status == ChangeStatus.NoChange:
                    newents.append(entid)
                elif status == ChangeStatus.Update:
                    newents.append(EntityId(newentid, entid.type))
                elif status == ChangeStatus.Remove:
                    continue
            if modified:
                newlinks.append(
                    Link(
                        start=link.start,
                        end=link.end,
                        url=link.url,
                        entities=newents,
                    )
                )

        if modified:
            return newlinks
        return None

    def transform_entity(self, entity: WDEntity | Entity) -> tuple[str, ChangeStatus]:
        """Transform an entity if possible.

        It detects if the entity is an instance of temporal classes such as Football Season, and if found, it will resolve to the correct entity using predefined rules that
        select the value of the property.
        """
        entclasses = [self.classes[cid] for cid in entity.instance_of()]
        if not any(
            any(
                tempcls == entclass.id or tempcls in entclass.ancestors
                for tempcls in self.temporal_classes
            )
            for entclass in entclasses
        ):
            return entity.id, ChangeStatus.NoChange

        found_surgeries = False
        for surgery in self.known_surgeries:
            newentid, stt = self.try_surgery(surgery, entity, entclasses)
            if stt == "notapplicable":
                continue

            found_surgeries = True

            if stt == "success":
                return newentid, ChangeStatus.Update

            if stt == "nodata":
                # this will print a warning, and we continue to try if there is any other surgery
                # that can be applied. but if there is no surgery that can be applied, we will
                # remove the entity.
                continue

        if not found_surgeries:
            raise Exception(f"Unknown surgery {entity.label} ({entity.id})")
        return entity.id, ChangeStatus.Remove

    def try_surgery(
        self,
        surgery: Surgery,
        entity: WDEntity | Entity,
        entclasses: list[OntologyClass],
    ) -> tuple[str, Literal["nodata", "notapplicable", "success"]]:
        current_ent = entity
        current_entclasses = entclasses

        n = len(surgery.steps)

        for i, step in enumerate(surgery.steps):
            if not any(
                step.type == entclass.id or step.type in entclass.ancestors
                for entclass in current_entclasses
            ):
                if i > 0 and not surgery.optional:
                    self.logger.warning(
                        f"The next step requires type {step.type}, but entity {current_ent.label} ({current_ent.id}) isn't of that type."
                    )
                    return "", "nodata"
                return "", "notapplicable"
            if step.prop not in current_ent.props:
                if not surgery.optional:
                    self.logger.warning(
                        f"Missing {step.prop} for {current_ent.label} ({current_ent.id})"
                    )
                return "", "nodata"

            potential_values = set(current_ent.get_object_prop_value(step.prop))
            if len(potential_values) > 1:
                # there are too many values, we don't know which one to choose.
                if entity.id == "Q2366374":
                    # hard code a few cases as it's take time more time to write code to handle them.
                    # these cases are error in Wikidata.
                    potential_values = {"Q213347"}
                elif entity.id == "Q749109" and step.prop == "P17":
                    potential_values = {"Q145"}
                else:
                    raise Exception(
                        f"Too many values for {current_ent.label} ({current_ent.id}) -> {step.prop}"
                    )

            if i == n - 1:
                return next(iter(potential_values)), "success"

            current_ent = self.entities[next(iter(potential_values))]
            current_entclasses = [
                self.classes[cid] for cid in current_ent.instance_of()
            ]

        raise Exception("Shouldn't reach here")


@dataclass
class Surgery:
    steps: list[SurgeryStep]
    optional: bool


@dataclass
class SurgeryStep:
    type: str
    prop: str
