"""
Attempt to fix incorrect auto links in the Wikipedia tables:

1. Detect linking to temporal entity and updating it.
"""

from __future__ import annotations

from typing import Literal, Optional

from gp.distantsupervision.make_dataset.mv1 import ChangeStatus, Surgery, TransformV1
from kgdata.models.entity import Entity
from kgdata.models.ont_class import OntologyClass
from kgdata.wikidata.models.wdentity import WDEntity
from sm.dataset import FullTable
from sm.inputs.link import EntityId, Link


class TransformV2(TransformV1):
    """In TransformV1, we map one entity to another entity. However, there are cases where

    we map from one to multiple entities.
    """

    VERSION = 100

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
                newentids, status = self.transform_entities(ent)
                modified = modified or status != ChangeStatus.NoChange
                if status == ChangeStatus.NoChange:
                    newents.append(entid)
                elif status == ChangeStatus.Update:
                    newents.extend(
                        (EntityId(newentid, entid.type) for newentid in newentids)
                    )
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

    def transform_entities(
        self, entity: WDEntity | Entity
    ) -> tuple[list[str], ChangeStatus]:
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
            return [entity.id], ChangeStatus.NoChange

        found_surgeries = False
        for surgery in self.known_surgeries:
            newentids, stt = self.try_surgery(surgery, entity, entclasses)
            if stt == "notapplicable":
                continue

            found_surgeries = True

            if stt == "success":
                return newentids, ChangeStatus.Update

            if stt == "nodata":
                # this will print a warning, and we continue to try if there is any other surgery
                # that can be applied. but if there is no surgery that can be applied, we will
                # remove the entity.
                continue

        if not found_surgeries:
            raise Exception(f"Unknown surgery {entity.label} ({entity.id})")
        return [entity.id], ChangeStatus.Remove

    def try_surgery(
        self,
        surgery: Surgery,
        entity: WDEntity | Entity,
        entclasses: list[OntologyClass],
    ) -> tuple[list[str], Literal["nodata", "notapplicable", "success"]]:
        current_ents = [(entity, entclasses)]
        n = len(surgery.steps)

        for i, step in enumerate(surgery.steps):
            matched_current_ents = [
                (current_ent, current_entclasses)
                for current_ent, current_entclasses in current_ents
                if any(
                    step.type == entclass.id or step.type in entclass.ancestors
                    for entclass in current_entclasses
                )
            ]

            if len(matched_current_ents) == 0:
                if i > 0 and not surgery.optional:
                    self.logger.warning(
                        "The next step requires type {}, but entities {} isn't of that type.",
                        step.type,
                        ", ".join(
                            f"`{current_ent.label} ({current_ent.id})`"
                            for current_ent, _ in current_ents
                        ),
                    )
                    return [], "nodata"
                return [], "notapplicable"

            matched_current_ents = [
                (current_ent, current_entclasses)
                for current_ent, current_entclasses in matched_current_ents
                if step.prop in current_ent.props
            ]

            if len(matched_current_ents) == 0:
                if not surgery.optional:
                    self.logger.warning(
                        "Missing {} for {}",
                        step.prop,
                        ", ".join(
                            f"`{current_ent.label} ({current_ent.id})`"
                            for current_ent, _ in current_ents
                        ),
                    )
                return [], "nodata"

            potential_values = sorted(
                {
                    val
                    for current_ent, _ in matched_current_ents
                    for val in current_ent.get_object_prop_value(step.prop)
                }
            )
            if i == n - 1:
                return potential_values, "success"

            current_ents = [
                (
                    (ent := self.entities[eid]),
                    [self.classes[cid] for cid in ent.instance_of()],
                )
                for eid in potential_values
            ]

        raise Exception("Shouldn't reach here")
