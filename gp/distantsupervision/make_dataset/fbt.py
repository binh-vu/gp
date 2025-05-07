from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from gp.distantsupervision.make_dataset.utils import FilterMixin, StrPath
from kgdata.models.entity import Entity
from kgdata.models.ont_class import OntologyClass
from kgdata.wikidata.models.wdclass import WDClass
from kgdata.wikidata.models.wdentity import WDEntity
from sm.dataset import FullTable

# fmt: off
DEFAULT_IGNORE_ENTS = {
    "Q13406463"  # wikimedia list article
}
# fmt: on


@dataclass
class FilterByEntTypeArgs:
    ignore_types: set[str] = field(
        default_factory=lambda: DEFAULT_IGNORE_ENTS,
        metadata={"help": "Entities belong to these types will be ignored"},
    )
    noisy_threshold: float = 0.5


class FilterByEntType(FilterMixin):
    VERSION = 100

    def __init__(
        self,
        args: FilterByEntTypeArgs,
        entities: Mapping[str, WDEntity | Entity],
        classes: Mapping[str, OntologyClass],
        log_file: Optional[StrPath] = None,
    ):
        super().__init__(log_file)
        self.args = args
        self.entities = entities
        self.classes = classes

    def is_noisy(self, table: FullTable, ci: int) -> tuple[bool, str]:
        nrows, ncols = table.table.shape()

        n_noisy_cells = 0
        for ri in range(nrows):
            if len(table.links[ri, ci]) == 0:
                continue
            entids = {
                str(entid) for link in table.links[ri, ci] for entid in link.entities
            }
            if any(self.has_ignore_type(eid) for eid in entids):
                n_noisy_cells += 1

        noisy_percentage = n_noisy_cells / nrows
        return (
            noisy_percentage >= self.args.noisy_threshold,
            f"{noisy_percentage:.03f} cells have ignore type",
        )

    def has_ignore_type(self, entid: str):
        ignore_types = self.args.ignore_types
        for cid in self.entities[entid].instance_of():
            if cid in ignore_types or not ignore_types.isdisjoint(
                self.classes[cid].ancestors
            ):
                return True
        return False
