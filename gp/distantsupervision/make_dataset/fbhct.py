import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import serde.json
import serde.yaml
from gp.distantsupervision.make_dataset.interface import LabelFn
from gp.distantsupervision.make_dataset.lv1 import LabelV1
from gp.distantsupervision.make_dataset.utils import FilterMixin, StrPath
from kgdata.models.ont_class import OntologyClass
from sm.dataset import FullTable

DEFAULT_WHITELIST_FILENAME = "whitelist.yml"


@dataclass
class FilterByHeaderColTypeArgs:
    whitelist_file: Path


class FilterByHeaderColType(FilterMixin):
    """This function is deprecated. Use a labeler that combines the whitelist instead"""

    VERSION = 101

    def __init__(
        self,
        args: FilterByHeaderColTypeArgs,
        labeler: LabelFn,
        log_file: Optional[StrPath] = None,
    ):
        super().__init__(log_file)
        # mapping from class id to list of allowed column names
        self.whitelist: dict[str, set[str]] = self.parse_whitelist(args.whitelist_file)
        self.whitelist_keys = set(self.whitelist.keys())
        assert isinstance(labeler, LabelV1), f"Expected LabelV1, got {type(labeler)}"
        self.labeler = labeler
        self.classes = labeler.classes

    def is_noisy(self, table: FullTable, ci: int) -> tuple[bool, str]:
        header = table.table.get_column_by_index(ci).clean_multiline_name
        if header is None:
            return True, "no header"
        col_types = self.labeler.label_column(table.links[:, ci])
        for col_type in col_types:
            if col_type.id in self.whitelist_keys:
                if header in self.whitelist[col_type.id]:
                    return False, ""
            else:
                cls = self.classes[col_type.id]
                common_ancestors = self.whitelist_keys.intersection(
                    cls.ancestors.keys()
                )
                if any(
                    header in self.whitelist[ancestor] for ancestor in common_ancestors
                ):
                    return False, ""
        return (
            True,
            f"Column {header} with candidate types {[f'{self.classes[ctype.id].label} ({ctype.id})' for ctype in col_types]} not in whitelist",
        )

    def parse_whitelist(self, infile: Path):
        whitelist: dict[str, set[str]] = {}
        label_parser = re.compile(r"[^(]+ \((Q\d+)\)")
        if infile.suffix in {".yml", ".yaml"}:
            label2headers = serde.yaml.deser(infile)
        elif infile.suffix in {".json"}:
            label2headers = serde.json.deser(infile)
        else:
            raise ValueError(f"Unknown whitelist file format: {infile.suffix}")

        for label, headers in label2headers.items():
            m = label_parser.match(label)
            assert m is not None
            class_id = m.group(1)
            whitelist[class_id] = set(headers)

        return whitelist
