from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Mapping,
    Optional,
    Sequence,
    TypeAlias,
    TypedDict,
    Union,
)

from gp.misc.evaluation.sm_wikidata import SemanticModelHelper
from gp_core.models import GramsDB, LocalGramsDB, RemoteGramsDB
from kgdata.db import GenericDB
from kgdata.models import Ontology
from kgdata.wikidata.db import WikidataDB
from libactor.actor import Actor
from libactor.cache import IdentObj
from sm.misc.prelude import import_attr
from sm.misc.ray_helper import get_instance
from sm.namespaces.namespace import KnowledgeGraphNamespace
from sm.namespaces.utils import KGName, get_kgns


@dataclass
class KGDBArgs:
    name: KGName
    version: str
    datadir: Path
    clspath: Optional[str] = None
    entity_url: Optional[str] = None
    entity_metadata_url: Optional[str] = None
    entity_batch_size: int = 64
    entity_metadata_batch_size: int = 128

    def to_dict(self):
        return {"name": self.name, "version": self.version}

    def get_entity_urls(self) -> list[str]:
        return self.get_urls(self.entity_url)

    def get_entity_metadata_urls(self) -> list[str]:
        return self.get_urls(self.entity_metadata_url)

    def get_urls(self, pattern: Optional[str]) -> list[str]:
        if pattern is None:
            return []
        ((begin, end),) = re.findall(r"(\d+)-(\d+)", pattern)
        template = pattern.replace(f"{begin}-{end}", "{}")
        return [template.format(i) for i in range(int(begin), int(end))]

    def get_key(self):
        return f"{self.name}:{self.version}:{self.datadir}"


@dataclass
class DBActorArgs:
    kgdbs: Sequence[KGDBArgs]

    def __post_init__(self):
        self.kgdbs = sorted(self.kgdbs, key=lambda db: db.name)
        assert len(set(db.name for db in self.kgdbs)) == len(
            self.kgdbs
        ), "Should not have duplicated KGs"

    def to_dict(self):
        return [{"name": kgdb.name, "version": kgdb.version} for kgdb in self.kgdbs]


class DBActor(Actor[DBActorArgs]):
    VERSION = 100

    def __init__(self, params: DBActorArgs):
        super().__init__(params, [])
        self.kgdbs: Mapping[KGName, KGDB] = {
            kgdb.name: KGDB.get_instance(kgdb) for kgdb in params.kgdbs
        }


PyKGDB: TypeAlias = Union[GenericDB, WikidataDB]


@dataclass
class KGDB:
    args: KGDBArgs

    @cached_property
    def kgns(self) -> KnowledgeGraphNamespace:
        return get_kgns(self.args.name)

    @property
    def kgname(self):
        return self.args.name

    @cached_property
    def pydb(self) -> PyKGDB:
        if self.args.name == KGName.Wikidata:
            return WikidataDB(self.args.datadir)
        if self.args.name == KGName.DBpedia:
            return GenericDB(self.args.datadir)
        if self.args.name == KGName.Generic:
            if self.args.clspath is not None:
                cls = import_attr(self.args.clspath)
            else:
                cls = GenericDB
            return cls(self.args.datadir)
        raise NotImplementedError(self.args.name)

    @cached_property
    def rudb(self) -> GramsDB:
        if self.args.entity_url is None:
            return LocalGramsDB(str(self.args.datadir))
        else:
            return RemoteGramsDB(
                str(self.args.datadir),
                self.args.get_entity_urls(),
                self.args.get_entity_metadata_urls(),
                self.args.entity_batch_size,
                self.args.entity_metadata_batch_size,
            )

    @cached_property
    def sm_helper(self):
        return SemanticModelHelper(
            self.pydb.entity_labels.cache(),
            self.pydb.props.cache(),
            self.kgns,
        )

    @cached_property
    def ontology(self) -> IdentObj[Ontology]:
        return IdentObj(
            key=f"ontology:{self.args.name}:{self.args.version}",
            value=Ontology(
                kgname=self.kgname,
                kgns=self.kgns,
                classes=self.pydb.classes.cache(),
                props=self.pydb.props.cache(),
            ),
        )

    @staticmethod
    def get_instance(db: KGDB | KGDBArgs) -> KGDB:
        if isinstance(db, KGDB):
            return db
        return get_instance(
            lambda: KGDB(db), f"kgdb:{db.name}:{db.version}:{db.datadir}"
        )
