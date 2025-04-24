from __future__ import annotations

from dataclasses import dataclass, field

from gp.entity_linking.candidate_recognition import ICanReg, OracleCanReg
from kgdata.models import Ontology
from libactor.actor import Actor
from libactor.cache import BackendFactory, IdentObj, cache
from sm.dataset import Example, FullTable
from sm.misc.funcs import import_attr


@dataclass
class CanRegActorArgs:
    clspath: str | type
    clsargs: dict | object = field(default_factory=dict)


class CanRegActor(Actor[CanRegActorArgs]):
    VERSION = 103

    def forward(
        self, input: IdentObj[Example[FullTable]], ontology: IdentObj[Ontology]
    ) -> IdentObj[list[int]]:
        ent_cols = self.invoke(input, ontology)
        return IdentObj(str(ent_cols), ent_cols)

    @cache(
        backend=BackendFactory.actor.sqlite.pickle(mem_persist=True),
    )
    def invoke(
        self, example: IdentObj[Example[FullTable]], ontology: IdentObj[Ontology]
    ) -> list[int]:
        return self.get_method()([example.value], ontology.value)[0]

    def is_oracle_entity_linking(self) -> bool:
        return isinstance(self.get_method(), OracleCanReg)

    @cache(backend=BackendFactory.actor.mem)
    def get_method(self) -> ICanReg:
        if isinstance(self.params.clspath, str):
            cls = import_attr(self.params.clspath)
        else:
            cls = self.params.clspath
        if isinstance(self.params.clsargs, dict):
            return cls(**self.params.clsargs)
        if isinstance(self.params.clsargs, tuple):
            return cls(*self.params.clsargs)
        return cls(self.params.clsargs)
