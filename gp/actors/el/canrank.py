from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from gp.actors.data.prelude import GPExample
from gp.actors.el.cangen import CanGenActor
from gp.entity_linking.cangen.common import TableCanGenResult
from gp.entity_linking.canrank.common import CanRankMethod, TableCanGenUpdateScores
from gp.misc.appconfig import AppConfig
from gp.misc.evaluation.el_osin_mixin import EntityLinkingOsinActorMixin
from ream.actors.base import BaseActor
from ream.cache_helper import Cache, MemBackend
from ream.helper import import_attr
from ream.params_helper import PluginParams


@dataclass
class CanRankActorArgs(PluginParams):
    pass


class CanRankActor(
    EntityLinkingOsinActorMixin[CanRankActorArgs], BaseActor[CanRankActorArgs]
):
    """Rank candidate entities for cells in a table."""

    VERSION = 103

    EXP_NAME = "Candidate Ranking"
    EXP_VERSION = 3

    def __init__(
        self,
        params: CanRankActorArgs,
        cangen_actor: CanGenActor,
    ):
        super().__init__(params, dep_actors=[cangen_actor])

        self.cangen_actor = cangen_actor
        self.db_actor = cangen_actor.db_actor

    def __call__(self, example: GPExample) -> TableCanGenResult:
        cans = self.cangen_actor(example)
        cans = cans.shallow_clone()
        cans.ent_score = self.impl_call(example).value
        return cans

    def batch_call(
        self, examples: Sequence[GPExample], verbose: bool = False
    ) -> list[TableCanGenResult]:
        ex_cans = self.cangen_actor.batch_call(examples)
        ex_cans_scores = self.impl_batch_call(examples, verbose)
        output = []
        for ei in range(len(examples)):
            cans = ex_cans[ei].shallow_clone()
            cans.ent_score = ex_cans_scores[ei].value
            output.append(cans)
        return output

    @Cache.cache(
        backend=Cache.sqlite.serde(
            cls=TableCanGenUpdateScores, filename="canrank", mem_persist=True
        ),
        cache_key=lambda self, example: example.id,
        disable=lambda self: not AppConfig.get_instance().is_cache_enable,
    )
    def impl_call(self, example: GPExample) -> TableCanGenUpdateScores:
        return self.get_method().batch_rank_candidates([example])[0]

    @Cache.flat_cache(
        backend=Cache.sqlite.serde(
            cls=TableCanGenUpdateScores, filename="canrank", mem_persist=True
        ),
        cache_key=lambda self, example, verbose=False: example.id,
        disable=lambda self: not AppConfig.get_instance().is_cache_enable,
    )
    def impl_batch_call(
        self, examples: Sequence[GPExample], verbose: bool = False
    ) -> list[TableCanGenUpdateScores]:
        return self.get_method().batch_rank_candidates(examples, verbose)

    @Cache.cache(backend=MemBackend())
    def get_method(self) -> CanRankMethod:
        if isinstance(self.params.clspath, str):
            cls = import_attr(self.params.clspath)
        else:
            cls = self.params.clspath

        if isinstance(self.params.clsargs, dict):
            return cls(**self.params.clsargs, cangen_actor=self.cangen_actor)

        return cls(self.params.clsargs, cangen_actor=self.cangen_actor)

    def get_candidate_entities(
        self, examples: list[GPExample]
    ) -> list[TableCanGenResult]:
        return self.batch_call(examples)
