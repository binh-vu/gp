from __future__ import annotations

from gp.actors.data import KGDB, GPExample
from gp.entity_linking.candidate_generation.common import (
    CanEnt,
    CanGenMethod,
    TableCanGenResult,
)
from sm.inputs.prelude import Link


class CanGenOracleMethod(CanGenMethod):
    VERSION = 100

    def __init__(self, impute_score: float = 1.0):
        self.impute_score = impute_score

    def get_candidates(
        self,
        examples: list[GPExample],
        entity_columns: list[list[int]],
    ) -> list[TableCanGenResult]:
        output = []

        for ei, ex in enumerate(examples):
            exentcols = set(entity_columns[ei])
            output.append(
                TableCanGenResult.from_matrix(
                    ex.table.links.map_with_index(
                        lambda ri, ci, links: (
                            map_links(links, self.impute_score)
                            if ci in exentcols
                            else []
                        )
                    )
                )
            )

        return output


def map_links(links: list[Link], impute_score: float) -> list[CanEnt]:
    gold_ents = {}
    for link in links:
        gold_ents.update((str(entid), impute_score) for entid in link.entities)
    return [CanEnt(entid, score) for entid, score in gold_ents.items()]
