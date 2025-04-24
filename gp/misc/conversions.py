from __future__ import annotations

import gp_core as gcore
from sm.dataset import Example, FullTable
from sm.inputs.column import Column
from sm.inputs.link import Link

from gp.el.candidate_generation.cangen_model import TableCandidateEntities


def to_rust_table(
    ex: Example[FullTable], cans: TableCandidateEntities
) -> gcore.models.Table:
    def to_col(col: Column) -> gcore.models.Column:
        values = []
        for v in col.values:
            if isinstance(v, str):
                values.append(v)
            elif v is None:
                values.append("")
            else:
                raise ValueError(f"Unsupported value type: {type(v)}")
        return gcore.models.Column(col.index, col.clean_multiline_name, values)

    def to_links(ri: int, ci: int, links: list[Link]) -> list[gcore.models.Link]:
        if cans.has_cell_candidates(ri, ci):
            cell_cans = cans.get_cell_candidates(ri, ci)
            candidates = [
                gcore.models.CandidateEntityId(
                    gcore.models.EntityId(cell_cans.id[i]),
                    cell_cans.score[i],
                )
                for i in range(len(cell_cans))
            ]
        else:
            cell_cans = None
            candidates = []

        if len(links) == 0:
            if len(candidates) > 0:
                return [
                    gcore.models.Link(
                        start=0,
                        end=len(ex.table.table[ri, ci]),
                        url=None,
                        entities=[],
                        candidates=candidates,
                    )
                ]
            return []

        return [
            gcore.models.Link(
                start=0,
                end=len(ex.table.table[ri, ci]),
                url=None,
                entities=[
                    gcore.models.EntityId(entid)
                    for entid in {entid for link in links for entid in link.entities}
                ],
                candidates=candidates,
            )
        ]

    return gcore.models.Table(
        ex.table.table.table_id,
        [
            [to_links(ri, ci, links) for ci, links in enumerate(row)]
            for ri, row in enumerate(ex.table.links.data)
        ],
        [to_col(col) for col in ex.table.table.columns],
        gcore.models.Context(
            None,
            None,
            [],
        ),
    )
    )
