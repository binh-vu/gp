from __future__ import annotations

import gp_core.models
from gp.entity_linking.candidate_generation.common import TableCanGenResult
from sm.dataset import Example, FullTable
from sm.inputs.column import Column
from sm.inputs.link import Link


def to_rust_table(
    ex: Example[FullTable], cans: TableCanGenResult
) -> gp_core.models.Table:
    def to_col(col: Column) -> gp_core.models.Column:
        values = []
        for v in col.values:
            if isinstance(v, str):
                values.append(v)
            elif v is None:
                values.append("")
            else:
                raise ValueError(f"Unsupported value type: {type(v)}")
        return gp_core.models.Column(col.index, col.clean_multiline_name, values)

    def to_links(ri: int, ci: int, links: list[Link]) -> list[gp_core.models.Link]:
        if cans.has_cell_candidates(ri, ci):
            cell_cans = cans.get_cell_candidates(ri, ci)
            candidates = [
                gp_core.models.CandidateEntityId(
                    gp_core.models.EntityId(cell_cans[i].id),
                    cell_cans[i].score,
                )
                for i in range(len(cell_cans))
            ]
        else:
            cell_cans = None
            candidates = []

        if len(links) == 0:
            if len(candidates) > 0:
                return [
                    gp_core.models.Link(
                        start=0,
                        end=len(ex.table.table[ri, ci]),
                        url=None,
                        entities=[],
                        candidates=candidates,
                    )
                ]
            return []

        return [
            gp_core.models.Link(
                start=0,
                end=len(ex.table.table[ri, ci]),
                url=None,
                entities=[
                    gp_core.models.EntityId(entid)
                    for entid in {entid for link in links for entid in link.entities}
                ],
                candidates=candidates,
            )
        ]

    return gp_core.models.Table(
        ex.table.table.table_id,
        [
            [to_links(ri, ci, links) for ci, links in enumerate(row)]
            for ri, row in enumerate(ex.table.links.data)
        ],
        [to_col(col) for col in ex.table.table.columns],
        gp_core.models.Context(
            None,
            None,
            [],
        ),
    )
