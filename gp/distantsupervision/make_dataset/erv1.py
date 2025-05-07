from dataclasses import dataclass

from sm.dataset import FullTable


@dataclass
class EntityRecognitionV1Args:
    min_link_freq: float = 0.7
    min_link_coverage: float = 0.9


class EntityRecognitionV1:
    VERSION = 100

    def __init__(self, args: EntityRecognitionV1Args):
        self.args = args

    def recognize(self, table: FullTable) -> list[int]:
        return [
            col.index
            for col in table.table.columns
            if self.is_entity_column(table, col.index)
        ]

    def is_entity_column(self, table: FullTable, ci: int) -> bool:
        cells = table.table.get_column_by_index(ci).values

        n_rows = len(cells)
        n_links = 0
        coverage_size = 0
        total_area = 0

        for ri, cell in enumerate(cells):
            links = table.links[ri, ci]
            if len(links) == 0:
                continue

            coverage_size += sum((link.end - link.start) for link in links)
            total_area += len(cell)

            n_links += 1

        return (
            n_links / n_rows >= self.args.min_link_freq
            and coverage_size / total_area >= self.args.min_link_coverage
        )
