from __future__ import annotations

from dataclasses import dataclass, field

import gp_core
import numpy as np
from gp.actors.data import KGDB, KGDBArgs
from gp.entity_linking.candidate_generation.common import populate
from gp.misc.conversions import to_rust_table
from loguru import logger
from sm.dataset import Example, FullTable
from sm.misc.ray_helper import enhance_error_info, ray_map, ray_put
from timer import watch_and_report


@dataclass
class LocalSearchArgs:
    similarity: str = field(
        default="levenshtein",
        metadata={
            "help": "augment candidate entities discovered from matching entity properties with values in the same row. "
            "Using the similarity function and threshold to filter out irrelevant candidates. "
        },
    )
    threshold: float = field(
        default=2.0,
        metadata={
            "help": "add candidate entities discovered from matching entity properties with values in the same row. "
            "Any value greater than 1.0 mean we do not apply the similarity function"
        },
    )
    use_column_name: bool = field(
        default=False,
        metadata={
            "help": "add column name to the search text. This is useful for value such as X town, Y town, Z town. "
        },
    )
    search_all_columns: bool = field(
        default=False,
        metadata={"help": "search all columns instead of just the entity columns."},
    )


class LocalSearch:
    VERSION = 101

    def __init__(self, args: LocalSearchArgs, db: KGDB):
        self.args = args
        self.db = db

    def augment_candidates(
        self, examples: list[Example[FullTable]], candidates: DatasetCandidateEntities
    ):
        cfg = gp_core.algorithms.CandidateLocalSearchConfig(
            self.args.similarity,
            self.args.threshold,
            self.args.use_column_name,
            None,
            self.args.search_all_columns,
        )

        logger.debug("prepare localsearch inputs...")
        tables = []
        table_cans = []
        for ex in examples:
            tcans = candidates.get_table_candidates(ex.table.table.table_id)
            table = to_rust_table(ex, tcans)
            tables.append(table)
            table_cans.append(tcans)

        logger.debug(
            "prepare localsearch inputs... done! running candidate local search..."
        )
        newtables = gcore.algorithms.par_candidate_local_search(
            self.db.rudb, tables, None, cfg
        )
        logger.debug("running candidate local search... done!")

        return DatasetCandidateEntities.from_table_candidates(
            {
                ex.table.table.table_id: extract_candidates_from_rust_table(
                    self.db, table_cans[ei], ex.table, newtables[ei]
                )
                for ei, ex in enumerate(examples)
            }
        )

    def augment_candidates_ray(
        self, examples: list[Example[FullTable]], candidates: DatasetCandidateEntities
    ):
        using_ray = len(examples) > 1
        dbref = ray_put(self.db.args, using_ray)
        paramref = ray_put(self.args, using_ray)

        table_cans = ray_map(
            rust_augment_candidates,
            [
                (
                    dbref,
                    ex,
                    candidates.get_table_candidates(ex.table.table.table_id),
                    paramref,
                    not using_ray,
                )
                for ex in examples
            ],
            verbose=True,
            desc="augmenting candidates",
            using_ray=using_ray,
            is_func_remote=False,
            auto_shutdown=True,
        )
        return DatasetCandidateEntities.from_table_candidates(
            {ex.table.table.table_id: tcans for ex, tcans in zip(examples, table_cans)}
        )


@enhance_error_info("1.table.table.table_id")
def rust_augment_candidates(
    kgdb: KGDB | KGDBArgs,
    example: Example[FullTable],
    candidates: TableCandidateEntities,
    params: LocalSearchArgs,
    verbose: bool,
):
    kgdb = KGDB.get_instance(kgdb)
    cfg = gcore.algorithms.CandidateLocalSearchConfig(
        params.similarity,
        params.threshold,
        params.use_column_name,
        None,
        params.search_all_columns,
    )

    newtable = to_rust_table(example, candidates)
    with watch_and_report(
        "create algorithm context",
        preprint=True,
        print_fn=logger.debug,
        disable=not verbose,
    ):
        context = kgdb.rudb.get_algo_context(newtable, n_hop=1, parallel=False)
    with watch_and_report(
        "Performing local search",
        preprint=True,
        print_fn=logger.debug,
        disable=not verbose,
    ):
        newtable = gcore.algorithms.candidate_local_search(
            newtable, context, cfg, False
        )

    return extract_candidates_from_rust_table(kgdb, candidates, example.table, newtable)


def extract_candidates_from_rust_table(
    kgdb: KGDB,
    candidates: TableCandidateEntities,
    pytable: FullTable,
    rutable: gcore.models.Table,
) -> TableCandidateEntities:
    nrows, ncols = pytable.table.shape()
    newcans = {}
    for ci in range(ncols):
        newcans[ci] = {}
        for ri in range(nrows):
            tmp_links = rutable.get_links(ri, ci)
            assert len(tmp_links) <= 1
            if len(tmp_links) == 0:
                assert (
                    not candidates.has_cell_candidates(ri, ci)
                    or len(candidates.get_cell_candidates(ri, ci)) == 0
                )
                continue

            tmp_link = tmp_links[0]
            if not candidates.has_cell_candidates(ri, ci):
                ids = [c.id.id for c in tmp_link.candidates]
                # print("row", ri, "col", ci, "new ids", ids)

                ids, labels, descs, aliases, popularity = populate(kgdb, ids)

                newcans[ci][ri] = CellCandidateEntities(
                    index=None,
                    id=np.array(ids, dtype=np.object_),
                    label=np.array(labels, dtype=np.object_),
                    description=np.array(descs, dtype=np.object_),
                    aliases=np.array(aliases, dtype=np.object_),
                    popularity=np.array(popularity, dtype=np.float64),
                    score=np.array(
                        [c.probability for c in tmp_link.candidates],
                        dtype=np.float64,
                    ),
                    provenance=np.array(["local_search"] * len(ids), dtype=np.object_),
                )
            else:
                cell_cans = candidates.get_cell_candidates(ri, ci)
                existing_ids = set(cell_cans.id)

                new_ids = []
                new_scores = []
                for c in tmp_link.candidates:
                    if c.id.id not in existing_ids:
                        new_ids.append(c.id.id)
                        new_scores.append(c.probability)

                if len(new_ids) > 0:
                    # print("row", ri, "col", ci, "new ids", new_ids)
                    new_ids, labels, descs, aliases, popularity = populate(
                        kgdb, new_ids
                    )
                    newcans[ci][ri] = CellCandidateEntities(
                        index=None,
                        id=np.concatenate(
                            [cell_cans.id, np.array(new_ids, dtype=np.object_)]
                        ),
                        label=np.concatenate(
                            [cell_cans.label, np.array(labels, dtype=np.object_)]
                        ),
                        description=np.concatenate(
                            [
                                cell_cans.description,
                                np.array(descs, dtype=np.object_),
                            ]
                        ),
                        aliases=np.concatenate(
                            [cell_cans.aliases, np.array(aliases, dtype=np.object_)]
                        ),
                        popularity=np.concatenate(
                            [
                                cell_cans.popularity,
                                np.array(popularity, dtype=np.float64),
                            ]
                        ),
                        score=np.concatenate(
                            [cell_cans.score, np.array(new_scores, dtype=np.float64)]
                        ),
                        provenance=np.concatenate(
                            [
                                cell_cans.provenance,
                                np.array(
                                    ["local_search"] * len(new_ids),
                                    dtype=np.object_,
                                ),
                            ]
                        ),
                    )
                else:
                    newcans[ci][ri] = cell_cans

    return TableCandidateEntities.from_cell_candidates(newcans)
