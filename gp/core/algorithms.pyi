from __future__ import annotations

from typing import Optional, Union

import polars as pl
from gp_core.literal_matchers import LiteralMatcher, LiteralMatcherConfig
from gp_core.models import (
    AlgoContext,
    CGNode,
    GramsDB,
    LocalGramsDB,
    MatchMethod,
    RemoteGramsDB,
    Table,
    TableCells,
)
from kgdata_core.base import RustVecView

def candidate_local_search(
    table: Table,
    context: Optional[AlgoContext],
    cfg: CandidateLocalSearchConfig,
    parallel: bool,
) -> Table: ...
def par_candidate_local_search(
    db: GramsDB,
    tables: list[Table],
    contexts: Optional[list[AlgoContext]],
    cfg: CandidateLocalSearchConfig,
) -> list[Table]: ...
def matching(
    table: Table,
    table_cells: TableCells,
    context: AlgoContext,
    literal_matcher: LiteralMatcher,
    ignored_columns: list[str],
    ignored_props: list[str],
    allow_same_ent_search: bool = False,
    allow_ent_matching: bool = True,
    use_context: bool = True,
    deterministic_order: bool = True,
    parallel: bool = False,
) -> DataMatchesResult: ...
def par_matching(
    db: GramsDB,
    tables: list[Table],
    table_cells: list[TableCells],
    contexts: Optional[list[AlgoContext]],
    literal_matcher: LiteralMatcher,
    ignored_columns: list[list[str]],
    ignored_props: list[str],
    allow_same_ent_search: bool = False,
    allow_ent_matching: bool = True,
    use_context: bool = True,
    deterministic_order: bool = True,
    verbose: bool = False,
) -> list[DataMatchesResult]: ...
def extract_cangraph(
    table: Table,
    cells: TableCells,
    db: GramsDB,
    cfg: CanGraphExtractorConfig,
    context: Optional[AlgoContext] = None,
    parallel: bool = False,
) -> CanGraphExtractedResult: ...
def par_extract_cangraphs(
    tables: list[Table],
    cells: list[TableCells],
    db: GramsDB,
    cfg: CanGraphExtractorConfig,
    context: Optional[list[AlgoContext]] = None,
    verbose: bool = False,
) -> list[CanGraphExtractedResult]: ...
def extract_candidate_entity_link_freqs(
    db: GramsDB,
    table: Table,
    table_cells: TableCells,
    contexts: Optional[AlgoContext],
    literal_matcher: LiteralMatcher,
    ignored_columns: list[str],
    ignored_props: list[str],
    allow_same_ent_search: bool = False,
    allow_ent_matching: bool = True,
    use_context: bool = True,
    deterministic_order: bool = True,
    parallel: bool = False,
) -> list[dict[str, list[int]]]: ...
def par_extract_candidate_entity_link_freqs(
    db: GramsDB,
    tables: list[Table],
    table_cells: list[TableCells],
    contexts: Optional[list[AlgoContext]],
    literal_matcher: LiteralMatcher,
    ignored_columns: list[list[str]],
    ignored_props: list[str],
    allow_same_ent_search: bool = False,
    allow_ent_matching: bool = True,
    use_context: bool = True,
    deterministic_order: bool = True,
    verbose: bool = False,
) -> list[list[dict[str, list[int]]]]: ...

class CandidateLocalSearchConfig:
    strsim: str
    threshold: float
    use_column_name: bool
    use_language: Optional[str]
    search_all_columns: bool

    def __init__(
        self,
        strsim: str,
        threshold: float,
        use_column_name: bool,
        use_language: Optional[str],
        search_all_columns: bool,
    ) -> None: ...

class CanGraphExtractorConfig:
    literal_matcher_config: LiteralMatcherConfig
    ignored_columns: list[int]
    ignored_props: set[str]
    allow_same_ent_search: bool
    allow_ent_matching: bool
    use_context: bool
    add_missing_property: bool
    run_subproperty_inference: bool
    run_transitive_inference: bool
    deterministic_order: bool
    correct_entity_threshold: float
    validate: bool
    n_hop: int

    def __init__(
        self,
        literal_matcher_config: LiteralMatcherConfig,
        ignored_columns: list[int],
        ignored_props: set[str],
        allow_same_ent_search: bool,
        allow_ent_matching: bool,
        use_context: bool,
        add_missing_property: bool,
        run_subproperty_inference: bool,
        run_transitive_inference: bool,
        deterministic_order: bool,
        correct_entity_threshold: float,
        validate: bool,
        n_hop: int,
    ): ...
    def save(self, outfile: str) -> None: ...

class CanGraphExtractedResult:
    nodes: list[CGNode]
    edges: list[str]
    edgedf: pl.DataFrame

class DataMatchesResult:
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> DataMatchesResult: ...
    def get_n_nodes(self) -> int: ...
    def edges(self) -> RustVecView[PotentialRelationships]: ...
    def is_cell_node(self, idx: int) -> bool: ...
    def get_cell_node(self, idx: int) -> CellNode: ...
    def is_entity_node(self, idx: int) -> bool: ...
    def get_entity_node(self, idx: int) -> EntityNode: ...

class CellNode:
    @property
    def col(self) -> int: ...
    @property
    def row(self) -> int: ...

class EntityNode:
    @property
    def entity_id(self) -> str: ...
    @property
    def entity_prob(self) -> float: ...

class PotentialRelationships:
    @property
    def source_id(self) -> int: ...
    @property
    def target_id(self) -> int: ...
    @property
    def rels(self) -> RustVecView[MatchedEntRel]: ...

class MatchedEntRel:
    @property
    def source_entity_id(self) -> str: ...
    @property
    def statements(self) -> RustVecView[MatchedStatement]: ...
    def get_matched_target_entities(self, context: AlgoContext) -> list[str]: ...

PP = property

class MatchedStatement:
    @PP
    def property(self) -> str: ...
    @PP
    def statement_index(self) -> int: ...
    @PP
    def matched_property(self) -> Optional[Match]: ...
    @PP
    def matched_qualifiers(self) -> RustVecView[MatchedQualifier]: ...

class MatchedQualifier:
    @property
    def qualifier(self) -> str: ...
    @property
    def qualifier_index(self) -> int: ...
    @property
    def score(self) -> Match: ...

class Match:
    @property
    def prob(self) -> float: ...
    @property
    def method(self) -> MatchMethod: ...
    def method(self) -> MatchMethod: ...
