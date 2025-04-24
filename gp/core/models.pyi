import enum
from typing import Optional, TypeAlias

from kgdata_core.models import EntityMetadata, Property, StatementView, Value, ValueView

from gp_core.literal_matchers import ParsedTextRepr

class LocalGramsDB:
    def __init__(self, datadir: str) -> None: ...
    @staticmethod
    def init(datadir: str) -> None: ...
    @staticmethod
    def get_instance() -> LocalGramsDB: ...
    def get_algo_context(
        self, table: Table, n_hop: int, parallel: bool
    ) -> AlgoContext: ...
    def get_property(self, id: str) -> Property: ...
    def has_entity(self, id: str) -> bool: ...
    def get_redirected_entity_id(self, id: str) -> Optional[str]: ...
    def get_entity_metadata(self, id: str) -> EntityMetadata: ...
    def get_entity_pagerank(self, id: str) -> float: ...

class RemoteGramsDB:
    def __init__(
        self,
        datadir: str,
        entity_urls: list[str],
        entity_metadata_urls: list[str],
        entity_batch_size: int,
        entity_metadata_batch_size: int,
    ) -> None: ...
    @staticmethod
    def init(
        datadir: str,
        entity_urls: list[str],
        entity_metadata_urls: list[str],
        entity_batch_size: int,
        entity_metadata_batch_size: int,
    ) -> None: ...
    @staticmethod
    def get_instance() -> LocalGramsDB: ...
    def get_algo_context(
        self, table: Table, n_hop: int, parallel: bool
    ) -> AlgoContext: ...
    def get_property(self, id: str) -> Property: ...
    def has_entity(self, id: str) -> bool: ...
    def get_redirected_entity_id(self, id: str) -> Optional[str]: ...
    def get_entity_metadata(self, id: str) -> EntityMetadata: ...
    def get_entity_pagerank(self, id: str) -> float: ...

GramsDB: TypeAlias = LocalGramsDB | RemoteGramsDB

class AlgoContext:
    def get_entity_statement(
        self, entity_id: str, prop: str, stmt_index: int
    ) -> StatementView: ...

class Table:
    id: str
    links: list[list[list[Link]]]
    columns: list[Column]
    context: Context

    def __init__(
        self,
        id: str,
        links: list[list[list[Link]]],
        columns: list[Column],
        context: Context,
    ) -> None: ...
    def get_links(self, row: int, col: int) -> list[Link]: ...
    def save(self, outfile: str) -> None: ...

class TableCells:
    cells: list[list[ParsedTextRepr]]

    def __init__(self, cell: list[list[ParsedTextRepr]]) -> None: ...
    def save(self, outfile: str) -> None: ...

class Link:
    start: int
    end: int
    url: Optional[str]
    entities: list[EntityId]
    candidates: list[CandidateEntityId]

    def __init__(
        self,
        start: int,
        end: int,
        url: Optional[str],
        entities: list[EntityId],
        candidates: list[CandidateEntityId],
    ) -> None: ...

class Column:
    index: int
    name: Optional[str]
    values: list[str]

    def __init__(self, index: int, name: Optional[str], values: list[str]) -> None: ...

class Context:
    page_title: Optional[str]
    page_url: Optional[str]
    page_entities: Optional[list[EntityId]]

    def __init__(
        self,
        page_title: Optional[str],
        page_url: Optional[str],
        page_entities: Optional[list[EntityId]],
    ) -> None: ...

class CandidateEntityId:
    id: EntityId
    probability: float

    def __init__(self, id: EntityId, probability: float) -> None: ...

class EntityId(tuple[str]):
    def __init__(self, id: str) -> None: ...
    @property
    def id(self) -> str: ...

class MatchMethod(enum.Enum):
    LiteralMatching = enum.auto()
    LinkMatching = enum.auto()

class CGNode:
    def __init__(self): ...
    def id(self) -> int: ...
    def is_column(self) -> bool: ...
    def is_statement(self) -> bool: ...
    def is_entity(self) -> bool: ...
    def is_literal(self) -> bool: ...
    def try_as_column(self) -> Optional[CGColumnNode]: ...
    def try_as_statement(self) -> Optional[CGStatementNode]: ...
    def try_as_entity(self) -> Optional[CGEntityNode]: ...
    def try_as_literal(self) -> Optional[CGLiteralNode]: ...

class CGColumnNode:
    id: int
    label: str
    column: int

class CGStatementNode:
    id: int

class CGEntityNode:
    id: int
    entity_id: str
    entity_prob: float
    is_in_context: bool

class CGLiteralNode:
    id: int
    value: ValueView
    value_prob: float
    is_in_context: bool

__all__ = [
    "LocalGramsDB",
    "AlgoContext",
    "Table",
    "TableCells",
    "Link",
    "Column",
    "Context",
    "EntityId",
    "CandidateEntityId",
    "Value",
    "MatchMethod",
    "CGNode",
    "CGColumnNode",
]
]
