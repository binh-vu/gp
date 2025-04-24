from __future__ import annotations

from dataclasses import dataclass, field

from gramsplus.core.algorithms import CanGraphExtractorConfig
from gramsplus.semanticmodeling.literal_matcher import LiteralMatcherConfig


@dataclass
class CanGraphExtractorCfg:
    literal_matcher: LiteralMatcherConfig = field(
        default_factory=LiteralMatcherConfig,
        metadata={"help": "Configuration for the literal matcher"},
    )
    n_hop: int = field(
        default=1,
        metadata={
            "help": "The number of hops to discover relationships between entities"
        },
    )
    ignored_columns: list[int] = field(
        default_factory=list,
        metadata={
            "help": "The columns to ignore and not used during **literal** matching"
        },
    )
    ignored_props: set[str] = field(
        default_factory=set,
        metadata={
            "help": "The properties to ignore and not used during **literal** matching"
        },
    )
    allow_same_ent_search: bool = field(
        default=False,
        metadata={
            "help": "whether we try to discover relationships between the same entity in same row but different columns"
        },
    )
    allow_ent_matching: bool = field(
        default=True,
        metadata={
            "help": "whether to match relationships between entities (i.e., if false only do literal matching)"
        },
    )
    use_context: bool = field(
        default=False,
        metadata={"help": "whether we use context entities in the search"},
    )
    add_missing_property: bool = field(
        default=True,
        metadata={
            "help": "whether to add a new node to represent a statement's property value if the statement has both qualifier and property pointing to the same target node to give the qualifier a chance to be selected."
        },
    )
    run_subproperty_inference: bool = field(
        default=False,
        metadata={
            "help": "whether to run subproperty inference to complete missing links"
        },
    )
    run_transitive_inference: bool = field(
        default=True,
        metadata={
            "help": "whether to run transitive inference to complete missing links"
        },
    )
    deterministic_order: bool = field(
        default=False,
        metadata={
            "help": "whether to sort the matched results so that the order is always deterministic"
        },
    )
    correct_entity_threshold: float = field(
        default=0.8,
        metadata={
            "help": "the threshold that a candidate entity is considered correct for contradicted information detection"
        },
    )
    validate: bool = field(
        default=False,
        metadata={
            "help": "whether to validate the result candidate graph. This is useful for debugging"
        },
    )

    def to_rust(self) -> CanGraphExtractorConfig:
        return CanGraphExtractorConfig(
            self.literal_matcher.to_rust(),
            self.ignored_columns,
            self.ignored_props,
            self.allow_same_ent_search,
            self.allow_ent_matching,
            self.use_context,
            self.add_missing_property,
            self.run_subproperty_inference,
            self.run_transitive_inference,
            self.deterministic_order,
            self.correct_entity_threshold,
            self.validate,
            self.n_hop,
        )
