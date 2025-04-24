from dataclasses import dataclass, field

import gramsplus.core.literal_matchers as gcore_matcher


@dataclass
class LiteralMatcherConfig:
    """Configuration for literal matcher functions"""

    STRING: str = field(
        default=".string_exact_test", metadata={"help": "list of functions "}
    )
    QUANTITY: str = field(default=".quantity_test", metadata={"help": ""})
    GLOBECOORDINATE: str = field(default=".globecoordinate_test", metadata={"help": ""})
    TIME: str = field(default=".time_test", metadata={"help": ""})
    MONOLINGUAL_TEXT: str = field(
        default=".monolingual_exact_test", metadata={"help": ""}
    )
    ENTITY: str = field(default="", metadata={"help": ""})

    def to_rust(self) -> gcore_matcher.LiteralMatcherConfig:
        def py2ru(ident):
            if ident == "":
                return "always_fail_test"
            assert ident.startswith("."), ident
            return ident[1:]

        return gcore_matcher.LiteralMatcherConfig(
            py2ru(self.STRING),
            py2ru(self.QUANTITY),
            py2ru(self.GLOBECOORDINATE),
            py2ru(self.TIME),
            py2ru(self.MONOLINGUAL_TEXT),
            py2ru(self.ENTITY),
        )
