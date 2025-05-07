# %%
from __future__ import annotations

import re
from functools import lru_cache, partial
from typing import Callable, Optional

import inflect


class NamingHelper:

    @staticmethod
    def norm(name: str, funcs: list[Callable[[str], str]]):
        for f in funcs:
            name = f(name)
        return name

    @staticmethod
    def make_dataset_v2_norm_fn(name: str):
        """The normalization function use in make_dataset_v2"""
        return NamingHelper.norm(
            name,
            funcs=[
                partial(NamingHelper.norm_ref_and_special_chars, tail=False),
                partial(NamingHelper.norm_plurals, parenthesized_plurals=True),
                NamingHelper.norm_spaces,
                partial(NamingHelper.mask_number, whole_word=False),
                str.lower,
            ],
        )

    @staticmethod
    def norm_spaces(name: str):
        words = re.split(r"([^\xa0\n\t\r\f ]+)", name)
        return " ".join((words[i] for i in range(1, len(words), 2)))

    @staticmethod
    def norm_plurals(name: str, parenthesized_plurals: bool = True):
        """Convert plural words in the string to singular form. This function keeps the spaces between words.

        Args:
            name: The input string.
            parenthesized_plurals: If true, remove parenthesized plurals such as `Director(s)`, `Winners(s) and nominee(s)`.
        """
        p = get_inflect_engine()
        # split but keep the separator
        words = re.split(r"([^\xa0\n\t\r\f ]+)", name)
        for i in range(1, len(words), 2):
            # has side effect that pronouns are converted as well (they -> it)
            sing = p.singular_noun(words[i])
            if sing:
                words[i] = sing

            if parenthesized_plurals:
                if words[i].endswith("(s)"):
                    words[i] = words[i][:-3]

        return "".join(words)

    @staticmethod
    def norm_ref_and_special_chars(name: str, tail: bool = True):
        """Detect and remove the following artifacts at anywhere or at end of the strings:

        1. References: [8], [7][8], [a], [b]
        2. Non alphanumeric characters: `:`
        3. Remove (v t e) found in many wikipedia tables
        """
        if tail:
            # we are going to use $ to match the end of the string
            postfix = r"[\xa0\n\t\r\f ]*$"
        else:
            postfix = ""

        is_updated = True

        # remove artifacts that are easy to detect because of special characters such as ( or [
        while is_updated:
            is_updated = False
            for endpattern in [
                r"(\[\d+\])+" + postfix,
                r"(\[[a-zA-Z]\])+" + postfix,
                r"([:;!])" + postfix,
                r"(\n?v\nt\ne)" + postfix,
            ]:
                m = re.search(endpattern, name)
                if m is not None:
                    name = name[: m.span()[0]] + name[m.span()[1] :]
                    is_updated = True

        return name

    @staticmethod
    def mask_number(name: str, whole_word: bool) -> str:
        """Masking numbers in a string to the special character D.

        For example:
            - `2015 Team` -> `DDDD Team`
            - `after 32 events` -> `after DD events`

        Args:
            name: The input string.
            whole_word: If true, only the whole word numbers are masked. Otherwise, every digit is masked.
        """
        if whole_word:
            words = re.split(r"([^\xa0\n\t\r\f ]+)", name)
            assert words[0] == ""
            for i in range(1, len(words), 2):
                if words[i].isdigit():
                    words[i] = "D" * len(words[i])
            return "".join(words)

        return re.sub(r"\d", "D", name)


@lru_cache()
def get_inflect_engine():
    return inflect.engine()


if __name__ == "__main__":
    testcases = [
        "United States (retail)[24]",
        "School [13][15]",
        "Title:",
        "Film![5]",
        "Title\n[19]",
        "Location(s)[3]",
        "Note(s)",
        "Type[3][b]",
        "State[2][3][4]",
        "Districts",
        "Winners and\xa0nominees",
        "Winner(s) and\xa0nominee(s)",
        "after 32 events",
        "after 2012/10 events",
        "Play-by-Play",  # to lowercase is just a side-effect of inflect singular_noun
        "Name\nv\nt\ne",
        "v\nt\ne\nWild Card team\n(Top team qualifies for postseason)",
        "!Name",
    ]

    norm_fns = [
        partial(NamingHelper.norm_ref_and_special_chars, tail=False),
        partial(NamingHelper.norm_plurals, parenthesized_plurals=True),
        NamingHelper.norm_spaces,
        partial(NamingHelper.mask_number, whole_word=False),
    ]

    for s in testcases:
        print(f"`{s}` -> `{NamingHelper.norm(s, norm_fns)}`")

# %%
