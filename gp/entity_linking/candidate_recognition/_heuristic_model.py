from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import spacy
from dateutil.parser import parse, parser
from gp.entity_linking.candidate_recognition._interface import ICanReg
from kgdata.models import Ontology
from sm.dataset import Example, FullTable
from sm.misc.ray_helper import ray_map


class HeuristicCanReg(ICanReg):
    VERSION = 102

    def __call__(
        self, examples: Sequence[Example[FullTable]], ontology: Ontology
    ) -> list[list[int]]:
        return ray_map(
            predict_entity_column,
            [(ex,) for ex in examples],
            verbose=True,
            desc="Predicting entity columns",
            is_func_remote=False,
            using_ray=len(examples) > 1,
        )


num_pattern = r"(\d+((\.)\d+)?)"
unit_pattern = r"(m|in|ft|cm|khz|hz|kw)"

insensitive_patterns = [
    r"^[+-T]?\s*\d+$",  # T3, -1, +1, etc.
    r"^[\[\(]\s*\d+\s*[\]\)]$",  # (1), [1], etc.
    r"^\d+(\s*[-=,]\s*\d+)+$",  # 1=2 | 1,2 | 1-2
    r"^\d+\s*\(\s*\d+\s*\)$",  # 1 (1)
    r"^(-|•|x|n/a)$",  # missing values
    # r".*\(\s*\d+(\s*,\s*\d+)+\s*\)?",  # (1, 2, 3)
    rf"{num_pattern}\s*{unit_pattern}",  # 1.2 m, 1.2 in, etc.
    rf"{num_pattern}\s*{unit_pattern}\s*{num_pattern}\s*{unit_pattern}",  # 5 ft 11 in
    rf"{num_pattern}\s*{unit_pattern}\s*\(\s*{num_pattern}\s*{unit_pattern}\s*{num_pattern}\s*{unit_pattern}\s*\)",  # 5 ft 11 in
]
regexes = [re.compile(p, flags=re.IGNORECASE) for p in insensitive_patterns]


def get_type(cell: str | int | float):
    global regexes

    if isinstance(cell, (int, float)):
        return DataType.NUM

    normcell = cell.strip()
    normcell = normcell.replace("−", "-").replace("–", "-")

    dtype = MSpacy.get_instance().get_type(normcell)
    if dtype != DataType.TEXT:
        return dtype
    if any(reg.match(normcell) is not None for reg in regexes):
        return DataType.NUM
    return dtype


def predict_entity_column(ex: Example[FullTable]) -> list[int]:
    ent_cols = []
    for col in ex.table.table.columns:
        dtype_counter = defaultdict(int)
        for ri, cell in enumerate(col.values):
            dtype_cell = get_type(cell)
            dtype_counter[dtype_cell] += 1

        temp_dtype = sorted(dtype_counter.items(), key=lambda x: x[1], reverse=True)
        if not len(temp_dtype):
            # number
            continue
        else:
            if (
                temp_dtype[0][0] == DataType.NONE
                and len(temp_dtype) > 1
                and temp_dtype[1][1]
            ):
                col_dtype = temp_dtype[1][0]
            else:
                col_dtype = temp_dtype[0][0]

        if col_dtype == DataType.TEXT:
            ent_cols.append(col.index)
    return ent_cols


class DataType:
    TEXT = 1
    NUM = 0
    NONE = 2


class MSpacy(object):
    """
    Code borrow from Phuc Nguyen. mtab.api.annotator.m_spacy
    """

    instance = None

    def __init__(self):
        self.model_names = {
            "en": "en_core_web_sm",  # sm, md, lg, trf
            "all": "xx_sent_ud_sm",  # python -m spacy download xx_sent_ud_sm
        }
        self.models = {
            "en": None,
            "all": None,
        }
        self.load_model(lang="en")
        self.load_model(lang="all")
        self.dims_num = {
            "CARDINAL",
            "PERCENT",
            "MONEY",
            "ORDINAL",
            "QUANTITY",
            "TIME",
            "DATE",
        }
        self.NONE_CELLS = {
            "''",
            '""',
            "-",
            "--",
            "'-'",
            '"-"',
            " ",
            ".",
            "' '",
            '" "',
            "nan",
            "none",
            "null",
            "blank",
            "yes",
            "unknown",
            "?",
            "??",
            "???",
            "0",
            "total",
        }
        self.NUM_SPACY = {
            "CARDINAL",
            "PERCENT",
            "MONEY",
            "ORDINAL",
            "QUANTITY",
            "TIME",
            "DATE",
        }

    @staticmethod
    def get_instance():
        if MSpacy.instance is None:
            MSpacy.instance = MSpacy()
        return MSpacy.instance

    def load_model(self, lang):
        """Load SpaCy model with the language of [lang]

        Args:
            lang (string): language id
        """
        if lang != "en":
            lang = "all"
        print("Loaded: Type entity model (%s)" % self.model_names[lang])
        self.models[lang] = spacy.load(self.model_names[lang])  # type: ignore

    def _parse(self, text, lang="en"):
        """[summary]

        Args:
            text ([type]): [description]
            lang (str, optional): [description]. Defaults to "en".

        Returns:
            [type]: [description]
        """

        spacy_model = self.models.get(lang, None)
        if not spacy_model:
            self.load_model(lang)
            spacy_model = self.models[lang]
        response = spacy_model(text)  # type: ignore
        return response

    def get_majority_type(self, text, lang="en"):
        """[summary]

        Args:
            text ([type]): [description]
            lang (str, optional): [description]. Defaults to "en".

        Returns:
            [type]: [description]
        """
        types = DataType.TEXT
        text = str(text)
        if text:
            # count_numbers = sum(c.isdigit() for c in text)
            # if count_numbers > len(text):
            response = self._parse(text, lang)
            if len(response.ents):
                num_len = 0
                num_types = Counter()
                text_types = Counter()
                for r in response.ents:
                    if r.label_ in self.dims_num:
                        num_len += len(r.text)
                        num_types[r.label_] += 1
                    else:
                        text_types[r.label_] += 1
                if num_len > 0.5 * len(text):
                    types = num_types.most_common(1)[0][0]
                elif len(text_types) > 0:
                    types = text_types.most_common(1)[0][0]
        return types

    def get_type(self, text, lang="en"):
        if not text or text.lower() in self.NONE_CELLS:
            return DataType.NONE

        if text[0] == "〒" or text[:2] == "代表":
            return DataType.NUM

        txt_tmp = text.replace("%", "").replace(" ", "")
        if self.convert_num(txt_tmp) is not None:
            return DataType.NUM

        if self.is_date(text):
            return DataType.NUM

        main_type = self.get_majority_type(text, lang)

        if main_type in self.NUM_SPACY:
            return DataType.NUM

        return DataType.TEXT

    def is_number(self, text, lang="en"):
        is_num = False
        if len(text):
            response = self._parse(text, lang)
            if len(response.ents):
                num_len = 0
                for r in response.ents:
                    if r.label_ in self.dims_num:
                        num_len += len(r.text)
                if num_len > 0.5 * len(text):
                    is_num = True
        return is_num

    def is_text(self, text, lang="en"):
        return not self.is_number(text, lang)

    def get_entity_text(self, text, lang="en"):
        results = []
        if len(text):
            response = self._parse(text, lang)
            if len(response.ents):
                for r in response.ents:
                    if r.label_ not in self.dims_num:
                        results.append((r.text, r.label_))
        return results

    @classmethod
    def convert_num(cls, text):
        if not text:
            return None
        try:
            text = cls.removeCommasBetweenDigits(text)
            # tmp = representsFloat(text)
            # if not tmp:
            #     return None
            #
            # return parseNumber(text)
            return float(text)  # type: ignore
        except ValueError:
            return None

    @classmethod
    def removeCommasBetweenDigits(cls, text):
        """
        :example:
        >>> removeCommasBetweenDigits("sfeyv dsf,54dsf ef 6, 6 zdgy 6,919 Photos and 3,3 videos6,")
        'sfeyv dsf,54dsf ef 6, 6 zdgy 6919 Photos and 33 videos6,'
        """
        if text is None:
            return None
        else:
            return re.sub(r"([0-9]),([0-9])", "\g<1>\g<2>", text)

    @classmethod
    def is_date_complete(cls, date, must_have_attr=("year", "month")):
        parse_res, _ = parser()._parse(date)  # type: ignore
        # If date is invalid `_parse` returns (None,None)
        if parse_res is None:
            return False
        # For a valid date `_result` object is returned. E.g. _parse("Sep 23") returns (_result(month=9, day=23), None)
        for attr in must_have_attr:
            if getattr(parse_res, attr) is None:
                return False
        return True

    @classmethod
    def is_date(cls, string, fuzzy=False):
        if not string:
            return False
        try:
            parse(string, fuzzy=fuzzy)
            if cls.is_date_complete(string):
                return True
            return False
        except Exception:
            return False
