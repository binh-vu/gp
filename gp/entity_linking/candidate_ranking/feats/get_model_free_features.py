from __future__ import annotations

import enum
import re

import numpy as np
import strsim
from gp.actors.data.prelude import GPExample
from gp.entity_linking.candidate_ranking.feats.feat_fn import CRFeatFn
from gp.entity_linking.candidate_ranking.feats.get_candidates import GetCandidatesFn
from ream.cache_helper import Cache
from ream.data_model_helper._numpy_model import Single2DNumpyArray
from sm.misc.funcs import batch
from sm.misc.ray_helper import is_parallelizable, ray_map

char_tokenizer = strsim.CharacterTokenizer()


CRModelFreeFeats = Single2DNumpyArray


class GetModelFreeFeatFn(CRFeatFn):
    use_args = []

    get_candidates: GetCandidatesFn

    @Cache.cache(
        backend=Cache.sqlite.serde(
            cls=CRModelFreeFeats,
            filename=lambda slf, ex: slf.cache_file + "/data.sqlite",
            mem_persist=True,
        ),
        cache_key=lambda self, ex: ex.id,
        disable="disable",
    )
    def __call__(self, ex: GPExample):
        kgdb_service = self.store.kgdb_service(ex.kgname)

        cans = self.get_candidates(ex)
        labels = kgdb_service.labels(ex.id, cans.ent_id)
        aliases = kgdb_service.aliases(ex.id, cans.ent_id)
        popularities = kgdb_service.pageranks(ex.id, cans.ent_id)

        return CRModelFreeFeats(
            extract_features(cans.cell, labels, aliases, popularities)
        )

    @Cache.flat_cache(
        backend=Cache.sqlite.serde(
            cls=CRModelFreeFeats,
            filename=lambda slf, ex: slf.cache_file + "/data.sqlite",
            mem_persist=True,
        ),
        cache_key=lambda self, ex: ex.id,
        disable="disable",
    )
    def batch_call(self, exs: list[GPExample]) -> list[CRModelFreeFeats]:
        if not is_parallelizable() or len(exs) == 0:
            return [self(ex) for ex in exs]

        kgdb_service = self.store.kgdb_batch_service(exs[0].kgname)

        lst_cans = self.get_candidates.batch_call(exs)
        lst_args = [(ex.id, cans.ent_id) for ex, cans in zip(exs, lst_cans)]
        lst_labels = kgdb_service.batch_labels(lst_args)
        lst_aliases = kgdb_service.batch_aliases(lst_args)
        lst_popularities = kgdb_service.batch_pageranks(lst_args)

        features = ray_map(
            extract_features,
            [
                (
                    lst_cans[ei].cell,
                    lst_labels[ei],
                    lst_aliases[ei],
                    lst_popularities[ei],
                )
                for ei in range(len(exs))
            ],
            verbose=False,
            desc="extract features",
            is_func_remote=False,
        )
        return [CRModelFreeFeats(features[ei]) for ei in range(len(exs))]


def extract_features(
    cells: np.ndarray,
    labels: list[str],
    aliases: list[list[str]],
    popularities: list[float],
):
    feats = []

    for i in range(len(labels)):
        cell = cells[i]

        feat = np.zeros((N_PAIRWISE_FEATURES + N_EXTRA_FEATURES,), dtype=np.float64)
        label_feat = extract_pairwise_features_v2(cell, labels[i])
        for alias in aliases[i]:
            label_feat = np.maximum(
                label_feat, extract_pairwise_features_v2(cell, alias)
            )

        feat[:N_PAIRWISE_FEATURES] = label_feat
        feat[N_PAIRWISE_FEATURES] = popularities[i]
        feats.append(feat)

    return np.array(feats, dtype=np.float64)


# number of extracted pairwise features
# for setting default gold_features for NIL entity
# one extra feature is added from pagerank
class PairwiseFeatures(enum.Enum):
    levenshtein_sim = 0
    jaro_winkler_sim = 1
    monge_elkan_sim = 2
    sym_monge_elkan_sim = 3
    hybrid_jaccard_sim = 4
    numeric_sym_monge_elkan_sim = 5
    numeric_hybrid_jaccard_sim = 6


N_PAIRWISE_FEATURES = len(list(PairwiseFeatures))
N_EXTRA_FEATURES = 1


def extract_pairwise_features_v2(text: str, entity_label: str):
    chartok = strsim.CharacterTokenizer()
    charseqtok = strsim.WhitespaceCharSeqTokenizer()

    text_t1 = chartok.tokenize(text)
    entity_label_t1 = chartok.tokenize(entity_label)

    text_t2 = charseqtok.tokenize(text)
    entity_label_t2 = charseqtok.tokenize(entity_label)

    text_t3 = charseqtok.unique_tokenize(text)
    entity_label_t3 = charseqtok.unique_tokenize(entity_label)

    out2 = [
        strsim.levenshtein_similarity(text_t1, entity_label_t1),
        strsim.jaro_winkler_similarity(text_t1, entity_label_t1),
        strsim.monge_elkan_similarity(text_t2, entity_label_t2),
        (
            sym_monge_score := strsim.symmetric_monge_elkan_similarity(
                text_t2, entity_label_t2
            )
        ),
        (hyjac_score := strsim.hybrid_jaccard_similarity(text_t3, entity_label_t3)),
        does_ordinal_match(text, entity_label, sym_monge_score, 0.7),
        does_ordinal_match(text, entity_label, hyjac_score, 0.7),
    ]
    return out2


def does_ordinal_match(s1: str, s2: str, sim: float, threshold: float) -> float:
    """Test for strings containing ordinal categorical values such as Su-30 vs Su-25, 29th Awards vs 30th Awards.

    Args:
        s1: Cell Label
        s2: Entity Label
    """
    if sim < threshold:
        return 0.4
    digit_tokens_1 = re.findall(r"\d+", s1)
    if len(digit_tokens_1) == 0:
        return 0.4

    digit_tokens_2 = re.findall(r"\d+", s2)
    if digit_tokens_1 == digit_tokens_2:
        return 1.0
    return 0.0
