from __future__ import annotations

from gp.actors.data.prelude import GPExample
from gp.entity_linking.candidate_ranking.feats.feat_fn import CRFeatFn
from gp.entity_linking.candidate_ranking.feats.get_candidates import CRTableCan, GetCandidatesFn
from gp.entity_linking.candidate_ranking.feats.get_model_free_features import GetModelFreeFeatFn


class GetTopKCandidateMaskFn(CRFeatFn):
    get_candidates: GetCandidatesFn
    get_model_free_feat: GetModelFreeFeatFn

    def __call__(self, ex: GPExample) -> CRTableCan:
        self.get_candidates(ex)
