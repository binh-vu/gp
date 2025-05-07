from __future__ import annotations

import numpy as np
from gp.actors.data.prelude import GPExample
from gp.entity_linking.candidate_ranking.feats.feat_fn import CRFeatFn, CRFeatFnArgs
from gp.entity_linking.candidate_ranking.feats.get_candidates import GetCandidatesFn


class GetEmptyHeaderEmbeddingFn(CRFeatFn):
    use_args = [
        CRFeatFnArgs.text_embedding_model,
    ]

    get_candidates: GetCandidatesFn

    def __call__(self, ex: GPExample) -> np.ndarray:
        if self.args.text_embedding_model == "sentence-transformers/all-mpnet-base-v2":
            embed_dim = 768
        else:
            raise NotImplementedError(self.args.text_embedding_model)

        n = self.get_candidates(ex).cell.shape[0]
        return np.zeros((n, embed_dim), dtype=np.float32)
