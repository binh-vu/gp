from __future__ import annotations

import re
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, get_type_hints

from gp.actors.data.prelude import GPExample
from gp.misc.appconfig import AppConfig
from gramsplus.semanticmodeling.literal_matcher import LiteralMatcherConfig
from gramsplus.semanticmodeling.text_parser import TextParserConfigs
from ream.cache_helper import CacheableFn, MemBackend, assign_dataclass_field_names

if TYPE_CHECKING:
    from gp.entity_linking.candidate_ranking.feats.make_cr_dataset import CRDatasetBuilder


@dataclass
class CRFeatFnArgs:
    topk: Optional[int] = field(
        default=None,
        metadata={"help": "The number of candidates to keep for each cell."},
    )
    topk_scoring: Literal["default"] = field(
        default="default",
        metadata={"help": "The scoring method for topk candidates."},
    )
    add_missing_gold: bool = field(
        default=False,
        metadata={
            "help": "Whether to add missing gold entities if candidate retrieval misses them"
        },
    )
    remove_nil_entity: bool = field(
        default=False,
        metadata={"help": "Whether to remove NIL entity from candidates"},
    )
    text_parser_cfg: TextParserConfigs = field(
        default_factory=TextParserConfigs,
        metadata={"help": "The text parser configuration"},
    )
    literal_matcher_cfg: LiteralMatcherConfig = field(
        default_factory=LiteralMatcherConfig,
        metadata={"help": "The literal matcher configuration"},
    )
    text_embedding_model: str = field(
        default="",
        metadata={"help": "The embedding model to calculate text embeddings"},
    )


assign_dataclass_field_names(CRFeatFnArgs)


class CRFeatFn(CacheableFn):
    backends = []
    use_args = []

    def __init__(self, store: CRDatasetBuilder):
        super().__init__(
            self.use_args,
            store.get_working_fs(),
            not AppConfig.get_instance().is_cache_enable,
        )
        # to assign functions based on type hints
        for name, type in get_type_hints(self.__class__).items():
            if issubclass(type, CacheableFn):
                setattr(self, name, store.get_func(type))
        self.store = store
        self.cangen_actor = store.cangen_actor
        self.args: CRFeatFnArgs = CRFeatFnArgs()

    @abstractmethod
    def __call__(self, ex: GPExample) -> Any:
        """This is where to put the function body. To cache it, wraps it with @Cache.<X> decorators"""
        ...

    def batch_call(self, exs: list[GPExample]) -> list[Any]:
        return [self(ex) for ex in exs]

    def set_args(self, args: CRFeatFnArgs):
        self.args = args
        self.cache_file = self.get_cache_file()
        return self

    def get_cache_file(self):
        if hasattr(self, "VERSION"):
            version = f"_v{getattr(self, 'VERSION')}"
        else:
            version = ""

        filename = f"%s%s" % (
            re.sub(
                r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__.replace("Fn", "")
            ).lower(),
            version,
        )

        fs = self.get_working_fs()
        file = fs.get(filename, key=self.get_cache_key(self, self.args), save_key=True)
        if file.exists():
            return str(file.get().relative_to(fs.root))

        with fs.acquire_write_lock(), file.reserve_and_track() as realfile:
            return str(realfile.relative_to(fs.root))

    @classmethod
    def new_mem_backend(cls):
        backend = MemBackend()
        cls.backends.append(backend)
        return backend

    @classmethod
    @contextmanager
    def auto_clear_mem_backend(cls):
        try:
            yield None
        finally:
            for backend in cls.backends:
                backend.clear()


# class GetTopKCanFn(CRFeatFn):
#     use_args = [CRFeatFnArgs.topk, CRFeatFnArgs.topk_scoring]
#     get_candidates: GetCandidatesFn
#     get_canbase: GetCanBaseFn
#     get_can_basefeat: GetCanBaseFeatFn

#     @Cache.cache(
#         backend=Cache.cls.dir(
#             cls=[DatasetCandidateEntities, SingleNumpyArray],
#             mem_persist=CRFeatFn.new_mem_backend(),
#             dirname=CRFeatFn.get_dirname,
#         ),
#         **CRFeatFn.cache_decorator_args(),
#     )
#     def __call__(
#         self, args: CRFeatFnArgs
#     ) -> tuple[DatasetCandidateEntities, Optional[SingleNumpyArray]]:
#         cans = self.get_candidates(args)
#         topk_index = None

#         if args.topk is not None:
#             assert args.topk_scoring == "default"

#             UPPERBOUND_SCORE = 1e6
#             cans.score[cans.score == GetCandidatesFn.DEFAULT_GOLD_IMPUTE_SCORE] = (
#                 UPPERBOUND_SCORE
#             )
#             cans, topk_index = cans.top_k_candidates(args.topk, return_index_remap=True)
#             cans.score[cans.score == UPPERBOUND_SCORE] = (
#                 GetCandidatesFn.DEFAULT_GOLD_IMPUTE_SCORE
#             )
#             topk_index = SingleNumpyArray(topk_index)

#         return cans, topk_index


# class GetTopKCanBaseFn(CRFeatFn):
#     get_canbase: GetCanBaseFn
#     get_topk_candidates: GetTopKCanFn

#     @Cache.cache(
#         backend=CRFeatFn.new_mem_backend(),
#         cache_key=CacheableFn.get_cache_key,
#     )
#     def __call__(self, args: CRFeatFnArgs) -> CRDatasetCan:
#         subset_index = self.get_topk_candidates(args)[1]
#         canbase = self.get_canbase(args)
#         if subset_index is not None:
#             canbase = canbase.select(subset_index.value)
#         return canbase


# class GetTopKCanBaseFeatFn(CRFeatFn):
#     get_can_basefeat: GetCanBaseFeatFn
#     get_topk_candidates: GetTopKCanFn

#     @Cache.cache(
#         backend=CRFeatFn.new_mem_backend(),
#         cache_key=CacheableFn.get_cache_key,
#     )
#     def __call__(self, args: CRFeatFnArgs) -> CRDatasetFeatures:
#         subset_index = self.get_topk_candidates(args)[1]
#         can_basefeat = self.get_can_basefeat(args)
#         if subset_index is not None:
#             can_basefeat = can_basefeat.select(subset_index.value)
#         return can_basefeat


# class GetHeaderEmbeddingFn(CRFeatFn):
#     use_args = [
#         CRFeatFnArgs.dsquery,
#         CRFeatFnArgs.text_embedding_model,
#     ]
#     get_topk_candidates: GetTopKCanFn

#     @Cache.cache(
#         backend=CRFeatFn.new_mem_backend(),
#         cache_key=CacheableFn.get_cache_key,
#     )
#     def __call__(self, args: CRFeatFnArgs) -> np.ndarray:
#         col_names = self.get_headers(args)
#         return self.store.get_text_embedding(args.text_embedding_model).batch_get(
#             col_names, verbose=AppConfig.get_instance().is_canrank_verbose
#         )

#     @Cache.cache(
#         backend=CRFeatFn.new_mem_backend(),
#         cache_key=CacheableFn.get_cache_key,
#     )
#     def get_headers(self, args: CRFeatFnArgs) -> BatchText:
#         examples = self.data_actor(args.dsquery)
#         topkcans = self.get_topk_candidates(args)[0]

#         col_names = BatchText()
#         for ex in examples:
#             for col_index, (cstart, cend, _) in topkcans.index[ex.table.table.table_id][
#                 2
#             ].items():
#                 col = ex.table.table.get_column_by_index(col_index)
#                 col_name = assert_not_null(col.clean_multiline_name)
#                 col_names.add_repeated_text(col_name, cend - cstart)

#         assert len(topkcans) == len(col_names)
#         return col_names


# class GetEmptyHeaderEmbeddingFn(CRFeatFn):
#     use_args = [
#         CRFeatFnArgs.dsquery,
#         CRFeatFnArgs.text_embedding_model,
#     ]
#     get_topk_candidates: GetTopKCanFn

#     @Cache.cache(
#         backend=CRFeatFn.new_mem_backend(),
#         cache_key=CacheableFn.get_cache_key,
#     )
#     def __call__(self, args: CRFeatFnArgs) -> np.ndarray:
#         if args.text_embedding_model == "sentence-transformers/all-mpnet-base-v2":
#             embed_dim = 768
#         else:
#             raise NotImplementedError(args.text_embedding_model)

#         topkcans = self.get_topk_candidates(args)[0]
#         return np.zeros((len(topkcans), embed_dim), dtype=np.float32)


# class GetAutoHeaderEmbeddingFn(CRFeatFn):
#     use_args = [
#         CRFeatFnArgs.dsquery,
#         CRFeatFnArgs.text_embedding_model,
#     ]
#     get_topk_candidates: GetTopKCanFn
#     get_header_embedding: GetHeaderEmbeddingFn

#     @Cache.cache(
#         backend=CRFeatFn.new_mem_backend(),
#         cache_key=CacheableFn.get_cache_key,
#     )
#     def __call__(self, args: CRFeatFnArgs) -> np.ndarray:
#         if args.text_embedding_model == "sentence-transformers/all-mpnet-base-v2":
#             embed_dim = 768
#         else:
#             raise NotImplementedError(args.text_embedding_model)

#         col_names = self.get_header_embedding.get_headers(args)
#         embeds = self.get_header_embedding(args)

#         empty_header = []
#         for name, idx in col_names.unique_text.items():
#             # for each column, if the header is "" or "colX", we convert it to empty embed
#             if name.strip() == "" or re.match(r"col\d+", name) is not None:
#                 empty_header.append(idx)

#         embeds[empty_header] = np.zeros(
#             (len(empty_header), embed_dim), dtype=np.float32
#         )
#         return embeds


# class HasHeaderFn(CRFeatFn):
#     use_args = [
#         CRFeatFnArgs.dsquery,
#         CRFeatFnArgs.text_embedding_model,
#     ]
#     get_topk_candidates: GetTopKCanFn
#     get_header_embedding: GetHeaderEmbeddingFn

#     @Cache.cache(
#         backend=Cache.cls.dir(
#             cls=SingleNumpyArray,
#             mem_persist=CRFeatFn.new_mem_backend(),
#             dirname=CRFeatFn.get_dirname,
#         ),
#         **CRFeatFn.cache_decorator_args(),
#     )
#     def __call__(self, args: CRFeatFnArgs) -> SingleNumpyArray:
#         col_names = self.get_header_embedding.get_headers(args)
#         empty_header = set()
#         for name, idx in col_names.unique_text.items():
#             # for each column, if the header is "" or "colX", we convert it to empty embed
#             if name.strip() == "" or re.match(r"col\d+", name) is not None:
#                 empty_header.add(idx)

#         return SingleNumpyArray(
#             np.asarray(
#                 [i not in empty_header for i in col_names.text_index], dtype=np.bool_
#             )
#         )


# class HasHeader2Fn(CRFeatFn):
#     use_args = [
#         CRFeatFnArgs.dsquery,
#         CRFeatFnArgs.text_embedding_model,
#     ]
#     get_topk_candidates: GetTopKCanFn
#     get_header_embedding: GetHeaderEmbeddingFn

#     @Cache.cache(
#         backend=Cache.cls.dir(
#             cls=SingleNumpyArray,
#             mem_persist=CRFeatFn.new_mem_backend(),
#             dirname=CRFeatFn.get_dirname,
#         ),
#         **CRFeatFn.cache_decorator_args(),
#     )
#     def __call__(self, args: CRFeatFnArgs) -> SingleNumpyArray:
#         col_names = self.get_header_embedding.get_headers(args)
#         empty_header = set()
#         for name, idx in col_names.unique_text.items():
#             # for each column, if the header is "" or "colX", we convert it to empty embed
#             if re.match(r"col\d+", name) is not None:
#                 empty_header.add(idx)

#         return SingleNumpyArray(
#             np.asarray(
#                 [i not in empty_header for i in col_names.text_index], dtype=np.bool_
#             )
#         )


# class GetEntityDescriptionEmbeddingFn(CRFeatFn):
#     use_args = [
#         CRFeatFnArgs.dsquery,
#         CRFeatFnArgs.text_embedding_model,
#     ]
#     get_topk_candidates: GetTopKCanFn

#     @Cache.cache(
#         backend=CRFeatFn.new_mem_backend(),
#         cache_key=CacheableFn.get_cache_key,
#     )
#     def __call__(self, args: CRFeatFnArgs) -> np.ndarray:
#         topkcans = self.get_topk_candidates(args)[0]

#         texts = BatchText.from_list_str(topkcans.description)
#         return self.store.get_text_embedding(args.text_embedding_model).batch_get(
#             texts, verbose=AppConfig.get_instance().is_canrank_verbose
#         )


# class GetEmptyEntityDescriptionEmbeddingFn(CRFeatFn):
#     use_args = [
#         CRFeatFnArgs.dsquery,
#         CRFeatFnArgs.text_embedding_model,
#     ]
#     get_topk_candidates: GetTopKCanFn

#     @Cache.cache(
#         backend=CRFeatFn.new_mem_backend(),
#         cache_key=CacheableFn.get_cache_key,
#     )
#     def __call__(self, args: CRFeatFnArgs) -> np.ndarray:
#         if args.text_embedding_model == "sentence-transformers/all-mpnet-base-v2":
#             embed_dim = 768
#         else:
#             raise NotImplementedError(args.text_embedding_model)

#         topkcans = self.get_topk_candidates(args)[0]
#         return np.zeros((len(topkcans), embed_dim), dtype=np.float32)


# if __name__ == "__main__":
#     import IPython

#     IPython.embed()
#     print(CRFeatFnArgs.dsquery)
