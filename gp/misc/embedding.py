from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import orjson
import serde.json
import torch
from hugedict.sqlite import SqliteDict, SqliteDictFieldType
from ream.cache_helper import Cache, MemBackend
from ream.workspace import ReamWorkspace
from sm.misc.funcs import (
    assert_all_item_not_null,
    assert_not_null,
    batch,
    get_incremental_path,
)
from sm.misc.ray_helper import ray_actor_map_1, ray_get_num_gpu
from tqdm.auto import tqdm


class TextEmbedding:
    def __init__(
        self,
        dir: Path,
        index: SqliteDict[str, tuple[int, int]],
        datasets: list[EmbeddingChunk],
        embedding_model: str,
        chunk_size: int,
    ):
        raise Exception("Deprecated")

        self.dir = dir
        # mapping from the text to the index of the dataset and the index of the example
        self.index = index
        self.datasets = datasets

        self.chunk_size = chunk_size
        self.index_buffer = {}
        self.data_buffer = []
        self.embedding_model = embedding_model

        self.available_devices = []
        if torch.cuda.is_available():
            self.n_gpus = torch.cuda.device_count()
            min_mem = EmbeddingModel.get_model_required_memory(self.embedding_model)
            for i in range(torch.cuda.device_count()):
                if torch.cuda.mem_get_info(i)[0] >= min_mem:
                    self.available_devices.append(f"cuda:{i}")
        else:
            self.n_gpus = 0

        if len(self.available_devices) == 0:
            self.available_devices = ["cpu"]

    @classmethod
    def from_disk(
        cls,
        dir: Path,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 64000,
    ):
        (dir / "datasets").mkdir(parents=True, exist_ok=True)
        index: SqliteDict[str, tuple[int, int]] = SqliteDict(
            dir / "index.sqlite",
            SqliteDictFieldType.str,
            orjson.dumps,
            orjson.loads,
            SqliteDictFieldType.bytes,
        )
        start = 0
        datasets = []
        for dsdir in sorted(
            (dir / "datasets").iterdir(), key=lambda x: int(x.name.split("_")[1])
        ):
            datasets.append(EmbeddingChunk.load(dsdir))
            assert (
                datasets[-1].start == start
            ), f"{dsdir.name} does not start at {start}"
            start = datasets[-1].end

        assert len(index) == sum(len(d) for d in datasets)
        return TextEmbedding(dir, index, datasets, embedding_model, chunk_size)

    def __contains__(self, text: str) -> bool:
        return text in self.index or text in self.index_buffer

    def retrieve(self, text: str) -> Optional[np.ndarray]:
        if text in self.index:
            dsidx, exidx = self.index[text]
            return self.datasets[dsidx][exidx]
        if text in self.index_buffer:
            return self.data_buffer[self.index_buffer[text]]
        return None

    def batch_retrieve_exist(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            if text in self.index:
                dsidx, exidx = self.index[text]
                out.append(self.datasets[dsidx][exidx])
            elif text in self.index_buffer:
                out.append(self.data_buffer[self.index_buffer[text]])
            else:
                raise KeyError(text)
        return np.stack(out)

    def batch_get(
        self,
        texts: Union[list[str], np.ndarray, BatchText],
        batch_size: int = 512,
        parallel: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        """Get the embeddings for the given texts."""
        if not isinstance(texts, BatchText):
            batch_text = BatchText.from_list_str(texts)
        else:
            batch_text = texts

        all_embs = [
            self.retrieve(text)
            for text in tqdm(
                batch_text.unique_text.keys(),
                desc="retrieving previously computed embeddings",
                disable=not verbose,
            )
        ]
        # we migrate enumerate(batch_text.unique_text.keys()) to batch_text.unique_text.items()
        # unknown_texts = [
        #     (i, text)
        #     for i, text in enumerate(batch_text.unique_text.keys())
        #     if all_embs[i] is None
        # ]
        assert all(
            i == batch_text.unique_text[text]
            for i, text in enumerate(batch_text.unique_text.keys())
        )
        unknown_texts = [
            (i, text)
            for text, i in batch_text.unique_text.items()
            if all_embs[i] is None
        ]

        if len(unknown_texts) > 0:
            # transform those unknown texts into embeddings
            batched_unk_texts = batch(batch_size, unknown_texts)
            num_gpu = int(ray_get_num_gpu())

            # we need to make sure the model has been fetched
            EmbeddingModel.ensure_loaded_model(*self.get_embedding_method_args())

            if parallel and num_gpu > 1 and len(batched_unk_texts) > 1:
                batched_embs: list[np.ndarray] = ray_actor_map_1(
                    EmbeddingModel,
                    "encode_texts",
                    [self.get_embedding_method_args() for _ in range(num_gpu)],
                    [([x[1] for x in b],) for b in batched_unk_texts],
                    verbose=verbose,
                    desc="compute text embeddings",
                    is_actor_remote=False,
                    remote_options={"num_gpus": 1},
                    before_shutdown=np.copy,
                    auto_shutdown=True,
                )
            else:
                emb_method = self.get_embedding_method()
                batched_embs: list[np.ndarray] = [
                    emb_method.encode_texts([x[1] for x in b])
                    for b in tqdm(
                        batched_unk_texts,
                        desc="compute text embeddings",
                        disable=not verbose,
                    )
                ]

            for i in range(len(batched_unk_texts)):
                for (j, unk_text), emb in zip(batched_unk_texts[i], batched_embs[i]):
                    self.index_buffer[unk_text] = len(self.data_buffer)
                    self.data_buffer.append(emb)
                    all_embs[j] = emb

            self.flush(True)

        new_all_embs = assert_all_item_not_null(all_embs)
        return np.stack([new_all_embs[i] for i in batch_text.text_index])

    def flush(self, soft: bool = False):
        """Flush the dataset to disk."""
        if soft and len(self.data_buffer) < self.chunk_size:
            return

        # determine the range in the buffer to save to disk, if hard flush, save everything
        assert len(self.index_buffer) == len(self.data_buffer)
        if len(self.data_buffer) == 0:
            return

        keys = [""] * len(self.index_buffer)
        for text, i in self.index_buffer.items():
            keys[i] = text
        data = self.data_buffer

        # determine if the last chunk is not full, we may need to update it.
        if len(self.datasets) > 0:
            last_chunk_size = len(self.datasets[-1])
            if last_chunk_size < self.chunk_size and (
                not soft or last_chunk_size + len(data) >= self.chunk_size
            ):
                # the last chunk isn't full and we can flush it
                newkeys = keys[: self.chunk_size - last_chunk_size]
                newdata = data[: self.chunk_size - last_chunk_size]

                ds = self.datasets[-1]
                ds_dir = self.dir / f"datasets/chunks_{len(self.datasets):02d}"
                ds_metadata = serde.json.deser(ds_dir / "metadata.json")
                assert (
                    ds_metadata["start"] == ds.start and ds_metadata["end"] == ds.end
                ), "Make sure that we got the right chunk location"
                assert ds.keys is not None, "The last chunk must load keys"

                ds.end += len(newdata)
                ds.keys.extend(newkeys)
                ds.data = np.concatenate([np.load(ds_dir / "data.npy"), newdata])

                ds.save(Path(str(ds_dir) + "_tmp"))
                shutil.rmtree(ds_dir)

                new_items: list[tuple[str, tuple[int, int]]] = []
                for idx, key in enumerate(ds.keys):
                    new_items.append((key, (len(self.datasets) - 1, idx)))
                self.index.batch_insert(new_items)

                os.rename(Path(str(ds_dir) + "_tmp"), ds_dir)

                data = data[self.chunk_size - last_chunk_size :]
                keys = keys[self.chunk_size - last_chunk_size :]

        if soft:
            save_to = int(len(self.data_buffer) / self.chunk_size) * self.chunk_size
        else:
            save_to = len(self.data_buffer)

        self.data_buffer = data[save_to:]
        self.index_buffer = {k: self.index_buffer[k] - save_to for k in keys[save_to:]}

        data = data[:save_to]
        keys = keys[:save_to]
        start = len(self.index)

        # save the data to disk
        for i in tqdm(range(0, len(data), self.chunk_size), desc="saving embeddings"):
            chunk_dir = get_incremental_path(self.dir / f"datasets/chunks")
            chunk_keys = keys[i : i + self.chunk_size]
            chunk_data = np.array(data[i : i + self.chunk_size])

            ds = EmbeddingChunk(
                start,
                start + len(chunk_data),
                keys=chunk_keys,
                data=chunk_data,
            )
            ds.save(chunk_dir)

            new_items: list[tuple[str, tuple[int, int]]] = []
            for idx, key in enumerate(chunk_keys):
                new_items.append((key, (len(self.datasets), idx)))
            self.index.batch_insert(new_items)

            self.datasets.append(ds)
            start += len(chunk_data)

    @Cache.cache(backend=MemBackend())
    def get_embedding_method(self):
        return EmbeddingModel(*self.get_embedding_method_args())

    def get_embedding_method_args(self) -> tuple:
        cache_dir = self.dir / "models"
        cache_dir.mkdir(exist_ok=True, parents=True)
        return self.embedding_model, cache_dir

    def reindex(self, batch_size: int = 512, verbose: bool = False):
        # verify our index
        for key, (dsidx, exidx) in tqdm(self.index.items(), total=len(self.index)):
            dataset = self.datasets[dsidx]
            assert dataset.keys is not None and dataset.keys[exidx] == key
        reindex_res = ray_actor_map_1(
            EmbeddingModel,
            "_reindex",
            [self.get_embedding_method_args() for _ in range(int(ray_get_num_gpu()))],
            [
                (self.dir / f"datasets/chunks_{i:02d}", batch_size)
                for i in range(1, len(self.datasets) + 1)
            ],
            verbose=verbose,
            desc="reindexing",
            is_actor_remote=False,
            remote_options={"num_gpus": 1},
            auto_shutdown=True,
        )
        for i in range(len(self.datasets)):
            if reindex_res[i]:
                print("Reindex chunk", i + 1)


class EmbeddingChunk:
    def __init__(
        self, start: int, end: int, keys: Optional[list[str]], data: np.ndarray
    ):
        self.start = start
        self.end = end
        self.keys = keys
        self.data = data

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return self.end - self.start

    def save(self, dir: Path):
        dir.mkdir(parents=True, exist_ok=True)
        assert not (dir / "metadata.json").exists()
        serde.json.ser(
            {
                "start": self.start,
                "end": self.end,
                "shape": list(self.data.shape),
                "dtype": self.data.dtype.name,
            },
            dir / "metadata.json",
        )
        if self.keys is not None:
            serde.json.ser(self.keys, dir / "keys.json.lz4")
        np.save(dir / "data.npy", self.data)

    @staticmethod
    def load(dir: Path, with_keys: bool = True, mem_map: bool = True) -> EmbeddingChunk:
        if with_keys:
            keys = serde.json.deser(dir / "keys.json.lz4")
        else:
            keys = None

        metadata = serde.json.deser(dir / "metadata.json")
        return EmbeddingChunk(
            metadata["start"],
            metadata["end"],
            keys,
            (
                np.lib.format.open_memmap(
                    dir / "data.npy",
                    dtype=np.dtype(metadata["dtype"]),
                    mode="r",
                    shape=tuple(metadata["shape"]),
                )
                if mem_map
                else np.load(dir / "data.npy")
            ),
        )


class EmbeddingModel:
    def __init__(self, embedding_model: str, cache_dir: Path):
        self.model = EmbeddingModel.get_embedding_model(embedding_model, cache_dir)
        self.model_name = embedding_model

    def encode_texts(self, texts: list[str]):
        embs = self.model.encode(texts)
        assert isinstance(embs, np.ndarray)
        return embs

    @staticmethod
    def get_embedding_model(model: str, cache_dir: Path):
        # TODO: it's difficult to specify a particular GPU in ray
        if model.startswith("sentence-transformers/"):
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(model, cache_folder=str(cache_dir))

        raise NotImplementedError(model)

    @staticmethod
    def ensure_loaded_model(embedding_model: str, cache_dir: Path):
        if (cache_dir / "_DOWNLOADED_SUCCESS").exists():
            return

        EmbeddingModel.get_embedding_model(embedding_model, cache_dir).encode(
            ["hello world"]
        )
        (cache_dir / "_DOWNLOADED_SUCCESS").touch()

    @staticmethod
    def get_model_required_memory(embedding_model: str):
        if embedding_model == "sentence-transformers/all-mpnet-base-v2":
            return 2.5 * (1024**3)  # 2.5GB
        return 1 * (1024**3)  # 1GB

    def _reindex(self, chunk_loc: Path, batch_size: int = 512):
        chunk = EmbeddingChunk.load(chunk_loc, with_keys=True, mem_map=False)
        chunk_emb = np.concatenate(
            [
                self.encode_texts(b)
                for b in batch(batch_size, assert_not_null(chunk.keys))
            ]
        )
        if not np.allclose(chunk_emb, chunk.data, rtol=0.0, atol=1e-8):
            # need to re-index
            if (chunk_loc / "data.npy.bad").exists():
                raise Exception("Previous reindexing does not work")
            np.save(chunk_loc / "data.tmp.npy", chunk_emb)
            os.rename(chunk_loc / "data.npy", chunk_loc / "data.bad.npy")
            os.rename(chunk_loc / "data.tmp.npy", chunk_loc / "data.npy")
            return True
        return False


@dataclass
class BatchText:
    """A utility class for handling a long list of texts with potentially duplicated."""

    unique_text: dict[str, int] = field(default_factory=dict)
    text_index: list[int] = field(default_factory=list)

    @staticmethod
    def from_list_str(texts: list[str] | np.ndarray) -> BatchText:
        batch_text = BatchText()
        for text in texts:
            batch_text.add_text(text)
        return batch_text

    def add_text(self, text: str):
        """Add text to this batch and return its position"""
        if text not in self.unique_text:
            index = len(self.unique_text)
            self.unique_text[text] = index
        else:
            index = self.unique_text[text]

        self.text_index.append(index)
        return index

    def add_repeated_text(self, text: str, n: int):
        if text not in self.unique_text:
            index = len(self.unique_text)
            self.unique_text[text] = index
        else:
            index = self.unique_text[text]

        self.text_index.extend([index] * n)
        return index

    def __len__(self):
        return len(self.text_index)


@dataclass
class BatchTextEmbedding:
    unique_text: dict[str, int]
    text_index: list[int]
    embeddings: np.ndarray

    @staticmethod
    def from_batch_text(texts: list[str] | BatchText, embs: np.ndarray):
        if not isinstance(texts, BatchText):
            batch_text = BatchText.from_list_str(texts)
        else:
            batch_text = texts
        return BatchTextEmbedding(batch_text.unique_text, batch_text.text_index, embs)

    def get_embedding(self, text: str):
        return self.embeddings[self.text_index[self.unique_text[text]]]


_text_embeddings: dict[str, TextEmbedding] = {}


def ream_get_text_embedding(model: str, singleton: bool = True):
    global _text_embeddings

    needinit = not singleton or model not in _text_embeddings
    if needinit:
        fspath = ReamWorkspace.get_instance().fs.get(
            f"embeddings/{model}",
            diskpath=f"embeddings/{model}",
            key={"model": model},
            save_key=True,
        )
        if fspath.exists():
            realdir = fspath.get()
        else:
            with fspath.reserve_and_track() as realdir:
                ...
        emb = TextEmbedding.from_disk(realdir, model)
        if singleton:
            _text_embeddings[model] = emb
    else:
        emb = _text_embeddings[model]

    return emb


@contextmanager
def auto_flush_text_embedding():
    global _text_embeddings

    try:
        yield None
    finally:
        for emb in _text_embeddings.values():
            emb.flush(soft=False)


if __name__ == "__main__":
    from scripts import DATA_DIR

    # validate if the state of the database is correct.
    EMB_DIR = DATA_DIR / "ream/embeddings/sentence_transformers/all_mpnet_base_v2_023"
    emb = TextEmbedding.from_disk(EMB_DIR)
    emb.reindex(verbose=True)

    # for text, (dsidx, exidx) in tqdm(emb.index.items(), desc="text index"):
    #     try:
    #         emb.datasets[dsidx][exidx]
    #     except:
    #         print("Missing", text, dsidx, exidx)
    #         raise
