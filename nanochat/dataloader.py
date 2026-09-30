"""Climbmix 预编码 token 数据管道：按 BOS 切文档 + best-fit 打包。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch

TOKEN_DIR = Path.home() / ".cache/nanochat/tokenized_climbmix"
# 与 rust-bpe SpecialToken::Bos.id() 一致：VOCAB_NO_SPECIAL_SIZE + 0
BOS_ID = (1 << 15) - 9
Split = Literal["train", "val"]


def list_climbmix_bins(split: Split) -> list[Path]:
    paths = sorted(TOKEN_DIR.glob("shard_*.bin"))
    assert paths, f"no bins in {TOKEN_DIR}"
    return paths[:-1] if split == "train" else paths[-1:]


def split_docs_by_bos(tokens: np.ndarray, bos_id: int = BOS_ID) -> Iterator[np.ndarray]:
    """按 BOS 切出一篇篇文档（每篇以 BOS 开头）。"""
    idx = np.flatnonzero(tokens == bos_id)
    if idx.size == 0:
        return
    for i, start in enumerate(idx):
        end = int(idx[i + 1]) if i + 1 < idx.size else tokens.shape[0]
        yield tokens[start:end].astype(np.int64, copy=False)


class ClimbmixDataset:
    """无限文档流。train 不含最后一个 shard；每完整扫完一轮 paths 后 epoch += 1。"""

    def __init__(self, split: Split):
        self.split = split
        self.paths = list_climbmix_bins(split)
        self.epoch = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            for path in self.paths:
                tokens = np.asarray(np.memmap(path, dtype="<u2", mode="r"))
                yield from split_docs_by_bos(tokens)
            self.epoch += 1


def pack_row(buffer: list[np.ndarray], capacity: int) -> np.ndarray:
    """
    BOS-aligned best-fit：
    1) 能完整放入 remaining 的文档中，取最长者整篇放入；
    2) 否则取最短文档，裁切填满 remaining。
    返回长度恰好为 capacity 的一行。
    """
    assert capacity > 0
    assert buffer, "doc buffer is empty"
    row = np.empty(capacity, dtype=np.int64)
    pos = 0
    while pos < capacity:
        remaining = capacity - pos
        best_i = -1
        best_len = 0
        for i, doc in enumerate(buffer):
            doc_len = int(doc.shape[0])
            if doc_len <= remaining and doc_len > best_len:
                best_i = i
                best_len = doc_len
        if best_i >= 0:
            doc = buffer.pop(best_i)
            row[pos : pos + best_len] = doc
            pos += best_len
        else:
            shortest_i = min(range(len(buffer)), key=lambda i: int(buffer[i].shape[0]))
            doc = buffer.pop(shortest_i)
            row[pos:capacity] = doc[:remaining]
            pos = capacity
    return row


class ClimbmixDataLoader:
    """消费 ClimbmixDataset，用 buffer + best-fit 产出 (inputs[B,T], targets[B,T])。"""

    def __init__(
        self,
        dataset: ClimbmixDataset,
        B: int,
        T: int,
        buffer_size: int = 1000,
        device: str | torch.device = "cuda",
    ):
        assert B > 0 and T > 0 and buffer_size > 0
        self.dataset = dataset
        self.B = B
        self.T = T
        self.buffer_size = buffer_size
        self.device = device
        self._doc_iter = iter(dataset)
        self._buffer: list[np.ndarray] = []

    def _refill(self) -> None:
        while len(self._buffer) < self.buffer_size:
            self._buffer.append(next(self._doc_iter))

    def __iter__(self) -> ClimbmixDataLoader:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        capacity = self.T + 1
        rows = np.empty((self.B, capacity), dtype=np.int64)
        for i in range(self.B):
            self._refill()
            rows[i] = pack_row(self._buffer, capacity)
        x = torch.from_numpy(rows)
        inputs = x[:, :-1].to(self.device, non_blocking=True)
        targets = x[:, 1:].to(self.device, non_blocking=True)
        return inputs, targets

