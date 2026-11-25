"""Read Rust-encoded uint16 shards and pack documents into training batches."""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from nanochat.tokenizer import SPECIAL_TOKENS


def list_token_shards(root: str | Path) -> list[Path]:
    """The final sorted shard is validation; tokens are little-endian uint16."""
    root = Path(root)
    if not (root / "vocab.tiktoken").is_file():
        raise ValueError(f"missing tokenizer: {root / 'vocab.tiktoken'}")
    shards = sorted(path for path in root.glob("shard_*.bin") if path.is_file())
    if len(shards) < 2:
        raise ValueError("need at least two token shards for train/validation isolation")
    for path in shards:
        size = path.stat().st_size
        if size == 0 or size % 2:
            raise ValueError(f"expected nonempty uint16 token shard: {path}")
    return shards


class ClimbmixDataset:
    def __init__(self, split: Literal["train", "val"], root: str | Path):
        self.root = Path(root)
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        shards = list_token_shards(root)
        self.paths = shards[:-1] if split == "train" else shards[-1:]
        self.bos_id = SPECIAL_TOKENS["<|bos|>"]

    def __iter__(self):
        while True:
            for path in self.paths:
                tokens = np.memmap(path, mode="r", dtype="<u2")
                # Scan in chunks so indexing a large shard does not allocate a huge mask.
                starts = np.concatenate([np.flatnonzero(tokens[i:i + 4_000_000] == self.bos_id) + i
                                         for i in range(0, len(tokens), 4_000_000)])
                if not len(starts) or starts[0] != 0:
                    raise ValueError(f"shard must start with BOS: {path}")
                ends = np.append(starts[1:], len(tokens))
                for start, end in zip(starts, ends):
                    yield tokens[start:end]


class ClimbmixDataLoader:
    def __init__(self, dataset: ClimbmixDataset, B: int, T: int, buffer_size: int, device):
        if min(B, T, buffer_size) <= 0:
            raise ValueError("batch size, sequence length and buffer must be positive")
        self.documents = iter(dataset)
        self.B, self.T, self.buffer_size, self.device = B, T, buffer_size, device
        self.buffer: list[np.ndarray] = []

    def __iter__(self):
        return self

    def __next__(self):
        rows = np.empty((self.B, self.T + 1), dtype=np.int64)
        for row in rows:
            while len(self.buffer) < self.buffer_size:
                self.buffer.append(next(self.documents))
            pos = 0
            while pos < len(row):
                if not self.buffer:
                    self.buffer.extend(next(self.documents) for _ in range(self.buffer_size))
                remaining = len(row) - pos
                fits = [i for i, doc in enumerate(self.buffer) if len(doc) <= remaining]
                index = (max(fits, key=lambda i: len(self.buffer[i])) if fits else
                         min(range(len(self.buffer)), key=lambda i: len(self.buffer[i])))
                doc = self.buffer.pop(index)
                count = min(len(doc), remaining)
                row[pos:pos + count] = doc[:count]
                pos += count
        tokens = torch.from_numpy(rows)  # CPU; transfer only the current batch to the training device.
        return tokens[:, :-1].to(self.device), tokens[:, 1:].to(self.device)
