"""tiktoken inference matching the Rust vocabulary, regex and special IDs."""
from __future__ import annotations
import base64
from pathlib import Path
from typing import overload
import torch

LOCAL_DIR = Path(__file__).resolve().parents[1] / "local"
TIKTOKEN_PATH = LOCAL_DIR / "vocab.tiktoken"
SPECIAL_NAMES = ("<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
                 "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>")
# Match rust-bpe/src/special.rs, including when loading a smaller test vocabulary.
SPECIAL_TOKENS = {name: (1 << 15) - len(SPECIAL_NAMES) + i for i, name in enumerate(SPECIAL_NAMES)}
# Kept byte-for-byte equivalent to rust-bpe/src/bpe/pretok.rs.
SPLIT_PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"


class Tokenizer:
    def __init__(self, mergeable_ranks: dict[bytes, int], special_tokens: dict[str, int] | None = None):
        import tiktoken
        self.mergeable_ranks = mergeable_ranks
        if sorted(mergeable_ranks.values()) != list(range(len(mergeable_ranks))):
            raise ValueError("vocabulary ranks must be contiguous")
        self.special_tokens = dict(SPECIAL_TOKENS if special_tokens is None else special_tokens)
        if set(self.special_tokens) != set(SPECIAL_NAMES):
            raise ValueError("tokenizer must define the nine Rust special tokens")
        if min(self.special_tokens.values()) < len(mergeable_ranks) or len(set(self.special_tokens.values())) != 9:
            raise ValueError("special IDs overlap ordinary tokens or each other")
        self.enc = tiktoken.Encoding(name="nanochat-custom", pat_str=SPLIT_PATTERN,
                                    mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)

    @classmethod
    def load_from_path(cls, path: str | Path = TIKTOKEN_PATH, special_tokens=None) -> "Tokenizer":
        ranks = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                token, rank = line.split()
                raw = base64.b64decode(token, validate=True)
                if raw in ranks:
                    raise ValueError("duplicate vocabulary token")
                ranks[raw] = int(rank)
        return cls(ranks, special_tokens)

    @overload
    def encode(self, text: str, add_bos: bool = False, prepend=None, append=None) -> list[int]: ...

    @overload
    def encode(self, text: list[str], add_bos: bool = False, prepend=None, append=None) -> list[list[int]]: ...

    def encode(self, text: str | list[str], add_bos: bool = False, prepend=None, append=None) -> list[int] | list[list[int]]:
        if isinstance(text, list):
            return [self.encode(value, add_bos=add_bos, prepend=prepend, append=append) for value in text]
        tokens = self.enc.encode_ordinary(text)
        if add_bos:
            prepend = "<|bos|>"
        if prepend is not None:
            tokens.insert(0, self.special_tokens[prepend] if isinstance(prepend, str) else prepend)
        if append is not None:
            tokens.append(self.special_tokens[append] if isinstance(append, str) else append)
        return tokens

    __call__ = encode

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.special_tokens["<|bos|>"]

    def encode_special(self, name: str) -> int:
        return self.special_tokens[name]

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return self.enc.decode_single_token_bytes(token_id)

    def token_bytes(self, device="cpu") -> torch.Tensor:
        lengths = torch.zeros(self.get_vocab_size(), dtype=torch.int64)
        for raw, rank in self.mergeable_ranks.items():
            lengths[rank] = len(raw)
        return lengths.to(device)
