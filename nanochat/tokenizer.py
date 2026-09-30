import base64
from pathlib import Path

LOCAL_DIR = Path(__file__).resolve().parents[1] / "local"
TIKTOKEN_PATH = LOCAL_DIR / "vocab.tiktoken"


class Tokenizer:
    def __init__(self, mergeable_ranks: dict[bytes, int]):
        self.mergeable_ranks = mergeable_ranks

    @classmethod
    def load_from_path(cls, path: str | Path = TIKTOKEN_PATH) -> "Tokenizer":
        ranks: dict[bytes, int] = {}
        with open(path) as f:
            for line in f:
                token, rank = line.split()
                ranks[base64.b64decode(token)] = int(rank)
        return cls(ranks)
