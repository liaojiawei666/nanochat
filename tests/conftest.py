import base64
from pathlib import Path
import numpy as np
import pytest
import torch
from nanochat.config import Config, GPTConfig, StorageConfig, TrainConfig
from nanochat.tokenizer import SPECIAL_TOKENS


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_config(tmp_path):
    root = tmp_path / "tokenized"
    root.mkdir()
    (root / "vocab.tiktoken").write_text(
        "".join(f"{base64.b64encode(bytes([i])).decode()} {i}\n" for i in range(256)))
    names = [f"shard_{i:05}.bin" for i in range(3)]
    bos_id = SPECIAL_TOKENS["<|bos|>"]
    for name, text in zip(names, (b"abcdefgh", b"ijklmnop", b"qrstuvwx")):
        np.asarray([token for _ in range(12) for token in (bos_id, *text)], dtype="<u2").tofile(root / name)
    return Config(model=GPTConfig.from_depth(1, aspect_ratio=32, head_dim=8, vocab_size=32768, sequence_len=8),
                  storage=StorageConfig(data_dir=str(root), archive_path=str(tmp_path / "data.tar.zst")),
                  train=TrainConfig(output_dir=str(tmp_path / "run"), dtype="fp32",
                                    micro_batch=1, total_batch_tokens=16, max_steps=4,
                                    buffer_size=8, eval_every=2, eval_batches=2, log_every=1))
