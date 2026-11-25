"""Model dimensions, training settings and the depth-based training recipe."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from math import ceil, cos, log2, pi, sqrt
from pathlib import Path
import json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


@dataclass
class GPTConfig:
    n_layer: int = 24
    sequence_len: int = 2048
    vocab_size: int = 32768
    head_dim: int = 128
    aspect_ratio: int = 64
    n_embd: int = 0  # Zero means derive from depth.
    n_kv_head: int = 0
    softcap: float = 15.

    def __post_init__(self):
        if min(self.n_layer, self.sequence_len, self.vocab_size, self.head_dim, self.aspect_ratio, self.softcap) <= 0:
            raise ValueError("model dimensions and softcap must be positive")
        if self.n_embd == 0:
            self.n_embd = ceil(self.n_layer * self.aspect_ratio / self.head_dim) * self.head_dim
        if self.head_dim % 2 or self.n_embd < 24 or self.n_embd % self.head_dim:
            raise ValueError("head_dim must be even; n_embd must be >=24 and divisible by head_dim")
        if self.n_kv_head == 0:
            self.n_kv_head = self.n_head
        if self.n_kv_head < 1 or self.n_head % self.n_kv_head:
            raise ValueError("n_kv_head must divide n_head")

    @property
    def n_head(self) -> int:
        return self.n_embd // self.head_dim

    @classmethod
    def from_depth(cls, n_layer: int, **kwargs) -> GPTConfig:
        return cls(n_layer=n_layer, **kwargs)


@dataclass
class StorageConfig:
    data_dir: str = "data/pretrain/climbmix/tokenized"
    archive_path: str = "data/pretrain/climbmix/data.tar.zst"


@dataclass
class TrainConfig:
    output_dir: str = "data/runs/run"
    dtype: str = "auto"
    seed: int = 1337
    micro_batch: int = 1
    total_batch_tokens: int | None = None
    token_param_ratio: float = 8.
    max_steps: int | None = None
    embedding_lr: float = .3
    unembedding_lr: float = .008
    matrix_lr: float = .02
    scalar_lr: float = .5
    weight_decay: float = .28
    warmup_steps: int = 40
    warmdown_ratio: float = .65
    final_lr_frac: float = .05
    eval_every: int = 100
    eval_batches: int = 32
    buffer_size: int = 1000
    log_every: int = 10

    def __post_init__(self):
        if min(self.micro_batch, self.eval_every, self.eval_batches, self.buffer_size, self.log_every) <= 0:
            raise ValueError("batch sizes, buffer and intervals must be positive")
        if not 0 <= self.warmdown_ratio <= 1 or not 0 <= self.final_lr_frac <= 1 or self.warmup_steps < 0:
            raise ValueError("invalid learning rate schedule")
        if self.dtype not in {"auto", "fp32", "bf16"}:
            raise ValueError("dtype must be auto/fp32/bf16")


@dataclass
class Config:
    model: GPTConfig = field(default_factory=GPTConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    platforms: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        sections = json.loads(Path(path if path is not None else DEFAULT_CONFIG_PATH).read_text())
        unknown = sections.keys() - {"model", "storage", "train", "platforms"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(GPTConfig(**sections.get("model", {})), StorageConfig(**sections.get("storage", {})),
                   TrainConfig(**sections.get("train", {})), sections.get("platforms", {}))

    def to_dict(self) -> dict:
        return asdict(self)


def derive_training(model, settings: TrainConfig) -> dict:
    from nanochat.gpt import GPT
    import torch
    counts = model.param_counts()
    p = counts["transform_matrix"] + counts["lm_head"]
    config = model.config
    reference = GPTConfig.from_depth(12, vocab_size=config.vocab_size, head_dim=config.head_dim,
                                     aspect_ratio=config.aspect_ratio, sequence_len=config.sequence_len)
    with torch.device("meta"):
        ref_counts = GPT(reference).param_counts()
    p12 = ref_counts["transform_matrix"] + ref_counts["lm_head"]
    batch = settings.total_batch_tokens
    if batch is None:
        batch = max(config.sequence_len, 2 ** round(log2(2 ** 19 * (p / p12) ** .383)))
    per_micro = settings.micro_batch * config.sequence_len
    if batch <= 0 or batch % per_micro:
        raise ValueError(f"total_batch_tokens={batch} must be divisible by micro_batch*sequence_len={per_micro}")
    target = int(settings.token_param_ratio * p)
    steps = settings.max_steps if settings.max_steps is not None else target // batch
    if steps <= 0 or target <= 0:
        raise ValueError("training budget must include at least one update")
    d_ref = settings.token_param_ratio * p12
    return dict(scaling_params=p, total_params=counts["total"], total_batch_tokens=batch,
                grad_accum_steps=batch // per_micro, target_tokens=target, max_steps=steps,
                training_tokens=steps * batch, batch_lr_scale=sqrt(batch / 2 ** 19),
                weight_decay=settings.weight_decay * sqrt(batch / 2 ** 19) * d_ref / target)


def schedule(step: int, max_steps: int, cfg: TrainConfig) -> dict:
    down = min(round(cfg.warmdown_ratio * max_steps), max_steps - 1)
    warm = min(cfg.warmup_steps, max_steps - down)
    if warm and step < warm:
        lr = (step + 1) / warm
    elif down and step >= max_steps - down:
        fraction = (max_steps - step) / down
        lr = fraction + (1 - fraction) * cfg.final_lr_frac
    else:
        lr = 1.
    momentum = .85 + (.97 - .85) * min(step / 400, 1.)
    if down and step >= max_steps - down:
        momentum = .97 + (.90 - .97) * ((step - (max_steps - down)) / down)
    return dict(lr_multiplier=lr, momentum=momentum,
                decay_multiplier=.5 * (1 + cos(pi * step / max_steps)))
