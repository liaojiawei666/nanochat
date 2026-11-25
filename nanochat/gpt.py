"""Causal GPT with value embeddings and smear; FP32 storage/BF16 computation."""

from __future__ import annotations
import math
from contextlib import nullcontext
from typing import Any, Iterable
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from nanochat.config import GPTConfig
from nanochat.optim import MuonAdamW


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x.float(), (x.size(-1),)).to(x.dtype)


def resolve_runtime(dtype: str = "auto") -> tuple[torch.device, torch.dtype]:
    """Training and generation require CUDA; never fall back to CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; check the NVIDIA driver and CUDA-enabled PyTorch")
    supported = torch.cuda.is_bf16_supported()
    if dtype == "bf16" and not supported:
        raise ValueError("BF16 requested but unsupported; use dtype=fp32")
    return torch.device("cuda"), torch.bfloat16 if supported and dtype != "fp32" else torch.float32


def autocast_context(device: torch.device, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype == torch.bfloat16 else nullcontext()


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)).square())


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, head_dim: int, n_kv_head: int, has_ve: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.W_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = nn.Linear(12, n_kv_head, bias=False) if has_ve else None

    def apply_rope(self, x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

    def forward(self, x, cos, sin, ve = None):
        b, t, _ = x.shape
        q = self.apply_rope(self.W_q(x).reshape(b, t, -1, self.head_dim), cos, sin)
        k = self.apply_rope(self.W_k(x).reshape(b, t, -1, self.head_dim), cos, sin)
        v = self.W_v(x).reshape(b, t, -1, self.head_dim)
        if ve is not None:
            if self.ve_gate is None:
                raise ValueError("value embeddings require a value gate")
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :12]))
            v = v + gate.unsqueeze(-1) * ve.reshape(b, t, -1, self.head_dim).to(v.dtype)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=True
        )
        return self.W_o(y.transpose(1, 2).contiguous().reshape(b, t, -1))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, head_dim, n_kv_head, has_ve=False):
        super().__init__()
        self.atten = CausalSelfAttention(n_embd, head_dim, n_kv_head, has_ve)
        self.mlp = MLP(n_embd)

    def forward(self, x, cos, sin, ve=None):
        x = x + self.atten(norm(x), cos, sin, ve)
        return x + self.mlp(norm(x))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


class Transformer(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList(
            [
                TransformerBlock(config.n_embd, config.head_dim, config.n_kv_head, has_ve(i, config.n_layer))
                for i in range(config.n_layer)
            ]
        )


class GPT(nn.Module):
    cos: Tensor
    sin: Tensor

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.sequence_len, self.head_dim, self.n_layer = config.sequence_len, config.head_dim, config.n_layer
        self.transformer = Transformer(config)
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, config.n_kv_head * config.head_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.smear_gate = nn.Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        channel = torch.arange(0, config.head_dim, 2, dtype=torch.float32)
        freqs = torch.outer(
            torch.arange(config.sequence_len, dtype=torch.float32), 100000 ** (-channel / config.head_dim)
        )
        self.register_buffer("cos", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, :, None, :], persistent=False)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.normal_(self.transformer.wte.weight, std=0.8)
        nn.init.normal_(self.lm_head.weight, std=0.001)
        bound = math.sqrt(3) * self.config.n_embd**-0.5
        for block in self.transformer.h:
            if not isinstance(block, TransformerBlock):
                raise TypeError("expected TransformerBlock in transformer.h")
            for layer in (block.atten.W_q, block.atten.W_k, block.atten.W_v):
                nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(block.atten.W_o.weight)
            nn.init.uniform_(block.mlp.fc1.weight, -bound * 0.4, bound * 0.4)
            nn.init.zeros_(block.mlp.fc2.weight)
            if block.atten.ve_gate is not None:
                nn.init.uniform_(block.atten.ve_gate.weight, 0.0, 0.02)
        for table in self.value_embeds.values():
            if not isinstance(table, nn.Embedding):
                raise TypeError("expected Embedding in value_embeds")
            nn.init.uniform_(table.weight, -bound, bound)
        for i in range(self.n_layer):
            self.resid_lambdas[i] = 1.15 - 0.1 * i / max(self.n_layer - 1, 1)
            self.x0_lambdas[i] = 0.2 - 0.15 * i / max(self.n_layer - 1, 1)
        nn.init.zeros_(self.smear_lambda)
        nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

    def param_counts(self) -> dict[str, int]:
        count = lambda module: sum(p.numel() for p in module.parameters())
        counts = dict(
            wte=count(self.transformer.wte),
            transform_matrix=count(self.transformer.h),
            value_embeds=count(self.value_embeds),
            lm_head=count(self.lm_head),
            resid_lambdas=self.resid_lambdas.numel(),
            x0_lambdas=self.x0_lambdas.numel(),
            smear_gate=count(self.smear_gate),
            smear_lambda=self.smear_lambda.numel(),
        )
        total = sum(p.numel() for p in self.parameters())
        assert sum(counts.values()) == total
        return {**counts, "total": total}

    def module_report(self) -> dict:
        return {
            name: {"params": sum(p.numel() for p in module.parameters(recurse=False))}
            for name, module in self.named_modules()
            if list(module.parameters(recurse=False))
        }

    def setup_optimizer(
        self,
        *,
        embedding_lr=0.3,
        unembedding_lr=0.008,
        matrix_lr=0.02,
        scalar_lr=0.5,
        weight_decay=0.28,
        batch_lr_scale=1.0,
        use_bf16=False,
    ) -> MuonAdamW:
        width_scale = (self.config.n_embd / 768) ** -0.5

        def adam(params: Iterable[Tensor], lr: float, betas: tuple[float, float], wd: float) -> dict[str, Any]:
            return dict(
                kind="adamw", params=list(params), lr=lr * batch_lr_scale, betas=betas, eps=1e-10, weight_decay=wd
            )

        groups: list[dict[str, Any]] = [
            adam(self.lm_head.parameters(), unembedding_lr * width_scale, (0.8, 0.96), 0.01),
            adam(self.transformer.wte.parameters(), embedding_lr * width_scale, (0.8, 0.995), 0.001),
            adam(self.value_embeds.parameters(), embedding_lr * width_scale * 0.5, (0.8, 0.995), 0.01),
            adam([self.resid_lambdas], scalar_lr * 0.01, (0.8, 0.95), 0.05),
            adam([self.x0_lambdas], scalar_lr, (0.96, 0.95), 0.0),
            adam([self.smear_gate.weight, self.smear_lambda], 0.2, (0.8, 0.95), 0.0),
        ]
        matrices = list(self.transformer.h.parameters())
        for shape in sorted({p.shape for p in matrices}):
            groups.append(
                dict(
                    kind="muon",
                    params=[p for p in matrices if p.shape == shape],
                    lr=matrix_lr * batch_lr_scale,
                    momentum=0.95,
                    beta2=0.9,
                    ns_steps=5,
                    weight_decay=weight_decay,
                )
            )
        assigned = [p for g in groups for p in g["params"]]
        assert len({id(p) for p in assigned}) == len(list(self.parameters())) == len(assigned)
        return MuonAdamW(groups, use_bf16=use_bf16)

    def forward(self, idx: Tensor, targets: Tensor | None = None, loss_reduction: str = "mean") -> Tensor:
        b, t = idx.shape
        if not 1 <= t <= self.sequence_len:
            raise ValueError("sequence length outside model context")
        x = self.transformer.wte(idx)
        if torch.is_autocast_enabled(idx.device.type):
            x = x.to(torch.get_autocast_dtype(idx.device.type))
        x = norm(x)
        if t > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i].to(x.dtype) * x + self.x0_lambdas[i].to(x.dtype) * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, self.cos[:, :t], self.sin[:, :t], ve)
        logits = self.lm_head(norm(x)).float()
        logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction=loss_reduction, ignore_index=-1
        )
        return loss.reshape(b, t) if loss_reduction == "none" else loss

    @torch.no_grad()
    def generate(self, tokens: list[int], max_tokens=32, temperature=0.0, generator=None) -> list[int]:
        if not tokens or temperature < 0 or max_tokens < 0:
            raise ValueError("invalid generation inputs")
        result = list(tokens)
        for _ in range(max_tokens):
            idx = torch.tensor([result[-self.sequence_len :]], device=self.lm_head.weight.device)
            logits = self(idx)[0, -1]
            token = (
                logits.argmax()
                if temperature == 0
                else torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=generator)[0]
            )
            result.append(int(token))
        return result


def resource_estimate(model: GPT, train_cfg=None) -> dict:
    """FP32 weights + gradients + actual-shaped optimizer states, excludes temporaries."""
    c = model.param_counts()
    matrices = list(model.transformer.h.parameters())
    adam = c["total"] - c["transform_matrix"]
    reduced = sum(max(p.shape) for p in matrices)
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    fixed = 8 * c["total"] + 8 * adam + 4 * (c["transform_matrix"] + reduced) + buffers
    return {
        "parameter_counts": c,
        "fixed_bytes_estimate": fixed,
        "excludes": "activations, workspace and temporary updates",
    }
