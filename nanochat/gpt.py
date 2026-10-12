import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from nanochat.optim import AdamWGroup, MuonGroup, ParamGroup



class GPTConfig:
    def __init__(self, n_layer: int) -> None:
        self.n_layer = n_layer
        self.aspect_ratio = 64
        self.vocab_size = 1 << 15
        self.sequence_len = 2048
        self.head_dim = 128
        self.batch = 4
        self.softcap = 15
        self.n_embd = self.aspect_ratio * self.n_layer
        self.n_kv_head = self.n_embd // self.head_dim


class MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.relu(x) ** 2
        x = self.fc2(x)
        return x


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, head_dim: int, n_kv_head: int, has_ve: bool = False) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.W_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        # attention 输出与 Q 同宽：n_head * head_dim == n_embd
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)
        if has_ve:
            self.ve_gate_channels = 12
            self.ve_gate = nn.Linear(self.ve_gate_channels, n_kv_head, bias=False)

    def apply_rope(self, x, cos, sin) -> Tensor:
        # x: [B,T,H,D],cos/sin:[1,T,1,D/2]
        mid = self.head_dim // 2
        x1 = x[..., :mid]
        x2 = x[..., mid:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.concat((y1, y2), dim=-1)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, ve: Tensor | None = None) -> Tensor:
        # x:[B,T,n_embd]
        B, T, _ = x.shape
        # q:[B,T,n_head,head_dim]
        q = self.W_q(x).view(B, T, -1, self.head_dim)
        q = self.apply_rope(q, cos, sin)
        # k,v:[B,T,n_kv_head,head_dim]
        k = self.W_k(x).view(B, T, -1, self.head_dim)
        k = self.apply_rope(k, cos, sin)
        v = self.W_v(x).view(B, T, -1, self.head_dim)
        if ve is not None:
            # gate:[B,T,n_kv_head],v:[B,T,n_kv_head,head_dim]
            gate = 3 * F.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            ve = ve.view(B, T, -1, self.head_dim)
            v = v + gate.unsqueeze(-1) * ve
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        )

        # a:[B,T,n_embd]
        a = a.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(a)


class TranformerBlock(nn.Module):
    def __init__(self, n_embd: int, head_dim: int, n_kv_head: int, has_ve: bool = False) -> None:
        super().__init__()
        self.atten = CausalSelfAttention(n_embd, head_dim, n_kv_head, has_ve)
        self.mlp = MLP(n_embd)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, ve: Tensor | None = None) -> Tensor:
        x = x + self.atten(norm(x), cos, sin, ve)
        x = x + self.mlp(norm(x))
        return x


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """隔层启用 value embedding，最后一层一定有。"""
    return layer_idx % 2 == (n_layer - 1) % 2


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, head_dim: int, n_kv_head: int, n_layer: int) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.h = nn.ModuleList(
            [TranformerBlock(n_embd, head_dim, n_kv_head, has_ve=has_ve(i, n_layer)) for i in range(n_layer)]
        )


class GPT(nn.Module):
    cos: Tensor
    sin: Tensor

    def __init__(
        self,
        sequence_len: int,
        vocab_size: int,
        n_embd: int,
        head_dim: int,
        n_kv_head: int,
        n_layer: int,
    ) -> None:
        super().__init__()
        self.sequence_len = sequence_len
        self.head_dim = head_dim
        self.n_layer = n_layer
        kv_dim = n_kv_head * head_dim

        self.transformer = Transformer(vocab_size, n_embd, head_dim, n_kv_head, n_layer)
        # 每层（若启用）一张表：token id → value 向量
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(vocab_size, kv_dim) for i in range(n_layer) if has_ve(i, n_layer)}
        )
        # 每层一个标量：混入初始 embedding；真实初值可在 init_weights 里再设
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        # Smear：全局一份，用前一 token 的 embedding 混进当前位（廉价 bigram）
        self.smear_gate = nn.Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))

        cos, sin = self._precompute_rope()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def _precompute_rope(self) -> tuple[Tensor, Tensor]:
        channel = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (100000 ** (channel / self.head_dim))
        t = torch.arange(self.sequence_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()  # (T, head_dim/2)
        sin = freqs.sin()
        # (1, T, 1, head_dim/2)，广播到 (B, T, H, D/2)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def param_counts(self) -> dict:
        """返回各模块参数量。"""

        def nparams(module: nn.Module | nn.Parameter) -> int:
            if isinstance(module, nn.Parameter):
                return module.numel()
            return sum(p.numel() for p in module.parameters())

        counts = {
            "wte": nparams(self.transformer.wte),
            "transform_matrix": nparams(self.transformer.h),
            "value_embeds": nparams(self.value_embeds),
            "resid_lambdas": nparams(self.resid_lambdas),
            "x0_lambdas": nparams(self.x0_lambdas),
            "smear_gate": nparams(self.smear_gate),
            "smear_lambda": nparams(self.smear_lambda),
            "lm_head": nparams(self.lm_head),
        }
        counts["total"] = nparams(self)
        return counts

    def set_optim(self):
        param_groups: list[ParamGroup] = [
            AdamWGroup(
                kind="adamw",
                params=list(self.transformer.wte.parameters()),
                lr=0.01,
                betas=(0.8, 0.96),
                eps=1e-10,
                weight_decay=0.1,
            ),
            AdamWGroup(
                kind="adamw",
                params=list(self.value_embeds.parameters()),
                lr=0.01,
                betas=(0.8, 0.96),
                eps=1e-10,
                weight_decay=0.1,
            ),
            AdamWGroup(
                kind="adamw",
                params=list(self.lm_head.parameters()),
                lr=0.01,
                betas=(0.8, 0.96),
                eps=1e-10,
                weight_decay=0.1,
            ),
            AdamWGroup(
                kind="adamw", params=[self.resid_lambdas], lr=0.01, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.1
            ),
            AdamWGroup(kind="adamw", params=[self.x0_lambdas], lr=0.01, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.1),
            AdamWGroup(
                kind="adamw",
                params=list(self.smear_gate.parameters()),
                lr=0.01,
                betas=(0.8, 0.96),
                eps=1e-10,
                weight_decay=0.1,
            ),
            AdamWGroup(
                kind="adamw", params=[self.smear_lambda], lr=0.01, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.1
            ),
        ]
        matrix_params = list(self.transformer.h.parameters())
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                MuonGroup(kind="muon", params=group_params, lr=0.02, beta=0.95, eps=1e-10, ns_steps=5, weight_decay=0.0)
            )

    def forward(self, idx: Tensor) -> Tensor:
        # idx: (B, T) token ids
        B, T = idx.shape
        assert T <= self.sequence_len
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        x = norm(self.transformer.wte(idx))  # (B, T, n_embd)
        # smear：位置 0 不动；1..T-1 混入上一个 token
        assert T > 1
        gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, cos, sin, ve)
        x = norm(x)
        logits = self.lm_head(x)
        return logits



