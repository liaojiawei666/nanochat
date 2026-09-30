import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

# 累计训练 token ≈ ratio × 参数量（对齐 nanochat 量级，可再调）
TOKEN_PARAM_RATIO = 8


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
        n_atten_params = 2 * self.n_embd**2 + 2 * self.n_embd * (
            self.n_kv_head * self.head_dim
        )
        n_mlp_params = 2 * self.n_embd * (4 * self.n_embd)
        n_lm_head_params = self.n_embd * self.vocab_size
        # 不计算wte，ve，rope等参数
        self.n_scalar_params = (
            n_layer * (n_mlp_params + n_atten_params) + n_lm_head_params
        )
        self.n_tokens = TOKEN_PARAM_RATIO * self.n_scalar_params


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
    def __init__(
        self, n_embd: int, head_dim: int, n_kv_head: int, has_ve: bool = False
    ) -> None:
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

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, ve: Tensor | None = None
    ) -> Tensor:
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
    def __init__(
        self, n_embd: int, head_dim: int, n_kv_head: int, has_ve: bool = False
    ) -> None:
        super().__init__()
        self.atten = CausalSelfAttention(n_embd, head_dim, n_kv_head, has_ve)
        self.mlp = MLP(n_embd)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, ve: Tensor | None = None
    ) -> Tensor:
        x = x + self.atten(norm(x), cos, sin, ve)
        x = x + self.mlp(norm(x))
        return x


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """隔层启用 value embedding，最后一层一定有。"""
    return layer_idx % 2 == (n_layer - 1) % 2


class GPT(nn.Module):
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

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                TranformerBlock(n_embd, head_dim, n_kv_head, has_ve=has_ve(i, n_layer))
                for i in range(n_layer)
            ]
        )
        # 每层（若启用）一张表：token id → value 向量
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(vocab_size, kv_dim)
                for i in range(n_layer)
                if has_ve(i, n_layer)
            }
        )

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

    def forward(self, idx: Tensor) -> Tensor:
        # idx: (B, T) token ids
        B, T = idx.shape
        assert T <= self.sequence_len
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        x = self.wte(idx)  # (B, T, n_embd)
        for i, block in enumerate(self.blocks):
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, cos, sin, ve)
        x = norm(x)
        logits = self.lm_head(x)
        return logits
