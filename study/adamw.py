"""小 GPT：对比 AdamW 每一步的预测损失下降、实际下降及两者之差。

运行：.venv/bin/python study/adamw.py
数据：已编码的小端 uint16 token 分片，默认 ~/.cache/nanochat/tokenized_climbmix。
u = m_hat / (sqrt(v_hat) + eps)，g 是当前 batch 的原始梯度。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    seq_len: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192


class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (a.reshape(b, t, self.n_head, c // self.n_head).transpose(1, 2) for a in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class SmallGPT(nn.Module):
    """Pre-LN、因果 attention、共享输入/输出 embedding，无 dropout。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if min(config.vocab_size, config.seq_len, config.n_layer, config.n_head, config.n_embd) <= 0:
            raise ValueError("GPT 配置必须为正整数")
        if config.n_embd % config.n_head:
            raise ValueError("n_embd 必须能被 n_head 整除")
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.seq_len, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln = nn.LayerNorm(config.n_embd)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * config.n_layer))
            nn.init.normal_(block.mlp[-1].weight, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, targets=None):
        if tokens.size(1) > self.config.seq_len:
            raise ValueError("输入超过模型的 seq_len")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.wte(tokens) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        logits = F.linear(self.ln(x), self.wte.weight)
        if targets is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


class TokenBatches:
    """内存映射预编码 token；按合法窗口数量加权抽取分片，再随机取连续窗口。"""

    def __init__(self, data: Path, batch_size: int, seq_len: int, vocab_size: int, seed: int):
        data = data.expanduser()
        paths = sorted(data.glob("*.bin")) if data.is_dir() else [data]
        self.paths, counts = [], []
        for path in paths:
            size = path.stat().st_size
            if size % 2:
                raise ValueError(f"{path} 字节数不是 2 的倍数，预期无文件头的小端 uint16")
            windows = size // 2 - seq_len
            if windows > 0:
                self.paths.append(path)
                counts.append(windows)
        if not self.paths:
            raise ValueError(f"{data} 没有足够长的 .bin token 分片（至少 seq_len+1 个 token）")
        self.counts = np.asarray(counts, dtype=np.int64)
        self.probabilities = self.counts / self.counts.sum()
        self.rng = np.random.default_rng(seed)
        self.maps = {}
        self.batch_size, self.seq_len, self.vocab_size = batch_size, seq_len, vocab_size

    def next(self, device):
        rows = np.empty((self.batch_size, self.seq_len + 1), dtype=np.int64)
        shards = self.rng.choice(len(self.paths), size=self.batch_size, p=self.probabilities)
        for row, shard in zip(rows, shards):
            shard = int(shard)
            if shard not in self.maps:
                self.maps[shard] = np.memmap(self.paths[shard], dtype="<u2", mode="r")
            start = int(self.rng.integers(self.counts[shard]))
            row[:] = self.maps[shard][start : start + self.seq_len + 1]
        if rows.max() >= self.vocab_size:
            raise ValueError(f"batch 中 token id={rows.max()} 超过 vocab_size={self.vocab_size}，请调整 --vocab-size")
        tokens = torch.from_numpy(rows).to(device)
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


class MeasuredAdamW(torch.optim.Optimizer):
    """显式执行 AdamW，同时统计实际用于本次更新的 u；不裁剪/缩放梯度。"""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("lr 和 eps 必须是有限正数")
        if not math.isfinite(weight_decay) or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("weight_decay 必须非负，betas 必须在 [0, 1) 内")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        totals = None
        for group in self.param_groups:
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise ValueError("本实验只支持稠密梯度")
                state = self.state[p]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                # u 是偏差修正后的更新方向，g 始终保留当前 batch 的原始梯度。
                u = (m / (1 - beta1**t)) / ((v / (1 - beta2**t)).sqrt() + eps)
                g_dot_u = (g * u).sum(dtype=torch.float64)
                decay_prediction = lr * wd * (g * p).sum(dtype=torch.float64)
                values = torch.stack((
                    g.abs().sum(dtype=torch.float64),
                    g_dot_u,
                    lr * g_dot_u,
                    decay_prediction,
                ))
                totals = values if totals is None else totals + values
                # θ_new = (1 - ηλ) θ_old - ηu。
                p.mul_(1 - lr * wd).add_(u, alpha=-lr)
        if totals is None:
            raise RuntimeError("没有可更新的梯度")
        if not torch.isfinite(totals).all().item():
            raise FloatingPointError("梯度或 AdamW 统计出现 NaN/Inf，请减小学习率")
        names = ("grad_l1", "g_dot_u", "predicted_adam_drop", "predicted_decay_drop")
        metrics = dict(zip(names, totals.tolist()))
        metrics["predicted_loss_drop"] = metrics["predicted_adam_drop"] + metrics["predicted_decay_drop"]
        return metrics


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0 else None


def prediction_error_percent(actual, predicted):
    """以实际下降为基准的有符号相对误差；实际为零时无定义。"""
    return 100 * (actual - predicted) / abs(actual) if actual != 0 else None


def summarize(rows):
    if not rows:
        return {"steps": 0}
    drops = np.array([r["loss_drop"] for r in rows])
    predictions = np.array([r["predicted_loss_drop"] for r in rows])
    return {
        "steps": len(rows),
        "pearson_g_dot_u_vs_loss_drop": pearson([r["g_dot_u"] for r in rows], drops),
        "pearson_prediction_vs_loss_drop": pearson(predictions, drops),
        "mean_loss_drop": float(drops.mean()),
        "fraction_loss_decreased": float((drops > 0).mean()),
        "prediction_mean_absolute_error": float(np.abs(predictions - drops).mean()),
        "mean_actual_minus_predicted": float((drops - predictions).mean()),
    }


def plot_metrics(rows, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("未安装 matplotlib，已保存 CSV 和统计；安装后可用 --plot-csv 单独绘图。", flush=True)
        return
    if not rows:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    drops = np.array([r["loss_drop"] for r in rows])
    predictions = np.array([r["predicted_loss_drop"] for r in rows])
    steps = [r["step"] for r in rows]
    axes[0].plot(steps, predictions, label="Predicted drop", linewidth=1.2)
    axes[0].plot(steps, drops, label="Actual drop (same batch)", linewidth=1.2, alpha=0.8)
    axes[0].set(ylabel="Loss drop", title="Per-step predicted and actual loss drop")
    axes[0].legend()
    axes[1].plot(steps, drops - predictions, color="tab:red", linewidth=1)
    axes[1].set(xlabel="Training step", ylabel="Actual - predicted", title="Prediction error (negative = overprediction)")
    for ax in axes:
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(alpha=0.2)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"逐步下降与差值图：{output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/nanochat/tokenized_climbmix")
    parser.add_argument("--out", type=Path, default=Path("study/adamw_runs") / time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--warmup-exclude", type=int, default=20, help="另行统计排除前 N 步后的相关性，不改变学习率")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-csv", type=Path, help="只读取已有 metrics.csv 并生成逐步下降与差值图")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.log_every) <= 0 or args.warmup_exclude < 0:
        parser.error("steps、batch-size、log-every 必须为正，warmup-exclude 必须非负")
    return args


def main():
    args = parse_args()
    if args.plot_csv:
        with args.plot_csv.open(newline="") as handle:
            rows = [{k: float(v) if v else None for k, v in row.items()} for row in csv.DictReader(handle)]
        plot_metrics(rows, args.plot_csv.with_name("relationship.png"))
        return
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请检查 GPU 环境，或使用 --device cpu")
    torch.manual_seed(args.seed)
    # FP32 + 禁用 TF32，减少损失差值测量中的低精度误差；没有 AMP/梯度裁剪。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    config = GPTConfig(args.vocab_size, args.seq_len, args.n_layer, args.n_head, args.n_embd)
    model = SmallGPT(config).to(device)
    optimizer = MeasuredAdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                              eps=args.eps, weight_decay=args.weight_decay)
    data = TokenBatches(args.data, args.batch_size, args.seq_len, args.vocab_size, args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": asdict(config), "parameters": n_params, "device": device,
        "torch_version": torch.__version__, "dtype": "float32",
        "data_files": [str(p) for p in data.paths],
        "measurement": "same batch before/after; no dropout, clipping or AMP; fixed learning rate",
    }
    if device == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name()
        torch.cuda.reset_peak_memory_stats()
    (args.out / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"GPT: {n_params:,} 参数 | {device} FP32 | batch={args.batch_size}, T={args.seq_len}", flush=True)
    print(f"读取 {len(data.paths)} 个预编码分片；结果：{args.out}", flush=True)
    rows = []
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = None
        for step in range(1, args.steps + 1):
            started = time.perf_counter()
            x, y = data.next(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"第 {step} 步 loss 出现 NaN/Inf")
            loss.backward()
            before = loss.item()
            del loss
            metrics = optimizer.step()
            with torch.no_grad():
                after = model(x, y).item()  # 必须复用同一批 x/y；不能比较下一个 batch。
            if not math.isfinite(after):
                raise FloatingPointError(f"第 {step} 步更新后 loss 出现 NaN/Inf")
            drop = before - after  # 保留负值，损失上升也是实验结果。
            row = dict(step=step, loss_before=before, loss_after=after, loss_drop=drop, **metrics,
                       first_order_error=drop - metrics["predicted_loss_drop"],
                       error_percent=prediction_error_percent(drop, metrics["predicted_loss_drop"]),
                       step_seconds=time.perf_counter() - started)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                error_percent = row["error_percent"]
                error_label = f"{error_percent:+.2f}%" if error_percent is not None else "N/A"
                print(f"step {step:4d} | loss {before:.6f} -> {after:.6f} | drop {drop:+.6f} | "
                      f"ηΣg·u {metrics['predicted_adam_drop']:+.6f} | "
                      f"actual-pred {row['first_order_error']:+.6f} | "
                      f"error% {error_label}", flush=True)
    summary = {"all_steps": summarize(rows),
               "after_warmup": summarize(rows[args.warmup_exclude:]),
               "warmup_excluded_steps": args.warmup_exclude}
    if device == "cuda":
        summary["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    torch.save({"model": model.state_dict(), "config": asdict(config), "step": args.steps}, args.out / "model.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.no_plot:
        plot_metrics(rows, args.out / "relationship.png")


if __name__ == "__main__":
    main()
