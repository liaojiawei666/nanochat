"""Human-readable parameter report from the same configuration as the runner."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from nanochat.config import GPTConfig, TrainConfig, derive_training
from nanochat.gpt import GPT, resource_estimate


def render_report(depth: int = 24, sequence_len=2048, micro_batch=1) -> str:
    config = GPTConfig.from_depth(depth, sequence_len=sequence_len)
    with torch.device("meta"):
        model = GPT(config)
    recipe = derive_training(model, TrainConfig(micro_batch=micro_batch))
    resources = resource_estimate(model)
    lines = ["# Nanochat 参数与训练配置", "",
             "配置、训练预算及参数统计与训练入口共用同一实现；未分配模型权重。", "",
             "| 模型配置 | 值 |", "| :--- | ---: |"]
    for name, value in {**asdict(config), "n_head": config.n_head}.items():
        lines.append(f"| {name} | {value} |")
    lines += ["", "| 参数分组 | 参数个数 |", "| :--- | ---: |"]
    for name, value in resources["parameter_counts"].items():
        lines.append(f"| {name} | {value:,} |")
    lines += ["", "| 训练推导 | 值 |", "| :--- | ---: |"]
    for name, value in recipe.items():
        lines.append(f"| {name} | {value:,} |")
    lines += ["", f"FP32 权重、梯度与优化器状态估算：{resources['fixed_bytes_estimate'] / 2**30:.2f} GiB。",
              "BF16 计算采用 FP32 参数及状态；上述估算不含激活、工作区和更新临时张量。",
              "实际显存峰值、吞吐和 ETA 由训练日志记录，估算不构成硬件容量保证。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layer", type=int, default=24)
    parser.add_argument("--sequence-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = render_report(args.n_layer, args.sequence_len, args.micro_batch)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
