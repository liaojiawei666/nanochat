"""Single-device training: prepare, update, validate, save final weights."""
from __future__ import annotations
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from nanochat.config import Config, derive_training, schedule
from nanochat.dataloader import ClimbmixDataLoader, ClimbmixDataset
from nanochat.eval import validation
from nanochat.gpt import GPT, autocast_context, resolve_runtime
from nanochat.tokenizer import Tokenizer


def train(cfg: Config) -> Path:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this training loop supports one process/device")
    output = Path(cfg.train.output_dir)
    if (output / "model.pt").exists():
        raise FileExistsError(f"choose a new output_dir; weights already exist at {output}")
    device, dtype = resolve_runtime(cfg.train.dtype)
    data_dir = Path(cfg.storage.data_dir)
    tokenizer = Tokenizer.load_from_path(data_dir / "vocab.tiktoken")
    model_cfg = replace(cfg.model, vocab_size=tokenizer.get_vocab_size())
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    model = GPT(model_cfg).to(device)  # Parameters stay on this device throughout training.
    recipe = derive_training(model, cfg.train)
    optimizer = model.setup_optimizer(
        embedding_lr=cfg.train.embedding_lr, unembedding_lr=cfg.train.unembedding_lr,
        matrix_lr=cfg.train.matrix_lr, scalar_lr=cfg.train.scalar_lr,
        weight_decay=recipe["weight_decay"], batch_lr_scale=recipe["batch_lr_scale"],
        use_bf16=dtype == torch.bfloat16)
    loader = ClimbmixDataLoader(ClimbmixDataset("train", data_dir), cfg.train.micro_batch,
                               model.sequence_len, cfg.train.buffer_size, device)
    accumulation, steps = recipe["grad_accum_steps"], recipe["max_steps"]
    print(f"device={device} compute={dtype} params={model.param_counts()['total']:,} "
          f"steps={steps} accumulation={accumulation}", flush=True)
    initial_loss = validation(model, data_dir, cfg.train, dtype)
    print(f"step=0 val_loss={initial_loss:.6f}", flush=True)
    model.train()
    for step in range(steps):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for _ in range(accumulation):
            x, y = next(loader)  # The loader has already transferred this batch to device.
            with autocast_context(device, dtype):
                loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss_sum += loss.detach()
            (loss / accumulation).backward()
        norms = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
        if not torch.isfinite(norms).all():
            raise FloatingPointError("non-finite gradient")
        values = schedule(step, steps, cfg.train)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * values["lr_multiplier"]
            if group["kind"] == "muon":
                group["momentum"] = values["momentum"]
                group["weight_decay"] = recipe["weight_decay"] * values["decay_multiplier"]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        duration = time.monotonic() - tick
        if (step + 1) % cfg.train.log_every == 0 or step == 0 or step + 1 == steps:
            print(f"step={step + 1}/{steps} train_loss={float(loss_sum) / accumulation:.6f} "
                  f"lr={optimizer.param_groups[0]['lr']:.4g} "
                  f"tokens/s={recipe['total_batch_tokens'] / duration:.0f}", flush=True)
        if (step + 1) % cfg.train.eval_every == 0 or step + 1 == steps:
            score = validation(model, data_dir, cfg.train, dtype)
            print(f"step={step + 1} val_loss={score:.6f}", flush=True)
    # Save a portable final model, without optimizer state or training-resume machinery.
    output.mkdir(parents=True, exist_ok=True)
    saved = dict(model=vars(model_cfg), train=vars(cfg.train), special_tokens=tokenizer.special_tokens)
    (output / "config.json").write_text(json.dumps(saved, indent=2) + "\n")
    shutil.copyfile(data_dir / "vocab.tiktoken", output / "vocab.tiktoken")
    weights = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(weights, output / "model.pt")
    print(f"saved final model to {output}", flush=True)
    return output
