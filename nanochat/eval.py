"""Validation loss and loading final weights for text generation."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from nanochat.config import GPTConfig
from nanochat.dataloader import ClimbmixDataset, ClimbmixDataLoader
from nanochat.gpt import GPT, autocast_context
from nanochat.tokenizer import Tokenizer


@torch.no_grad()
def validation(model: GPT, data_dir: str | Path, settings, dtype) -> float:
    device = next(model.parameters()).device
    loader = ClimbmixDataLoader(ClimbmixDataset("val", data_dir), settings.micro_batch,
                               model.sequence_len, settings.buffer_size, device)
    training = model.training
    model.eval()
    total = torch.zeros((), device=device)
    try:
        with autocast_context(device, dtype):
            for _ in range(settings.eval_batches):
                x, y = next(loader)
                total += model(x, y)
        result = float(total) / settings.eval_batches
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite validation loss")
        return result
    finally:
        model.train(training)


def load_model(output_dir: str | Path, device) -> tuple[GPT, Tokenizer]:
    root = Path(output_dir)
    saved = json.loads((root / "config.json").read_text())
    model = GPT(GPTConfig(**saved["model"])).to(device)
    model.load_state_dict(torch.load(root / "model.pt", map_location=device, weights_only=True))
    tokenizer = Tokenizer.load_from_path(root / "vocab.tiktoken", saved["special_tokens"])
    return model.eval(), tokenizer
