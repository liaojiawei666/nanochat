from dataclasses import replace
from pathlib import Path
import shutil
import pytest
import torch
from nanochat.eval import load_model, validation
from nanochat.gpt import GPT
from nanochat.optim import MuonAdamW
from nanochat.train import train


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_train_save_and_generate_without_dataset(tiny_config, capsys):
    torch.manual_seed(tiny_config.train.seed)
    initial = GPT(tiny_config.model).transformer.wte.weight.detach().clone()
    output = train(tiny_config)
    assert {p.name for p in output.iterdir()} == {"model.pt", "config.json", "vocab.tiktoken"}
    shutil.rmtree(tiny_config.storage.data_dir)
    model, tokenizer = load_model(output, torch.device("cuda"))
    assert not torch.equal(initial, model.transformer.wte.weight.detach().cpu())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.device.type == "cuda" for p in model.parameters())
    prompt = tokenizer.encode("a", add_bos=True)
    generated = model.generate(prompt, max_tokens=3)
    assert generated[:len(prompt)] == prompt and len(generated) == len(prompt) + 3
    assert isinstance(tokenizer.decode(generated), str)
    log = capsys.readouterr().out
    assert "step=4/4" in log and "step=0 val_loss=" in log and "step=4 val_loss=" in log
    with pytest.raises(FileExistsError):
        train(tiny_config)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gradient_accumulation_matches_larger_batch(tiny_config, tmp_path, monkeypatch):
    gradients = []
    original_step = MuonAdamW.step
    def capture_step(self):
        gradients.append([p.grad.clone() for group in self.param_groups for p in group["params"]])
        return original_step(self)
    monkeypatch.setattr(MuonAdamW, "step", capture_step)
    tiny_config.train.max_steps = 1
    train(tiny_config)
    larger = replace(tiny_config, train=replace(tiny_config.train, micro_batch=2,
                                              output_dir=str(tmp_path / "larger")))
    train(larger)
    # Compare gradients before Adam's tiny epsilon amplifies rounding near zero.
    for accumulated, batched in zip(*gradients):
        torch.testing.assert_close(accumulated, batched, atol=1e-6, rtol=1e-4)


def test_validation_preserves_model_mode_rng_and_gradients(tiny_config):
    model = GPT(tiny_config.model)
    tokens = torch.arange(8).reshape(1, -1)
    model(tokens, tokens).backward()
    grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    first = validation(model, tiny_config.storage.data_dir, tiny_config.train, torch.float32)
    second = validation(model, tiny_config.storage.data_dir, tiny_config.train, torch.float32)
    assert first == second and model.training
    assert torch.equal(rng, torch.get_rng_state())
    for p, grad in zip(model.parameters(), grads):
        if grad is not None:
            torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nonfinite_training_loss_does_not_save_weights(tiny_config, monkeypatch):
    original = GPT.forward
    def forward(self, *args, **kwargs):
        if self.training:
            return self.lm_head.weight.sum() * float("nan")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(GPT, "forward", forward)
    with pytest.raises(FloatingPointError, match="training loss"):
        train(tiny_config)
    assert not (Path(tiny_config.train.output_dir) / "model.pt").exists()
