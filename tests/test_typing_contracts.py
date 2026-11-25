"""Optimizer interfaces and the separate AdamW study remain covered."""
import pytest
import torch
from nanochat.optim import MuonAdamW
from study.adamw import MeasuredAdamW, SmallGPT, GPTConfig as StudyGPTConfig


@pytest.mark.parametrize("measured", [False, True])
def test_optimizer_step_closure_matches_pytorch(measured):
    parameter = torch.nn.Parameter(torch.tensor([1., -2.]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    lr, betas, eps, weight_decay = .01, (.8, .95), 1e-8, .03
    if measured:
        optimizer = MeasuredAdamW([parameter], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        optimizer = MuonAdamW([dict(kind="adamw", params=[parameter], lr=lr,
                                    betas=betas, eps=eps, weight_decay=weight_decay)])
    expected = torch.optim.AdamW([reference], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def closure():
        assert torch.is_grad_enabled()
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    with torch.no_grad():
        loss = optimizer.step(closure)
    assert loss.item() == 5.
    reference.grad = 2 * reference.detach()
    expected.step()
    torch.testing.assert_close(parameter, reference)
    parameter.grad = 2 * parameter.detach()
    reference.grad = 2 * reference.detach()
    assert optimizer.step() is None
    expected.step()
    torch.testing.assert_close(parameter, reference)


def test_measured_step_preserves_stats_and_small_gpt_initialization():
    model = SmallGPT(StudyGPTConfig(vocab_size=16, seq_len=4, n_layer=1, n_head=2, n_embd=8))
    tokens = torch.tensor([[1, 2, 3, 4]])
    loss = model(tokens, tokens)
    loss.backward()
    optimizer = MeasuredAdamW(model.parameters(), lr=.001, weight_decay=.03)
    metrics = optimizer.step_with_metrics()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["predicted_loss_drop"] == pytest.approx(
        metrics["predicted_adam_drop"] + metrics["predicted_decay_drop"])
    assert torch.isfinite(model(tokens, tokens))
