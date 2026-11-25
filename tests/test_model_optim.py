import copy
import pytest
import torch
from nanochat.config import GPTConfig, TrainConfig, derive_training, schedule
from nanochat.gpt import GPT, resolve_runtime, autocast_context
from nanochat.optim import MuonAdamW, polar_express, muon_update, adamw_update
from tests.reference_optim import muon_step_fused


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_depth_shapes_counts_causality_and_gradients(depth):
    config = GPTConfig.from_depth(depth, aspect_ratio=32, head_dim=8, sequence_len=8, vocab_size=265)
    model = GPT(config)
    counts = model.param_counts()
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert sum(v for k, v in counts.items() if k != "total") == counts["total"]
    assert sum(row["params"] for row in model.module_report().values()) == counts["total"]
    assert model.get_submodule("transformer.h.0.atten") is model.transformer.h[0].atten
    x = torch.randint(0, 256, (2, 8))
    y = x.clone()
    y[:, 5:] = torch.randint(0, 256, (2, 3))
    torch.testing.assert_close(model(x)[:, :5], model(y)[:, :5], rtol=0, atol=0)
    loss = model(x, y)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model(torch.zeros((1, 1), dtype=torch.long)).shape == (1, 1, 265)
    assert len(model.generate([256], max_tokens=2)) == 3


def test_odd_depth_rounds_width_and_training_recipe():
    config = GPTConfig.from_depth(3)
    assert config.n_embd == 256 and config.n_head == 2
    with torch.device("meta"):
        model = GPT(config)
    recipe = derive_training(model, TrainConfig(micro_batch=1))
    assert recipe["target_tokens"] == 8 * recipe["scaling_params"]
    assert recipe["training_tokens"] == recipe["max_steps"] * recipe["total_batch_tokens"]
    with pytest.raises(ValueError, match="divisible"):
        derive_training(model, TrainConfig(total_batch_tokens=17))
    for steps in (1, 2, 3, 10, 100):
        values = [schedule(i, steps, TrainConfig()) for i in range(steps)]
        assert all(0 < v["lr_multiplier"] <= 1 for v in values)


def test_adamw_matches_pytorch_and_live_groups_after_restore():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(4, 3))
    q = torch.nn.Parameter(p.detach().clone())
    group = dict(kind="adamw", params=[p], lr=.01, betas=(.8, .96), eps=1e-10, weight_decay=.03)
    actual = MuonAdamW([group])
    expected = torch.optim.AdamW([q], lr=.01, betas=(.8, .96), eps=1e-10, weight_decay=.03)
    for _ in range(4):
        grad = torch.randn_like(p)
        p.grad, q.grad = grad.clone(), grad.clone()
        actual.step()
        expected.step()
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)
    restored_p = torch.nn.Parameter(p.detach().clone())
    restored = MuonAdamW([{**group, "params": [restored_p], "lr": 9.}])
    restored.load_state_dict(copy.deepcopy(actual.state_dict()))
    actual.param_groups[0]["lr"] = restored.param_groups[0]["lr"] = .002
    grad = torch.randn_like(p)
    p.grad, restored_p.grad = grad.clone(), grad.clone()
    actual.step()
    restored.step()
    torch.testing.assert_close(p, restored_p, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(4, 4), (7, 3), (3, 7)])
def test_muon_matches_pinned_upstream(shape):
    torch.manual_seed(4)
    grad, params = torch.randn(shape), torch.randn(shape)
    original_grad = grad.clone()
    first = torch.randn(shape) * .1
    second = torch.ones((shape[0], 1) if shape[0] >= shape[1] else (1, shape[1])) * .1
    expected_p, expected_g, expected_m, expected_v = [v.clone() for v in (params, grad, first, second)]
    scalar = lambda value: torch.tensor(value, dtype=torch.float32)
    lr = .01 * max(1., shape[0] / shape[1]) ** .5
    muon_step_fused(expected_g, expected_p, expected_m, expected_v, scalar(.95), scalar(lr),
                    scalar(.03), scalar(.9), 5, -1 if shape[0] >= shape[1] else -2)
    update, actual_m, actual_v = muon_update(grad, first, second)
    actual_p = params - lr * update - lr * .03 * params * ((update * params) >= 0)
    torch.testing.assert_close(actual_p, expected_p, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual_m, expected_m)
    torch.testing.assert_close(actual_v, expected_v)
    assert torch.equal(grad, original_grad)


@pytest.mark.parametrize("shape", [(4, 4), (8, 3), (3, 8)])
def test_polar_svd_orientation_zero_and_degenerate(shape):
    torch.manual_seed(3)
    matrix = torch.randn(shape)
    q = polar_express(matrix)
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    in_basis = u.mT @ q @ vh.mT
    torch.testing.assert_close(in_basis, torch.diag(in_basis.diag()), atol=1e-5, rtol=1e-5)
    assert torch.all(in_basis.diag() > 0) and torch.linalg.svdvals(q).max() < 1.7
    assert torch.equal(polar_express(torch.zeros(shape)), torch.zeros(shape))
    degenerate = torch.zeros(shape)
    degenerate[0, 0] = 1.
    assert torch.isfinite(polar_express(degenerate)).all()
    with pytest.raises(ValueError):
        polar_express(matrix, 6)


def test_runtime_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        resolve_runtime()


def test_runtime_precision_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_runtime() == (torch.device("cuda"), torch.float32)
    with pytest.raises(ValueError, match="unsupported"):
        resolve_runtime("bf16")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_runtime() == (torch.device("cuda"), torch.bfloat16)
    assert resolve_runtime("fp32") == (torch.device("cuda"), torch.float32)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="requires CUDA BF16")
def test_cuda_bf16_forward_backward_and_update():
    model = GPT(GPTConfig.from_depth(1, aspect_ratio=32, head_dim=8, sequence_len=8, vocab_size=265)).cuda()
    optimizer = model.setup_optimizer(use_bf16=True)
    tokens = torch.randint(0, 256, (1, 8), device="cuda")
    with autocast_context(torch.device("cuda"), torch.bfloat16):
        loss = model(tokens, tokens)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(p.dtype == torch.float32 for p in model.parameters())
