"""Single-device Muon/AdamW adapted from nanochat 92d63d4; see third_party/."""
from __future__ import annotations
from typing import Callable, overload
import torch
from torch import Tensor

POLAR_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def polar_express(matrix: Tensor, ns_steps: int = 5) -> Tensor:
    """Approximate the polar factor of (..., rows, columns), including zero."""
    if matrix.ndim < 2 or not 1 <= ns_steps <= len(POLAR_COEFFICIENTS):
        raise ValueError("polar_express requires a matrix and 1..5 iterations")
    x = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in POLAR_COEFFICIENTS[:ns_steps]:
        if x.shape[-2] > x.shape[-1]:
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
        else:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x


def nesterov_momentum(grad: Tensor, previous: Tensor, momentum: float) -> tuple[Tensor, Tensor]:
    updated = torch.lerp(previous, grad, 1 - momentum)
    return torch.lerp(grad, updated, momentum), updated


def variance_normalize(update: Tensor, previous: Tensor, beta2: float, red_dim: int) -> tuple[Tensor, Tensor]:
    variance = update.float().square().mean(dim=red_dim, keepdim=True)
    updated = torch.lerp(previous, variance, 1 - beta2)
    inverse = updated.clamp_min(1e-10).rsqrt()
    norm = (variance.sum(dim=(-2, -1), keepdim=True) * update.shape[red_dim]).sqrt()
    scaled_norm = (variance * update.shape[red_dim] * inverse.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    return update * inverse * (norm / scaled_norm.clamp_min(1e-10)), updated


def muon_update(grad: Tensor, momentum_buffer: Tensor, second_buffer: Tensor, *, momentum: float = .95,
                beta2: float = .9, ns_steps: int = 5, use_bf16: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    direction, first = nesterov_momentum(grad.float(), momentum_buffer, momentum)
    x = direction.bfloat16() if use_bf16 else direction
    target = x.float().norm(dim=(-2, -1), keepdim=True) / x.shape[-2] ** .5
    row_norm = x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = x * (target / row_norm).to(x.dtype)
    x = polar_express(x, ns_steps).float()
    x = x * (min(x.shape[-2:]) ** .5 / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6))
    red_dim = -1 if x.shape[-2] >= x.shape[-1] else -2
    update, second = variance_normalize(x, second_buffer, beta2, red_dim)
    return update, first, second


def adamw_update(param: Tensor, grad: Tensor, first: Tensor, second: Tensor, *, step: int,
                 lr: float, betas: tuple[float, float], eps: float, weight_decay: float) -> tuple[Tensor, Tensor, Tensor]:
    beta1, beta2 = betas
    first = torch.lerp(first, grad.float(), 1 - beta1)
    second = torch.lerp(second, grad.float().square(), 1 - beta2)
    denom = (second / (1 - beta2 ** step)).sqrt() + eps
    result = param.float() * (1 - lr * weight_decay) - (lr / (1 - beta1 ** step)) * first / denom
    return result, first, second


class MuonAdamW(torch.optim.Optimizer):
    """FP32 parameters/states, optionally BF16 polar computation; no cached groups."""
    def __init__(self, param_groups: list[dict], use_bf16: bool = False):
        for group in param_groups:
            if group.get("kind") not in {"adamw", "muon"}:
                raise ValueError("parameter groups require kind adamw or muon")
        super().__init__(param_groups, defaults={})
        self.use_bf16 = use_bf16
        for group in self.param_groups:
            group.setdefault("initial_lr", group["lr"])

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @overload
    def step(self, closure: Callable[[], Tensor]) -> Tensor: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float | Tensor] | None = None) -> float | Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.dtype != torch.float32 or param.grad.is_sparse:
                    raise ValueError("MuonAdamW requires dense gradients and FP32 parameters")
                state = self.state[param]
                if group["kind"] == "adamw":
                    if not state:
                        state.update(step=0, exp_avg=torch.zeros_like(param), exp_avg_sq=torch.zeros_like(param))
                    state["step"] += 1
                    updated, first, second = adamw_update(
                        param, param.grad, state["exp_avg"], state["exp_avg_sq"], step=state["step"],
                        lr=group["lr"], betas=group["betas"], eps=group["eps"], weight_decay=group["weight_decay"])
                    param.copy_(updated)
                    state["exp_avg"].copy_(first)
                    state["exp_avg_sq"].copy_(second)
                else:
                    if param.ndim != 2:
                        raise ValueError("Muon parameters must be 2D")
                    if not state:
                        shape = (param.shape[0], 1) if param.shape[0] >= param.shape[1] else (1, param.shape[1])
                        state.update(momentum_buffer=torch.zeros_like(param),
                                     second_momentum_buffer=torch.zeros(shape, device=param.device))
                    update, first, second = muon_update(
                        param.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                        momentum=group["momentum"], beta2=group["beta2"], ns_steps=group["ns_steps"],
                        use_bf16=self.use_bf16)
                    lr = group["lr"] * max(1., param.shape[0] / param.shape[1]) ** .5
                    param.sub_(lr * update + lr * group["weight_decay"] * param * ((update * param) >= 0))
                    state["momentum_buffer"].copy_(first)
                    state["second_momentum_buffer"].copy_(second)
        return loss
