import torch
from typing import Any, Literal, TypedDict, cast

from torch import Tensor, optim
from torch.nn import Parameter


class AdamWGroup(TypedDict):
    kind: Literal["adamw"]
    params: list[Parameter]
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


class MuonGroup(TypedDict):
    kind: Literal["muon"]
    params: list[Parameter]
    lr: float
    beta: float  # 一阶动量系数；
    eps: float
    ns_steps: int
    weight_decay: float


ParamGroup = AdamWGroup | MuonGroup

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class AdamWState(TypedDict):
    step: int
    exp_avg: Tensor  # 一阶矩 m（梯度的 EMA）
    exp_avg_sq: Tensor  # 二阶矩 v（梯度平方的 EMA）


class MuonState(TypedDict):
    momentum_buffer: Tensor  # 一阶动量（和 param 同形状）
    second_momentum_buffer: Tensor  # NorMuon 二阶（按行/列 reduce 后的形状）


class MuonAdamW(optim.Optimizer):
    def __init__(self, param_groups: list[ParamGroup]) -> None:
        super().__init__(cast(list[dict[str, Any]], param_groups), defaults={})
        # 保持分组：每组自己的 params / lr / betas，不要 flatten 成两个大 list
        self.adamw_groups: list[AdamWGroup] = []
        self.muon_groups: list[MuonGroup] = []
        for group in param_groups:
            if group["kind"] == "adamw":
                self.adamw_groups.append(group)
            elif group["kind"] == "muon":
                self.muon_groups.append(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

    @torch.no_grad()
    def _compute_adamW(self) -> None:
        """AdamW: EMA → bias correction → decoupled weight decay → param update."""
        for group in self.adamw_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = cast(AdamWState, self.state[param])
                if len(state) == 0:
                    # exp_* = exponential moving average（指数滑动平均），不是 exp()
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # m_t = β1 m_{t-1} + (1-β1) g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = β2 v_{t-1} + (1-β2) g_t²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias correction: m̂ = m/(1-β1^t), v̂ = v/(1-β2^t)
                m_hat = exp_avg / (1 - beta1**t)
                v_hat = exp_avg_sq / (1 - beta2**t)
                # denom = √v̂ + ε
                denom = v_hat.sqrt().add_(eps)

                # decoupled weight decay, then θ ← θ - lr * m̂ / denom
                param.mul_(1 - lr * weight_decay)
                param.addcdiv_(m_hat, denom, value=-lr)

    @torch.no_grad()
    def _compute_muon(self) -> None:
        for group in self.muon_groups:
            (beta, eps, lr, weight_decay) = (
                group["beta"],
                group["eps"],
                group["lr"],
                group["weight_decay"],
            )  # 一阶动量系数
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = cast(MuonState, self.state[param])
                n_row, n_col = param.shape[-2], param.shape[-1]
                red_dim = -1 if n_row >= n_col else -2
                v_shape = (*param.shape[:-1], 1) if n_row >= n_col else (*param.shape[:-2], 1, n_col)
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                    state["second_momentum_buffer"] = torch.zeros(v_shape, dtype=param.dtype, device=param.device)

                momentum_buffer = state["momentum_buffer"]
                # 一阶动量（Nesterov 可再按 nanochat 补全）
                momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                # 先归一化，再迭代
                X = momentum_buffer / (momentum_buffer.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
                for i in range(5):
                    a, b, c = polar_express_coeffs[i]
                    if n_row >= n_col:
                        A = X.mT @ X
                        B = b * A + c * (A @ A)
                        X = a * X + X @ B
                    else:
                        A = X @ X.mT
                        B = b * A + c * (A @ A)
                        X = a * X + B @ X

                v_mean = X.square().mean(dim=red_dim, keepdim=True)
                X = X / (v_mean.sqrt() + eps)

                param.mul_(1 - lr * weight_decay).sub_(lr * X)
