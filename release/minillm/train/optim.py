from __future__ import annotations

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            }

        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:

            lr = group["lr"]  # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state.get("step", 0)  # Get iteration number from the state, or 0.

                grad = p.grad.data  # Get the gradient of loss with respect to p.
                if t != 0:
                    adjusted_lr = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)
                else:
                    adjusted_lr = lr

                # apply weight decay
                p.data -= lr * weight_decay * p.data
                # update first moment estimate
                m = state["exp_avg"]
                state["exp_avg"] = beta1 * m + (1 - beta1) * grad
                # update second moment estimate
                v = state["exp_avg_sq"]
                state["exp_avg_sq"] = beta2 * v + (1 - beta2) * (grad ** 2)
                # update weight
                p.data -= adjusted_lr * (state["exp_avg"] / (state["exp_avg_sq"] ** 0.5 + eps))

        return loss

def clip_grad_norm_(parameters, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    # parameters could be an iterable obj that can only be iterated once
    grads: list[torch.Tensor] = []
    for p in parameters:
        if p.grad is None:
            continue
        grads.append(p.grad)

    if len(grads) == 0:
        return torch.tensor(0.0)

    total_norm = grads[0].new_zeros(())

    # grad is reference
    for grad in grads:
        l_2_norm = grad.pow(2).sum()
        total_norm += l_2_norm
    total_norm = torch.sqrt(total_norm)

    if total_norm >= max_norm:
        for grad in grads:
            grad.mul_(max_norm / (total_norm + eps))

    return total_norm
