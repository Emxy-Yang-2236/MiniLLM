from __future__ import annotations

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        raise NotImplementedError("Week 2 TODO: implement AdamW using torch.optim.Optimizer base class")

    def step(self, closure=None):
        raise NotImplementedError


def clip_grad_norm_(parameters, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    raise NotImplementedError("Week 2 TODO: implement global gradient clipping")
