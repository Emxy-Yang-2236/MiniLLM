from __future__ import annotations

import math

import numpy as np
import torch

from adapters import (
    get_adamw_cls,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
)


def test_cs336_get_batch_returns_shifted_x_y():
    torch.manual_seed(0)
    dataset = np.arange(20, dtype=np.uint32)
    x, y = run_get_batch(dataset, batch_size=4, context_length=5, device="cpu")
    assert x.shape == (4, 5)
    assert y.shape == (4, 5)
    assert torch.equal(y, x + 1)


def test_cs336_adamw_updates_and_decouples_weight_decay():
    AdamW = get_adamw_cls()
    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([0.0])
    opt = AdamW([param], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    opt.step()
    assert torch.allclose(param.detach(), torch.tensor([0.99]), atol=1e-6)

    param2 = torch.nn.Parameter(torch.tensor([1.0]))
    param2.grad = torch.tensor([0.5])
    opt2 = AdamW([param2], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    opt2.step()
    assert param2.item() < 1.0
    assert opt2.state[param2]["step"] == 1


def test_cs336_gradient_clipping_caps_global_norm():
    p1 = torch.nn.Parameter(torch.tensor([0.0]))
    p2 = torch.nn.Parameter(torch.tensor([0.0]))
    p1.grad = torch.tensor([3.0])
    p2.grad = torch.tensor([4.0])
    original = run_gradient_clipping([p1, p2], max_l2_norm=1.0)
    clipped = torch.sqrt(p1.grad.pow(2).sum() + p2.grad.pow(2).sum())
    assert torch.allclose(original, torch.tensor(5.0), atol=1e-5)
    assert clipped <= 1.0 + 1e-5


def test_cs336_cosine_schedule_contract_points():
    lr0 = run_get_lr_cosine_schedule(0, 1.0, 0.1, warmup_iters=2, cosine_cycle_iters=6)
    lr1 = run_get_lr_cosine_schedule(1, 1.0, 0.1, warmup_iters=2, cosine_cycle_iters=6)
    lr2 = run_get_lr_cosine_schedule(2, 1.0, 0.1, warmup_iters=2, cosine_cycle_iters=6)
    lr4 = run_get_lr_cosine_schedule(4, 1.0, 0.1, warmup_iters=2, cosine_cycle_iters=6)
    lr7 = run_get_lr_cosine_schedule(7, 1.0, 0.1, warmup_iters=2, cosine_cycle_iters=6)
    assert lr0 == 0.0
    assert lr1 == 0.5
    assert lr2 == 1.0
    assert math.isclose(lr4, 0.55, rel_tol=1e-6)
    assert lr7 == 0.1
