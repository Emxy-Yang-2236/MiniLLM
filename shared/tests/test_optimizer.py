from __future__ import annotations

import torch

from adapters import build_optimizer, clip_grad_norm, make_scheduler


def test_adamw_updates_parameter_and_state():
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    opt = build_optimizer([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.01)
    p.grad = torch.tensor([0.5, -0.25])
    before = p.detach().clone()
    opt.step()
    assert not torch.allclose(p.detach(), before)
    state = opt.state[p]
    assert state["step"] == 1
    assert "exp_avg" in state and "exp_avg_sq" in state


def test_clip_grad_norm_scales_large_gradients():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p.grad = torch.tensor([3.0, 4.0])
    total = clip_grad_norm([p], max_norm=1.0)
    assert torch.allclose(total, torch.tensor(5.0))
    assert torch.linalg.norm(p.grad) <= 1.0001


def test_warmup_scheduler_state_roundtrip():
    sched = make_scheduler(max_lr=1.0, max_steps=10, warmup_steps=2, min_lr=0.1, kind="cosine")
    lrs = [sched.step(), sched.step(), sched.step()]
    assert lrs[0] < lrs[1]
    state = sched.state_dict()
    restored = make_scheduler()
    restored.load_state_dict(state)
    assert restored.last_step == sched.last_step
    assert restored.get_lr() == sched.get_lr()
