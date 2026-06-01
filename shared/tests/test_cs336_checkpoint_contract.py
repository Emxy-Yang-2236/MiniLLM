from __future__ import annotations

import torch

from adapters import build_model, get_adamw_cls, make_config, run_load_checkpoint, run_save_checkpoint


def _one_step(model, optimizer, x, y):
    optimizer.zero_grad()
    loss = model(x, y)["loss"]
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_cs336_checkpoint_restores_model_optimizer_iteration_and_resume(tmp_path):
    torch.manual_seed(0)
    cfg = make_config(vocab_size=32, context_length=6, d_model=16, d_ff=32, num_heads=4, num_layers=1)
    model = build_model(cfg)
    AdamW = get_adamw_cls()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))

    _one_step(model, optimizer, x, y)
    path = tmp_path / "checkpoint.pt"
    run_save_checkpoint(model, optimizer, iteration=1, out=path)

    _one_step(model, optimizer, x, y)
    continued_params = [param.detach().clone() for param in model.parameters()]

    restored = build_model(cfg)
    restored_optimizer = AdamW(restored.parameters(), lr=1e-3, weight_decay=0.01)
    iteration = run_load_checkpoint(path, restored, restored_optimizer)
    assert iteration == 1

    for left, right in zip(restored.parameters(), build_model(cfg).parameters()):
        assert not torch.allclose(left, right)
    assert restored_optimizer.state

    _one_step(restored, restored_optimizer, x, y)
    for expected, actual in zip(continued_params, restored.parameters()):
        assert torch.allclose(actual, expected, atol=1e-6)
