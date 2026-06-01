from __future__ import annotations

import torch

from adapters import build_model, build_optimizer, load_checkpoint, make_config, make_scheduler, save_checkpoint, train_state


def test_checkpoint_roundtrip(tmp_path):
    cfg = make_config(vocab_size=64)
    model = build_model(cfg)
    opt = build_optimizer(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, step=3, config=cfg.to_dict(), tokenizer_path="tok.json")
    payload = load_checkpoint(path, map_location="cpu")
    assert payload["step"] == 3
    assert payload["config"]["vocab_size"] == 64
    assert payload["tokenizer_path"] == "tok.json"


def test_checkpoint_restores_model_optimizer_scheduler_and_train_state(tmp_path):
    cfg = make_config(vocab_size=64, context_length=8)
    model = build_model(cfg)
    opt = build_optimizer(model.parameters(), lr=1e-3)
    sched = make_scheduler(max_lr=1e-3, max_steps=10, warmup_steps=1)
    state = train_state(step=5, tokens_seen=128, best_valid_loss=1.25)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
    y = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
    loss = model(x, y)["loss"]
    loss.backward()
    opt.step()
    sched.step()
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, opt, step=state.step, config=cfg.to_dict(), scheduler=sched, train_state=state)

    restored = build_model(cfg)
    restored_opt = build_optimizer(restored.parameters(), lr=1e-3)
    restored_sched = make_scheduler()
    restored_state = train_state()
    payload = load_checkpoint(path, restored, restored_opt, restored_sched, restored_state)
    assert payload["step"] == 5
    assert restored_state.step == 5
    assert restored_state.tokens_seen == 128
    assert restored_sched.last_step == sched.last_step
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(left, right)
