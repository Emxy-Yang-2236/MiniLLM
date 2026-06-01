from __future__ import annotations

import json

from adapters import apply_lr, build_model, create_optimizer_from_cfg, create_scheduler_from_cfg, jsonl_logger, make_config, train_state


def test_train_state_roundtrip_resets_wallclock_start():
    state = train_state(step=7, tokens_seen=224, best_valid_loss=1.5)
    payload = state.state_dict()
    restored = train_state()
    before = restored.start_time
    restored.load_state_dict(payload)
    assert restored.step == 7
    assert restored.tokens_seen == 224
    assert restored.best_valid_loss == 1.5
    assert restored.start_time >= before


def test_jsonl_logger_appends_rows(tmp_path):
    logger = jsonl_logger(tmp_path / "metrics.jsonl")
    logger.log({"step": 1, "loss": 2.0})
    logger.log({"step": 2, "loss": 1.5})
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.5}]


def test_create_optimizer_scheduler_and_apply_lr_from_cfg():
    model = build_model(make_config(vocab_size=64, context_length=8))
    cfg = {
        "learning_rate": 0.01,
        "betas": (0.8, 0.9),
        "eps": 1e-6,
        "weight_decay": 0.0,
        "warmup_steps": 2,
        "min_learning_rate": 0.001,
        "schedule": "linear",
    }
    opt = create_optimizer_from_cfg(model, cfg)
    assert opt.param_groups[0]["lr"] == 0.01
    sched = create_scheduler_from_cfg(cfg, max_steps=10)
    assert sched.kind == "linear"
    apply_lr(opt, 0.002)
    assert all(group["lr"] == 0.002 for group in opt.param_groups)
