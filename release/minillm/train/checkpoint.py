from __future__ import annotations


def save_checkpoint(
    path,
    model,
    optimizer=None,
    step: int = 0,
    config: dict | None = None,
    tokenizer_path: str | None = None,
    scheduler=None,
    train_state=None,
    tokenizer_metadata: dict | None = None,
):
    raise NotImplementedError("Week 2 TODO: save model, optimizer, step, config, and tokenizer path")


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, train_state=None, map_location="cpu"):
    raise NotImplementedError("Week 2 TODO: load checkpoint and optionally restore model/optimizer")
