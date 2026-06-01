from __future__ import annotations


def evaluate_loss(model, loader, device, max_batches: int = 10) -> float:
    raise NotImplementedError("Week 3 TODO: compute mean validation loss")


def train_pretrain(cfg: dict, max_steps: int | None = None) -> dict:
    raise NotImplementedError("Week 3 TODO: implement pretraining loop, metrics, checkpoint, and sample")
