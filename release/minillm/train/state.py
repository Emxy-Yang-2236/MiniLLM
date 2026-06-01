from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    best_valid_loss: float | None = None

    def state_dict(self) -> dict:
        raise NotImplementedError("Week 2 TODO: serialize training state")

    def load_state_dict(self, data: dict) -> None:
        raise NotImplementedError("Week 2 TODO: restore training state")


class JsonlLogger:
    def __init__(self, path: str | Path):
        raise NotImplementedError("Week 2 TODO: initialize a JSONL metrics logger")

    def log(self, row: dict) -> None:
        raise NotImplementedError("Week 2 TODO: append one metrics row as JSON")


def create_optimizer(model, cfg: dict):
    raise NotImplementedError("Week 2 TODO: create AdamW from config")


def create_scheduler(cfg: dict, max_steps: int):
    raise NotImplementedError("Week 2 TODO: create learning-rate scheduler from config")


def apply_lr(optimizer, lr: float) -> None:
    raise NotImplementedError("Week 2 TODO: set optimizer learning rate")


def autocast_context(device, enabled: bool = False, dtype: str = "bf16"):
    raise NotImplementedError("Week 2 TODO: return CUDA autocast when enabled; use a no-op context on CPU/MPS")


def load_model_optimizer_scheduler(ckpt_path, map_location="cpu"):
    raise NotImplementedError("Week 2 TODO: load model, optimizer, scheduler, and train state from checkpoint")
