from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .schedules import WarmupScheduler
    from .state import TrainState

import torch


def save_checkpoint(
    path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    config: dict | None = None,
    tokenizer_path: str | None = None,
    scheduler: WarmupScheduler | None =None,
    train_state: TrainState | None =None,
    tokenizer_metadata: dict | None = None,
):

    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state" : optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state" : scheduler.state_dict() if scheduler is not None else None,
        "step" : step,
        "config" : config if config is not None else {},
        "tokenizer_path": tokenizer_path,
        "train_state" : train_state.state_dict() if train_state is not None else None,
        "tokenizer_metadata": tokenizer_metadata if tokenizer_metadata is not None else {},
    }

    torch.save(payload, path)


def load_checkpoint(
        path,
        model : torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: WarmupScheduler | None =None,
        train_state: TrainState | None =None,
        map_location="cpu"):

    payload = torch.load(path, map_location= map_location)

    if model is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload["optimizer_state"] is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload["scheduler_state"] is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if train_state is not None and payload["train_state"] is not None:
        train_state.load_state_dict(payload["train_state"])

    return payload
