from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from minillm.model.config import TransformerConfig
from minillm.model.transformer import TransformerLM
from minillm.train.checkpoint import load_checkpoint
from minillm.train.optim import AdamW
from minillm.train.schedules import WarmupScheduler


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    best_valid_loss: float | None = None
    start_time: float = field(default_factory=time.perf_counter)

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, data: dict) -> None:
        self.step = int(data.get("step", 0))
        self.tokens_seen = int(data.get("tokens_seen", 0))
        self.best_valid_loss = data.get("best_valid_loss")
        self.start_time = time.perf_counter()


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def create_optimizer(model, cfg: dict):
    return AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
        eps=cfg.get("eps", 1e-8),
        weight_decay=cfg.get("weight_decay", 0.01),
    )


def create_scheduler(cfg: dict, max_steps: int):
    return WarmupScheduler(
        max_lr=cfg["learning_rate"],
        max_steps=max_steps,
        warmup_steps=cfg.get("warmup_steps", 10),
        min_lr=cfg.get("min_learning_rate", 0.0),
        kind=cfg.get("schedule", "cosine"),
    )


def apply_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def autocast_context(device, enabled: bool = False, dtype: str = "bf16"):
    device = torch.device(device)
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


def load_model_optimizer_scheduler(ckpt_path, map_location="cpu"):
    payload = load_checkpoint(ckpt_path, map_location=map_location)
    model = TransformerLM(TransformerConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state"])
    optimizer = AdamW(model.parameters())
    if payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    scheduler = WarmupScheduler.from_state_dict(payload.get("scheduler_state", {}))
    state = TrainState()
    state.load_state_dict(payload.get("train_state", {"step": payload.get("step", 0)}))
    return model, optimizer, scheduler, state, payload
