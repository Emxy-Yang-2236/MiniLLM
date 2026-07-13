from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def cosine_warmup(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    # diff stage
    if step < warmup_steps :
        return max_lr * (step / warmup_steps)
    elif step > max_steps :
        return min_lr
    else:
        theta = (step - warmup_steps) * math.pi / (max_steps - warmup_steps)
        cos_anneal_lr = min_lr + 0.5 * (1 + math.cos(theta)) * (max_lr - min_lr)
        return cos_anneal_lr


@dataclass
class WarmupScheduler:
    max_lr: float = 1e-3
    max_steps: int = 1000
    warmup_steps: int = 10
    min_lr: float = 0.0
    kind: str = "cosine"
    last_step: int = 0

    def get_lr(self, step: int | None = None) -> float:
        if self.kind != "cosine":
            raise ValueError(f"unsupported scheduler kind: {self.kind}")
        step = self.last_step if step is None else step
        return cosine_warmup(step, self.max_steps, self.warmup_steps, self.max_lr, self.min_lr)

    def step(self) -> float:
        lr = self.get_lr(self.last_step)
        self.last_step += 1
        return lr

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, data: dict) -> None:
        for key, value in (data or {}).items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_state_dict(cls, data: dict) -> "WarmupScheduler":
        obj = cls()
        obj.load_state_dict(data or {})
        return obj
