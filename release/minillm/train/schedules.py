from __future__ import annotations


def linear_warmup_decay(step: int, max_steps: int, warmup_steps: int, max_lr: float) -> float:
    raise NotImplementedError("Week 2 TODO: implement linear warmup then decay")


def cosine_warmup(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    raise NotImplementedError("Week 2 TODO: implement cosine schedule with warmup")


class WarmupScheduler:
    def __init__(
        self,
        max_lr: float = 1e-3,
        max_steps: int = 1000,
        warmup_steps: int = 10,
        min_lr: float = 0.0,
        kind: str = "cosine",
        last_step: int = 0,
    ):
        raise NotImplementedError("Week 2 TODO: implement scheduler state and stepping")

    def get_lr(self, step: int | None = None) -> float:
        raise NotImplementedError

    def step(self) -> float:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, data: dict) -> None:
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, data: dict) -> "WarmupScheduler":
        raise NotImplementedError
