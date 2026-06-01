from __future__ import annotations

from pathlib import Path
from typing import Iterable


def load_jsonl_metrics(path: str | Path) -> list[dict]:
    raise NotImplementedError("Week 2 TODO: load JSONL training metrics")


def write_metrics_csv(rows: Iterable[dict], path: str | Path) -> None:
    raise NotImplementedError("Week 2 TODO: export metrics rows to CSV")


def perplexity(loss: float | None, max_value: float = 1e9) -> float | None:
    raise NotImplementedError("Week 2 TODO: convert cross-entropy loss to perplexity")


def last_metric(rows: list[dict], key: str):
    raise NotImplementedError("Week 2 TODO: find the last non-null metric value")


def best_metric(rows: list[dict], key: str, mode: str = "min") -> float | None:
    raise NotImplementedError("Week 2 TODO: compute min/max metric from logged rows")


def smooth_values(values: list[float], window: int = 5) -> list[float]:
    raise NotImplementedError("Week 2 TODO: moving-average smoothing for reports")


def summarize_metrics(rows_or_path: list[dict] | str | Path) -> dict:
    raise NotImplementedError("Week 2 TODO: summarize losses, perplexities, lr, and tokens from metrics")
