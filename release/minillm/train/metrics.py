from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable


def load_jsonl_metrics(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_metrics_csv(rows: Iterable[dict], path: str | Path) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def perplexity(loss: float | None, max_value: float = 1e9) -> float | None:
    if loss is None:
        return None
    if loss > math.log(max_value):
        return max_value
    return float(math.exp(loss))


def last_metric(rows: list[dict], key: str):
    for row in reversed(rows):
        value = row.get(key)
        if value is not None:
            return value
    return None


def best_metric(rows: list[dict], key: str, mode: str = "min") -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    if mode == "min":
        return min(values)
    if mode == "max":
        return max(values)
    raise ValueError("mode must be 'min' or 'max'")


def smooth_values(values: list[float], window: int = 5) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        span = values[start : i + 1]
        out.append(sum(span) / len(span))
    return out


def summarize_metrics(rows_or_path: list[dict] | str | Path) -> dict:
    rows = load_jsonl_metrics(rows_or_path) if not isinstance(rows_or_path, list) else rows_or_path
    train_loss = last_metric(rows, "train_loss")
    valid_loss = last_metric(rows, "valid_loss")
    best_valid = best_metric(rows, "valid_loss", mode="min")
    tokens_seen = last_metric(rows, "tokens_seen")
    if tokens_seen is None:
        tokens_seen = last_metric(rows, "tokens")
    return {
        "steps": int(last_metric(rows, "step") or 0),
        "rows": len(rows),
        "last_train_loss": train_loss,
        "last_valid_loss": valid_loss,
        "best_valid_loss": best_valid,
        "last_train_ppl": perplexity(float(train_loss)) if train_loss is not None else None,
        "last_valid_ppl": perplexity(float(valid_loss)) if valid_loss is not None else None,
        "last_lr": last_metric(rows, "lr"),
        "tokens_seen": int(tokens_seen or 0),
    }
