from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def load_benchmark_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_float(value) -> float | None:
    if value in (None, "", "unavailable"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_rows(rows: list[dict], metric: str = "tokens_per_sec", higher_is_better: bool = True) -> list[dict]:
    ok_rows = [row for row in rows if row.get("status", "ok") == "ok" and _as_float(row.get(metric)) is not None]
    if not ok_rows:
        return []
    key = lambda row: _as_float(row.get(metric)) or 0.0
    best = max(ok_rows, key=key) if higher_is_better else min(ok_rows, key=key)
    best_value = key(best)
    return [row for row in ok_rows if key(row) == best_value]


def compare_attention_backends(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = (
            row.get("precision"),
            row.get("batch_size"),
            row.get("context_length"),
            row.get("d_model"),
            row.get("num_layers"),
            row.get("num_heads"),
        )
        groups[key][str(row.get("attention"))] = row
    comparisons: list[dict] = []
    for key, by_attention in sorted(groups.items()):
        naive = by_attention.get("naive")
        sdpa = by_attention.get("sdpa")
        if naive is None or sdpa is None:
            continue
        naive_tps = _as_float(naive.get("tokens_per_sec"))
        sdpa_tps = _as_float(sdpa.get("tokens_per_sec"))
        if naive_tps is None or sdpa_tps is None or naive_tps == 0:
            continue
        precision, batch_size, context_length, d_model, num_layers, num_heads = key
        comparisons.append(
            {
                "precision": precision,
                "batch_size": batch_size,
                "context_length": context_length,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "naive_tokens_per_sec": naive_tps,
                "sdpa_tokens_per_sec": sdpa_tps,
                "sdpa_over_naive": sdpa_tps / naive_tps,
            }
        )
    return comparisons


def benchmark_report(rows: list[dict]) -> dict:
    best = best_rows(rows)
    comparisons = compare_attention_backends(rows)
    ok = sum(1 for row in rows if row.get("status", "ok") == "ok")
    skipped = sum(1 for row in rows if row.get("status") == "skipped")
    errors = sum(1 for row in rows if row.get("status") == "error")
    return {
        "rows": len(rows),
        "ok": ok,
        "skipped": skipped,
        "errors": errors,
        "best": best,
        "attention_comparisons": comparisons,
    }
