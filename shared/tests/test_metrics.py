from __future__ import annotations

import csv
import json

from adapters import load_jsonl_metrics, summarize_metrics, write_metrics_csv


def test_load_summarize_and_export_metrics(tmp_path):
    rows = [
        {"step": 1, "train_loss": 3.0, "valid_loss": 3.2, "lr": 0.001, "tokens_seen": 32},
        {"step": 2, "train_loss": 2.5, "valid_loss": 2.8, "lr": 0.0005, "tokens_seen": 64},
        {"step": 3, "train_loss": 2.0, "lr": 0.0001, "tokens_seen": 96},
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loaded = load_jsonl_metrics(path)
    assert loaded == rows
    summary = summarize_metrics(loaded)
    assert summary["steps"] == 3
    assert summary["rows"] == 3
    assert summary["last_train_loss"] == 2.0
    assert summary["last_valid_loss"] == 2.8
    assert summary["best_valid_loss"] == 2.8
    assert summary["tokens_seen"] == 96
    assert summary["last_train_ppl"] > 1.0

    csv_path = tmp_path / "metrics.csv"
    write_metrics_csv(loaded, csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        exported = list(csv.DictReader(f))
    assert exported[-1]["step"] == "3"


def test_summarize_metrics_handles_empty_rows():
    summary = summarize_metrics([])
    assert summary["steps"] == 0
    assert summary["rows"] == 0
    assert summary["last_train_loss"] is None
    assert summary["last_valid_ppl"] is None
    assert summary["tokens_seen"] == 0
