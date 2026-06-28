from __future__ import annotations

from pathlib import Path

from adapters import benchmark_report, compare_attention_backends, load_benchmark_csv, write_benchmark_csv


def test_benchmark_report_selects_best_and_counts_statuses():
    rows = [
        {
            "status": "ok",
            "attention": "naive",
            "precision": "fp32",
            "batch_size": "1",
            "context_length": "8",
            "d_model": "16",
            "num_layers": "1",
            "num_heads": "4",
            "tokens_per_sec": "100.0",
        },
        {
            "status": "ok",
            "attention": "sdpa",
            "precision": "fp32",
            "batch_size": "1",
            "context_length": "8",
            "d_model": "16",
            "num_layers": "1",
            "num_heads": "4",
            "tokens_per_sec": "125.0",
        },
        {"status": "skipped", "attention": "sdpa", "precision": "fp16", "tokens_per_sec": ""},
    ]
    report = benchmark_report(rows)
    assert report["rows"] == 3
    assert report["ok"] == 2
    assert report["skipped"] == 1
    assert report["best"][0]["attention"] == "sdpa"
    assert report["attention_comparisons"][0]["sdpa_over_naive"] == 1.25


def test_compare_attention_backends_requires_matched_configs():
    rows = [
        {
            "status": "ok",
            "attention": "naive",
            "precision": "fp32",
            "batch_size": "1",
            "context_length": "8",
            "tokens_per_sec": "100",
        },
        {
            "status": "ok",
            "attention": "sdpa",
            "precision": "fp32",
            "batch_size": "2",
            "context_length": "8",
            "tokens_per_sec": "200",
        },
    ]
    assert compare_attention_backends(rows) == []


def test_benchmark_csv_roundtrip_keeps_schema(tmp_path):
    rows = [
        {
            "status": "ok",
            "attention": "naive",
            "precision": "fp32",
            "batch_size": 1,
            "context_length": 8,
            "tokens_per_sec": 10.0,
            "full_step_mean_sec": 0.1,
        }
    ]
    path = tmp_path / "bench.csv"
    write_benchmark_csv(rows, path)
    loaded = load_benchmark_csv(path)
    assert loaded[0]["attention"] == "naive"
    assert loaded[0]["full_step_mean_sec"] == "0.1"


def test_training_measurement_report_template_exists():
    repo = Path(__file__).resolve().parents[2]
    assert (repo / "reports" / "templates" / "training_measurement_report.md").exists()
    assert (repo / "docs" / "Training Measurement Mini-lab.md").exists()
