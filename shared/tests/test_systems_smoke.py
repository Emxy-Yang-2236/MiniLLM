from __future__ import annotations

from adapters import (
    benchmark_step,
    benchmark_sweep,
    check_attention_backend_correctness,
    flatten_benchmark_row,
    make_config,
)


def test_benchmark_step_reports_expected_fields():
    cfg = make_config(vocab_size=64, context_length=8, d_model=16, d_ff=64, num_heads=4)
    result = benchmark_step(cfg, attention="naive", device="cpu", warmup=0, steps=1, batch_size=1)
    assert result["device"] == "cpu"
    assert result["attention"] == "naive"
    assert result["forward"]["mean_sec"] > 0
    assert result["forward"]["median_sec"] > 0
    assert result["forward"]["p90_sec"] > 0
    assert result["full_step"]["mean_sec"] > 0
    assert result["full_step"]["median_sec"] > 0
    assert result["full_step"]["p90_sec"] > 0
    assert result["optimizer_step"]["mean_sec"] > 0
    assert result["tokens_per_sec"] > 0
    assert "peak_memory_bytes" in result


def test_attention_backend_correctness_check_runs_on_cpu():
    cfg = make_config(vocab_size=64, context_length=8, d_model=16, d_ff=64, num_heads=4)
    result = check_attention_backend_correctness(cfg, batch_size=1, seq_len=8, device="cpu")
    assert result["device"] == "cpu"
    assert result["loss_naive"] > 0
    assert result["loss_sdpa"] > 0
    assert result["abs_loss_diff"] >= 0
    assert result["max_abs_diff_logits"] >= 0
    assert result["passed"] is True


def test_benchmark_sweep_rows_have_flat_csv_schema_on_cpu():
    cfg = make_config(vocab_size=64, context_length=8, d_model=16, d_ff=64, num_heads=4)
    rows = benchmark_sweep(
        cfg,
        seq_lens=[8],
        batch_sizes=[1],
        attentions=["naive", "sdpa"],
        precisions=["fp32"],
        device="cpu",
        warmup=0,
        steps=1,
    )
    assert len(rows) == 2
    required = {
        "status",
        "attention",
        "precision",
        "batch_size",
        "context_length",
        "forward_mean_sec",
        "forward_median_sec",
        "forward_p90_sec",
        "forward_backward_mean_sec",
        "optimizer_step_mean_sec",
        "full_step_mean_sec",
        "full_step_median_sec",
        "full_step_p90_sec",
        "tokens_per_sec",
    }
    assert required <= rows[0].keys()
    assert flatten_benchmark_row({"attention": "naive", "forward": {"mean_sec": 1.0}})["forward_mean_sec"] == 1.0
