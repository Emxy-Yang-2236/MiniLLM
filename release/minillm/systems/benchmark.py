"""Provided training measurement utilities for the Week 4 mini-lab.

Students run these helpers and interpret the CSV/markdown outputs. They are
not expected to implement systems optimizations or compete on speed.
"""

from __future__ import annotations

import contextlib
import csv
import math
import statistics
import time
from pathlib import Path

import torch

from minillm.model.config import TransformerConfig
from minillm.model.transformer import TransformerLM
from minillm.train.optim import AdamW


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "mps" and not _mps_available():
        raise RuntimeError("MPS was requested, but this PyTorch install cannot use MPS.")
    return torch.device(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "cpu" and precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def precision_supported(device: torch.device, precision: str) -> bool:
    if precision == "fp32":
        return True
    if precision == "bf16":
        return device.type == "cpu" or (device.type == "cuda" and torch.cuda.is_bf16_supported())
    if precision == "fp16":
        return device.type == "cuda"
    return False


def available_precisions(device: str = "auto") -> list[str]:
    dev = _resolve_device(device)
    candidates = ["fp32", "bf16", "fp16"]
    return [precision for precision in candidates if precision_supported(dev, precision)]


def _summary(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "mean_sec": statistics.fmean(times),
        "median_sec": statistics.median(times),
        "p90_sec": ordered[p90_index],
        "std_sec": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_sec": min(times),
        "max_sec": max(times),
    }


def check_attention_backend_correctness(
    cfg: TransformerConfig,
    batch_size: int = 2,
    seq_len: int | None = None,
    device: str = "auto",
    atol: float = 1e-4,
    rtol: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Compare naive attention and SDPA on identical weights and inputs."""
    dev = _resolve_device(device)
    context_length = seq_len or cfg.context_length
    base = {**cfg.to_dict(), "context_length": context_length}
    naive_cfg = TransformerConfig.from_dict({**base, "attention_backend": "naive"})
    sdpa_cfg = TransformerConfig.from_dict({**base, "attention_backend": "sdpa"})

    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    naive_model = TransformerLM(naive_cfg).to(dev).eval()
    sdpa_model = TransformerLM(sdpa_cfg).to(dev).eval()
    sdpa_model.load_state_dict(naive_model.state_dict())

    torch.manual_seed(seed + 1)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1)
    x = torch.randint(0, cfg.vocab_size, (batch_size, context_length), device=dev)
    y = torch.randint(0, cfg.vocab_size, (batch_size, context_length), device=dev)

    with torch.no_grad():
        naive_out = naive_model(x, y)
        sdpa_out = sdpa_model(x, y)
    _sync(dev)

    naive_loss = float(naive_out["loss"].detach().cpu())
    sdpa_loss = float(sdpa_out["loss"].detach().cpu())
    abs_loss_diff = abs(naive_loss - sdpa_loss)
    logits_diff = (naive_out["logits"] - sdpa_out["logits"]).abs()
    max_abs_diff_logits = float(logits_diff.max().detach().cpu())
    max_ref_logit = float(naive_out["logits"].abs().max().detach().cpu())
    loss_ok = abs_loss_diff <= atol + rtol * abs(naive_loss)
    logits_ok = max_abs_diff_logits <= atol + rtol * max_ref_logit
    return {
        "device": str(dev),
        "batch_size": batch_size,
        "context_length": context_length,
        "loss_naive": naive_loss,
        "loss_sdpa": sdpa_loss,
        "abs_loss_diff": abs_loss_diff,
        "max_abs_diff_logits": max_abs_diff_logits,
        "atol": atol,
        "rtol": rtol,
        "passed": bool(loss_ok and logits_ok),
    }


def benchmark_step(
    cfg: TransformerConfig,
    attention: str = "sdpa",
    device: str = "auto",
    warmup: int = 2,
    steps: int = 5,
    batch_size: int = 2,
    precision: str = "fp32",
    compile_model: bool = False,
    memory_snapshot: str | None = None,
):
    dev = _resolve_device(device)
    if not precision_supported(dev, precision):
        raise RuntimeError(f"precision {precision} is not supported on {dev.type}")
    cfg = TransformerConfig.from_dict({**cfg.to_dict(), "attention_backend": attention})
    model = TransformerLM(cfg).to(dev)
    if compile_model:
        model = torch.compile(model)
    opt = AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.context_length), device=dev)
    y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.context_length), device=dev)

    def forward_only():
        with _autocast(dev, precision):
            return model(x, y)["loss"]

    def forward_backward():
        opt.zero_grad(set_to_none=True)
        loss = forward_only()
        loss.backward()
        return loss

    def full_step():
        loss = forward_backward()
        opt.step()
        return loss

    for _ in range(warmup):
        full_step()
    _sync(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        if memory_snapshot:
            torch.cuda.memory._record_memory_history(max_entries=100000)

    timings: dict[str, list[float]] = {"forward": [], "forward_backward": [], "optimizer_step": [], "full_step": []}
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with torch.no_grad():
            forward_only()
        _sync(dev)
        timings["forward"].append(time.perf_counter() - start)

        start = time.perf_counter()
        forward_backward()
        _sync(dev)
        timings["forward_backward"].append(time.perf_counter() - start)

        opt.zero_grad(set_to_none=True)
        forward_backward()
        start = time.perf_counter()
        opt.step()
        _sync(dev)
        timings["optimizer_step"].append(time.perf_counter() - start)

        start = time.perf_counter()
        full_step()
        _sync(dev)
        timings["full_step"].append(time.perf_counter() - start)

    if dev.type == "cuda" and memory_snapshot:
        path = Path(memory_snapshot)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(path))
        torch.cuda.memory._record_memory_history(enabled=None)
    peak = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
    tokens = batch_size * cfg.context_length
    full_mean = _summary(timings["full_step"])["mean_sec"]
    return {
        "device": str(dev),
        "attention": attention,
        "precision": precision,
        "compile": bool(compile_model),
        "batch_size": batch_size,
        "context_length": cfg.context_length,
        "d_model": cfg.d_model,
        "d_ff": cfg.d_ff,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "warmup": warmup,
        "steps": steps,
        "forward": _summary(timings["forward"]),
        "forward_backward": _summary(timings["forward_backward"]),
        "optimizer_step": _summary(timings["optimizer_step"]),
        "full_step": _summary(timings["full_step"]),
        "tokens_per_sec": tokens / full_mean,
        "peak_memory_bytes": peak if peak is not None else "unavailable",
    }


def benchmark_sweep(
    base_cfg: TransformerConfig,
    seq_lens: list[int],
    batch_sizes: list[int],
    attentions: list[str],
    precisions: list[str],
    device: str = "auto",
    warmup: int = 2,
    steps: int = 5,
) -> list[dict]:
    rows: list[dict] = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            for attention in attentions:
                for precision in precisions:
                    cfg = TransformerConfig.from_dict({**base_cfg.to_dict(), "context_length": seq_len})
                    dev = _resolve_device(device)
                    if not precision_supported(dev, precision):
                        rows.append(
                            flatten_benchmark_row(
                                {
                                    "status": "skipped",
                                    "error": f"precision {precision} is not supported on {dev.type}",
                                    "device": str(dev),
                                    "attention": attention,
                                    "precision": precision,
                                    "batch_size": batch_size,
                                    "context_length": seq_len,
                                    "d_model": cfg.d_model,
                                    "num_layers": cfg.num_layers,
                                    "num_heads": cfg.num_heads,
                                }
                            )
                        )
                        continue
                    try:
                        result = benchmark_step(
                            cfg,
                            attention=attention,
                            device=device,
                            warmup=warmup,
                            steps=steps,
                            batch_size=batch_size,
                            precision=precision,
                        )
                        result["status"] = "ok"
                    except RuntimeError as exc:
                        result = {
                            "status": "error",
                            "error": str(exc).splitlines()[0],
                            "attention": attention,
                            "precision": precision,
                            "batch_size": batch_size,
                            "context_length": seq_len,
                        }
                    rows.append(flatten_benchmark_row(result))
    return rows


def flatten_benchmark_row(result: dict) -> dict:
    row = {
        "status": result.get("status", "ok"),
        "device": result.get("device"),
        "cuda_available": torch.cuda.is_available(),
        "attention": result.get("attention"),
        "backend": result.get("attention"),
        "precision": result.get("precision"),
        "dtype": result.get("precision"),
        "compile": result.get("compile", False),
        "batch_size": result.get("batch_size"),
        "context_length": result.get("context_length"),
        "seq_len": result.get("context_length"),
        "d_model": result.get("d_model"),
        "num_layers": result.get("num_layers"),
        "num_heads": result.get("num_heads"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "peak_memory_bytes": result.get("peak_memory_bytes"),
        "error": result.get("error", ""),
    }
    for prefix in ["forward", "forward_backward", "optimizer_step", "full_step"]:
        stats = result.get(prefix, {})
        for key in ["mean_sec", "median_sec", "p90_sec", "std_sec", "min_sec", "max_sec"]:
            row[f"{prefix}_{key}"] = stats.get(key)
    row["forward_sec"] = row.get("forward_mean_sec")
    row["fwd_bwd_sec"] = row.get("forward_backward_mean_sec")
    row["optimizer_sec"] = row.get("optimizer_step_mean_sec")
    row["full_step_sec"] = row.get("full_step_mean_sec")
    return row


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Measurement Sanity Benchmark",
        "",
        "These small-scale results are for teaching and comparison. This is not a leaderboard or an optimization task. SDPA may be faster or use less memory; bf16/fp16 results can be noisy and may change numerical behavior.",
        "",
        "| status | attention | precision | batch | seq | full_step_mean_sec | full_step_median_sec | full_step_p90_sec | tokens/sec | peak_memory |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {status} | {attention} | {precision} | {batch_size} | {context_length} | {full_step_mean_sec} | {full_step_median_sec} | {full_step_p90_sec} | {tokens_per_sec} | {peak_memory_bytes} |".format(
                **{k: row.get(k, "") for k in [
                    "status",
                    "attention",
                    "precision",
                    "batch_size",
                    "context_length",
                    "full_step_mean_sec",
                    "full_step_median_sec",
                    "full_step_p90_sec",
                    "tokens_per_sec",
                    "peak_memory_bytes",
                ]}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
