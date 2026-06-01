#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.model.config import TransformerConfig
from minillm.model.transformer import TransformerLM, model_summary
from minillm.systems.analysis import benchmark_report, load_benchmark_csv
from minillm.train.checkpoint import load_checkpoint
from minillm.train.metrics import load_jsonl_metrics, summarize_metrics, write_metrics_csv


def summarize_checkpoint(path: str | Path) -> dict:
    payload = load_checkpoint(path, map_location="cpu")
    cfg = TransformerConfig.from_dict(payload["config"])
    model = TransformerLM(cfg)
    model.load_state_dict(payload["model_state"])
    return {
        "step": payload.get("step"),
        "tokenizer_path": payload.get("tokenizer_path"),
        "model": model_summary(model),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--benchmark_csv", default=None)
    parser.add_argument("--metrics_csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir is not None else None
    metrics_path = Path(args.metrics) if args.metrics is not None else (run_dir / "metrics.jsonl" if run_dir else None)
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint is not None else (run_dir / "checkpoint_last.pt" if run_dir else None)
    )

    summary: dict = {}
    if metrics_path is not None and metrics_path.exists():
        rows = load_jsonl_metrics(metrics_path)
        summary["metrics"] = summarize_metrics(rows)
        if args.metrics_csv is not None:
            write_metrics_csv(rows, args.metrics_csv)
    if checkpoint_path is not None and checkpoint_path.exists():
        summary["checkpoint"] = summarize_checkpoint(checkpoint_path)
    if args.benchmark_csv is not None:
        summary["benchmark"] = benchmark_report(load_benchmark_csv(args.benchmark_csv))

    text = json.dumps(summary, indent=2)
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
