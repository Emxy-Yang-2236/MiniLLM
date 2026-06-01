#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.model.config import TransformerConfig
from minillm.systems.benchmark import benchmark_sweep, write_csv, write_markdown_summary
from minillm.utils.config import load_yaml


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def _strs(text: str) -> list[str]:
    return [x for x in text.split(",") if x]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seq_lens", default="64,128")
    parser.add_argument("--batch_sizes", default="1,2")
    parser.add_argument("--attentions", default="naive,sdpa")
    parser.add_argument("--precisions", default="fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--csv", default="reports/benchmarks/sweep.csv")
    parser.add_argument("--md", default="reports/benchmarks/sweep.md")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    model_cfg = TransformerConfig.from_dict(cfg["model"] | {"vocab_size": cfg.get("vocab_size", 512)})
    rows = benchmark_sweep(
        model_cfg,
        seq_lens=_ints(args.seq_lens),
        batch_sizes=_ints(args.batch_sizes),
        attentions=_strs(args.attentions),
        precisions=_strs(args.precisions),
        device=args.device,
        warmup=args.warmup,
        steps=args.steps,
    )
    write_csv(rows, args.csv)
    write_markdown_summary(rows, args.md)
    print({"rows": len(rows), "csv": args.csv, "md": args.md})


if __name__ == "__main__":
    main()
