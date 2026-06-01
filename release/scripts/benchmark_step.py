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
from minillm.systems.benchmark import benchmark_step
from minillm.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--attention", choices=["naive", "sdpa"], default="sdpa")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--memory_snapshot", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    model_data = dict(cfg["model"])
    if args.seq_len is not None:
        model_data["context_length"] = args.seq_len
    model_cfg = TransformerConfig.from_dict(model_data | {"vocab_size": cfg.get("vocab_size", model_data.get("vocab_size", 512))})
    result = benchmark_step(
        model_cfg,
        attention=args.attention,
        device=args.device,
        warmup=args.warmup,
        steps=args.steps,
        batch_size=args.batch_size or cfg.get("batch_size", 2),
        precision=args.precision,
        compile_model=args.compile,
        memory_snapshot=args.memory_snapshot,
    )
    text = json.dumps(result, indent=2)
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
