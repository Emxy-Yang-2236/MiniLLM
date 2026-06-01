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
from minillm.systems.benchmark import check_attention_backend_correctness
from minillm.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_data = dict(cfg["model"])
    if args.seq_len is not None:
        model_data["context_length"] = args.seq_len
    model_cfg = TransformerConfig.from_dict(
        model_data | {"vocab_size": cfg.get("vocab_size", model_data.get("vocab_size", 512))}
    )
    result = check_attention_backend_correctness(
        model_cfg,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        atol=args.atol,
        rtol=args.rtol,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2)
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
