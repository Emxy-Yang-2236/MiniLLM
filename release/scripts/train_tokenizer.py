#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.tokenizer.bpe import train_bpe_from_file
from minillm.utils.config import load_yaml


def _max_bytes(cfg: dict):
    if cfg.get("max_bytes") is not None:
        return int(cfg["max_bytes"])
    if cfg.get("tokenizer_train_mb") is not None:
        return int(float(cfg["tokenizer_train_mb"]) * 1024 * 1024)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    tok = train_bpe_from_file(
        cfg["input_path"],
        int(cfg["vocab_size"]),
        cfg["output_path"],
        special_tokens=cfg.get("special_tokens"),
        min_frequency=cfg.get("min_frequency", 2),
        max_bytes=_max_bytes(cfg),
        pretokenizer=cfg.get("pretokenizer", "gpt2_like"),
        tie_break=cfg.get("tie_break", "max"),
        num_workers=int(cfg.get("num_workers", 0) or 0),
        num_chunks=int(cfg["num_chunks"]) if cfg.get("num_chunks") is not None else None,
    )
    print({"vocab_size": tok.vocab_size, "output_path": cfg["output_path"]})


if __name__ == "__main__":
    main()
