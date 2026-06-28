#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.train.pretrain import train_pretrain
from minillm.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    if args.run_dir is not None:
        cfg["run_dir"] = args.run_dir
    print(train_pretrain(cfg, max_steps=args.max_steps))


if __name__ == "__main__":
    main()
