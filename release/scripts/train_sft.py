#!/usr/bin/env python
from __future__ import annotations

import argparse

from minillm.train.sft import train_sft
from minillm.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    if args.run_dir is not None:
        cfg["run_dir"] = args.run_dir
    print(train_sft(cfg, base_ckpt=args.base_ckpt, max_steps=args.max_steps))


if __name__ == "__main__":
    main()
