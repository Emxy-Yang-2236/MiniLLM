#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a TinyStories plaintext file.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train_chars", type=int, default=2_000_000)
    parser.add_argument("--valid_chars", type=int, default=200_000)
    args = parser.parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"missing TinyStories source file: {source}")
    text = source.read_text(encoding="utf-8", errors="replace")
    needed = args.train_chars + args.valid_chars
    if len(text) < needed:
        raise ValueError(f"TinyStories source has {len(text)} chars; need at least {needed}")
    out = Path(args.out)
    (out / "pretrain").mkdir(parents=True, exist_ok=True)
    (out / "pretrain" / "tinystories_train.txt").write_text(text[: args.train_chars], encoding="utf-8")
    (out / "pretrain" / "tinystories_valid.txt").write_text(
        text[args.train_chars : args.train_chars + args.valid_chars], encoding="utf-8"
    )
    print({"out": str(out), "train_chars": args.train_chars, "valid_chars": args.valid_chars})


if __name__ == "__main__":
    main()
