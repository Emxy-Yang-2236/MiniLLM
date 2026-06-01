#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.data.pretrain_dataset import encode_text_file_with_manifest, encoded_manifest_path
from minillm.tokenizer.bpe import ByteBPETokenizer
from minillm.utils.config import load_yaml


def _encode_one(tokenizer_path: str, input_path: str, output_path: str, max_bytes=None, add_eos: bool = True) -> dict:
    tok = ByteBPETokenizer.load(tokenizer_path)
    return encode_text_file_with_manifest(
        input_path,
        tok,
        output_path,
        tokenizer_path=tokenizer_path,
        max_bytes=max_bytes,
        add_eos=add_eos,
    )


def _run_config(path: str) -> list[dict]:
    cfg = load_yaml(path)
    rows = []
    for item in cfg["datasets"]:
        manifest = _encode_one(
            cfg["tokenizer_path"],
            item["input_path"],
            item["output_path"],
            max_bytes=item.get("max_bytes"),
            add_eos=bool(item.get("add_eos", cfg.get("add_eos", True))),
        )
        rows.append({"output": item["output_path"], "tokens": manifest["token_count"], "manifest": str(encoded_manifest_path(item["output_path"]))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_bytes", type=int, default=None)
    args = parser.parse_args()
    if args.config:
        print(json.dumps(_run_config(args.config), indent=2))
        return
    if not (args.tokenizer and args.input and args.output):
        parser.error("either --config or --tokenizer/--input/--output is required")
    tok = ByteBPETokenizer.load(args.tokenizer)
    manifest = encode_text_file_with_manifest(args.input, tok, args.output, tokenizer_path=args.tokenizer, max_bytes=args.max_bytes)
    print({"output": args.output, "tokens": manifest["token_count"], "manifest": str(encoded_manifest_path(args.output))})


if __name__ == "__main__":
    main()
