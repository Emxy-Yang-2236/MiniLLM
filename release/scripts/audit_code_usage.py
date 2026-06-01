#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


MAIN_PIPELINE = {
    "data/pretrain_dataset.py",
    "data/release.py",
    "data/sft_dataset.py",
    "eval/arithmetic.py",
    "eval/sft.py",
    "eval/story.py",
    "model/attention.py",
    "model/config.py",
    "model/generation.py",
    "model/layers.py",
    "model/transformer.py",
    "systems/analysis.py",
    "systems/benchmark.py",
    "tokenizer/bpe.py",
    "train/checkpoint.py",
    "train/metrics.py",
    "train/optim.py",
    "train/pretrain.py",
    "train/schedules.py",
    "train/seed.py",
    "train/sft.py",
    "train/state.py",
    "utils/config.py",
    "utils/device.py",
}

PUBLIC_TESTS = {
    "data/pretrain_dataset.py",
    "data/release.py",
    "data/sft_dataset.py",
    "eval/arithmetic.py",
    "eval/sft.py",
    "eval/story.py",
    "model/attention.py",
    "model/config.py",
    "model/generation.py",
    "model/layers.py",
    "model/transformer.py",
    "systems/analysis.py",
    "systems/benchmark.py",
    "tokenizer/bpe.py",
    "train/checkpoint.py",
    "train/metrics.py",
    "train/optim.py",
    "train/overfit.py",
    "train/pretrain.py",
    "train/schedules.py",
    "train/seed.py",
    "train/sft.py",
    "train/state.py",
}

OPTIONAL_COMMANDS = {
    "utils/config.py": [
        "scripts/train_pretrain.py",
        "scripts/train_sft.py",
        "scripts/run_sft_demo.py",
        "scripts/check_attention_backends.py",
        "scripts/benchmark_sweep.py",
    ],
    "utils/device.py": ["scripts/generate.py", "scripts/eval_sft.py", "scripts/sample_before_after.py", "scripts/run_sft_demo.py"],
}


def classify(rel: str) -> dict:
    if rel.endswith("__init__.py"):
        return {"status": "trivial_init", "used_by": ["package exports"]}
    used_by: list[str] = []
    if rel in MAIN_PIPELINE:
        used_by.append("main_pipeline")
    if rel in PUBLIC_TESTS:
        used_by.append("public_tests")
    if rel in OPTIONAL_COMMANDS:
        used_by.extend(f"optional:{cmd}" for cmd in OPTIONAL_COMMANDS[rel])
    return {"status": "used" if used_by else "unused", "used_by": used_by}


def audit(root: str | Path) -> dict:
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        info = classify(rel)
        rows.append({"file": rel, **info})
    unused = [row["file"] for row in rows if row["status"] == "unused"]
    return {"root": str(root), "files": rows, "unused": unused, "ok": not unused}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="minillm")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fail-on-unused", action="store_true")
    args = parser.parse_args()
    report = audit(args.root)
    text = json.dumps(report, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.fail_on_unused and report["unused"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
