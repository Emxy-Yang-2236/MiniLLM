#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _label_texts(rows: list[dict]) -> set[str]:
    texts = set()
    for row in rows:
        if row.get("task_type") != "label_template":
            continue
        prompt = row["prompt"]
        if "Text: " in prompt and "\nAnswer:" in prompt:
            texts.add(prompt.split("Text: ", 1)[1].split("\nAnswer:", 1)[0])
    return texts


def check_sft_splits(dataset_dir: str | Path) -> dict:
    dataset_dir = Path(dataset_dir)
    train = _jsonl(dataset_dir / "sft" / "train.jsonl")
    valid = _jsonl(dataset_dir / "sft" / "valid.jsonl")
    eval_rows = _jsonl(dataset_dir / "sft" / "eval.jsonl")
    fixed = _jsonl(dataset_dir / "sft" / "fixed_prompts.jsonl")
    splits = {"train": train, "valid": valid, "eval": eval_rows, "fixed": fixed}
    prompts = {name: {row["prompt"] for row in rows} for name, rows in splits.items()}
    ids = {name: {row["id"] for row in rows if "id" in row} for name, rows in splits.items()}
    label_texts = {name: _label_texts(rows) for name, rows in splits.items()}
    pair_names = [("train", "valid"), ("train", "eval"), ("train", "fixed"), ("valid", "eval"), ("valid", "fixed"), ("eval", "fixed")]
    report = {
        **{f"{left}_{right}_prompt_overlap_count": len(prompts[left] & prompts[right]) for left, right in pair_names},
        **{f"{left}_{right}_id_overlap_count": len(ids[left] & ids[right]) for left, right in pair_names},
        **{f"{left}_{right}_label_text_overlap_count": len(label_texts[left] & label_texts[right]) for left, right in pair_names},
        "train_tasks": sorted({row.get("task_type") for row in train}),
        "valid_tasks": sorted({row.get("task_type") for row in valid}),
        "eval_tasks": sorted({row.get("task_type") for row in eval_rows}),
        "fixed_tasks": sorted({row.get("task_type") for row in fixed}),
        "arithmetic_in_main_splits": any(row.get("task_type") == "arithmetic_json" for rows in splits.values() for row in rows),
    }
    report["ok"] = all(value == 0 for key, value in report.items() if key.endswith("_overlap_count")) and not report["arithmetic_in_main_splits"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="../data/full_release")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (ROOT / dataset_dir).resolve()
    report = check_sft_splits(dataset_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
