#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.data.release import prepare_student_release_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="../data/full_release")
    parser.add_argument("--train_bytes", type=int, default=536_870_912)
    parser.add_argument("--valid_bytes", type=int, default=33_554_432)
    parser.add_argument("--tokenizer_bytes", type=int, default=134_217_728)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (ROOT / dataset_dir).resolve()
    report = prepare_student_release_dataset(
        dataset_dir,
        train_bytes=args.train_bytes,
        valid_bytes=args.valid_bytes,
        tokenizer_bytes=args.tokenizer_bytes,
        force=args.force,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
