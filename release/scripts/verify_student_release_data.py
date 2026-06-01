#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.data.release import require_valid_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "student"], default="student", help="Accepted for command symmetry.")
    parser.add_argument("--dataset_dir", default="../data/full_release")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (ROOT / dataset_dir).resolve()
    print(json.dumps(require_valid_manifest(dataset_dir), indent=2))


if __name__ == "__main__":
    main()
