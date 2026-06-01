#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.data.release import write_sft_release_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="../data/full_release")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (ROOT / dataset_dir).resolve()
    manifest = write_sft_release_data(dataset_dir, seed=args.seed)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
