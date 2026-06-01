#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "scripts/run_student_pipeline.py",
            "--mode",
            "smoke",
            "--device",
            args.device,
            "--output_dir",
            args.output_dir,
        ],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
