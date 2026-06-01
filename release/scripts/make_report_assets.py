#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "report_summary.json"
    metrics = []
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {"run_dir": str(run_dir), "num_metric_rows": len(metrics), "last_metric": metrics[-1] if metrics else None}
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
