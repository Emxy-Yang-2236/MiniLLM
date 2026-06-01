#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "\n".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}" />' for x, y in points)
    return f'<polyline points="{data}" fill="none" stroke="{color}" stroke-width="2.2" />\n{circles}'


def _panel(rows: list[dict], title: str, x: int, y: int, w: int, h: int) -> str:
    if not rows:
        return f'<text x="{x}" y="{y + 20}" class="title">{escape(title)}: no metrics</text>'

    steps = [float(row["step"]) for row in rows]
    losses = [float(row[key]) for row in rows for key in ("train_loss", "valid_loss") if key in row and row[key] is not None]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(losses), max(losses)
    y_pad = max(0.05, (y_max - y_min) * 0.08)
    y_min = max(0.0, y_min - y_pad)
    y_max = y_max + y_pad

    left = x + 58
    right = x + w - 18
    top = y + 36
    bottom = y + h - 42

    def pt(row: dict, key: str) -> tuple[float, float] | None:
        if key not in row or row[key] is None:
            return None
        px = _scale(float(row["step"]), x_min, x_max, left, right)
        py = _scale(float(row[key]), y_min, y_max, bottom, top)
        return px, py

    train = [p for row in rows if (p := pt(row, "train_loss")) is not None]
    valid = [p for row in rows if (p := pt(row, "valid_loss")) is not None]

    grid = []
    for i in range(5):
        yy = top + (bottom - top) * i / 4
        val = y_max - (y_max - y_min) * i / 4
        grid.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid" />')
        grid.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{val:.2f}</text>')
    for i in range(5):
        xx = left + (right - left) * i / 4
        val = x_min + (x_max - x_min) * i / 4
        grid.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" class="grid" />')
        grid.append(f'<text x="{xx:.1f}" y="{bottom + 18}" text-anchor="middle" class="tick">{int(val)}</text>')

    first = rows[0]
    last = rows[-1]
    summary = (
        f'train {float(first["train_loss"]):.3f} -> {float(last["train_loss"]):.3f}, '
        f'valid {float(first["valid_loss"]):.3f} -> {float(last["valid_loss"]):.3f}'
    )
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="panel" />',
            f'<text x="{x + 18}" y="{y + 24}" class="title">{escape(title)}</text>',
            f'<text x="{x + w - 18}" y="{y + 24}" text-anchor="end" class="subtitle">{escape(summary)}</text>',
            *grid,
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis" />',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis" />',
            f'<text x="{(left + right) / 2:.1f}" y="{y + h - 8}" text-anchor="middle" class="label">step</text>',
            f'<text x="{x + 16}" y="{(top + bottom) / 2:.1f}" transform="rotate(-90 {x + 16} {(top + bottom) / 2:.1f})" text-anchor="middle" class="label">loss</text>',
            _polyline(train, "#2563eb"),
            _polyline(valid, "#dc2626"),
            f'<rect x="{right - 138}" y="{top + 8}" width="122" height="42" rx="4" class="legend-bg" />',
            f'<line x1="{right - 128}" y1="{top + 22}" x2="{right - 106}" y2="{top + 22}" stroke="#2563eb" stroke-width="2.2" />',
            f'<text x="{right - 98}" y="{top + 26}" class="legend">train_loss</text>',
            f'<line x1="{right - 128}" y1="{top + 40}" x2="{right - 106}" y2="{top + 40}" stroke="#dc2626" stroke-width="2.2" />',
            f'<text x="{right - 98}" y="{top + 44}" class="legend">valid_loss</text>',
        ]
    )


def write_training_curves(pretrain_metrics: Path, sft_metrics: Path, output: Path) -> None:
    pretrain_rows = _read_jsonl(pretrain_metrics)
    sft_rows = _read_jsonl(sft_metrics)
    width = 980
    height = 660
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .panel {{ fill: #ffffff; stroke: #d1d5db; stroke-width: 1; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .axis {{ stroke: #374151; stroke-width: 1.2; }}
  .title {{ font-size: 17px; font-weight: 700; }}
  .subtitle {{ font-size: 12px; fill: #4b5563; }}
  .tick {{ font-size: 11px; fill: #4b5563; }}
  .label {{ font-size: 12px; fill: #374151; }}
  .legend {{ font-size: 12px; fill: #374151; }}
  .legend-bg {{ fill: #ffffff; stroke: #e5e7eb; }}
</style>
<rect width="100%" height="100%" fill="#f9fafb" />
<text x="24" y="30" class="title">MiniLLM Training Curves</text>
{_panel(pretrain_rows, "Pretraining", 24, 52, 932, 276)}
{_panel(sft_rows, "SFT", 24, 356, 932, 276)}
</svg>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_metrics", default="outputs/release_candidate/metrics_pretrain.jsonl")
    parser.add_argument("--sft_metrics", default="outputs/release_candidate/metrics_sft.jsonl")
    parser.add_argument("--output", default="outputs/release_candidate/training_curves.svg")
    args = parser.parse_args()
    write_training_curves(Path(args.pretrain_metrics), Path(args.sft_metrics), Path(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
