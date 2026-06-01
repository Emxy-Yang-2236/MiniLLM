from __future__ import annotations

import json
import re
from pathlib import Path

from minillm.model.generation import generate


ANSWER_RE = re.compile(r"\{[^{}]*\"answer\"\s*:\s*(-?\d+)[^{}]*\}")


def extract_answer(text: str) -> tuple[int | None, bool]:
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("answer"), int):
            return int(data["answer"]), True
    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group(0)), False
    return None, False


def format_valid(text: str) -> bool:
    text = text.strip()
    if not ANSWER_RE.fullmatch(text):
        return False
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    return set(data.keys()) == {"answer"} and isinstance(data["answer"], int)


def grade_response(text: str, answer: int) -> dict[str, float]:
    pred, valid_json = extract_answer(text)
    return {
        "json_valid": float(valid_json),
        "format_accuracy": float(format_valid(text)),
        "exact_match": float(pred == answer),
        "length": float(len(text)),
    }


def evaluate_arithmetic(model, tokenizer, path: str | Path, device=None, max_examples: int | None = None) -> dict[str, float]:
    """Small diagnostic evaluator, not part of the main SFT success criteria."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_examples:
        rows = rows[:max_examples]
    totals = {"json_valid": 0.0, "format_accuracy": 0.0, "exact_match": 0.0, "length": 0.0}
    outputs = []
    for row in rows:
        suffix = generate(model, tokenizer, row["prompt"], max_new_tokens=32, temperature=0.0, device=device)
        score = grade_response(suffix, int(row["answer"]))
        outputs.append({"prompt": row["prompt"], "output": suffix, **score})
        for key in totals:
            totals[key] += score[key]
    n = max(1, len(rows))
    return {key: value / n for key, value in totals.items()} | {"n": float(len(rows)), "outputs": outputs}
