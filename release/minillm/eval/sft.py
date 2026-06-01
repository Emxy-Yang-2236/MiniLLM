from __future__ import annotations

import json
import re
from pathlib import Path

from minillm.eval.arithmetic import grade_response
from minillm.model.generation import generate


def _json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def grade_json_format(text: str, expected: dict) -> dict[str, float]:
    data = _json_object(text)
    return {
        "json_valid": float(data is not None),
        "json_schema_accuracy": float(data is not None and set(data.keys()) == set(expected.keys())),
        "exact_match": float(data == expected),
        "format_accuracy": float(data == expected),
        "length": float(len(text)),
    }


def grade_json_field_copy(text: str, expected: dict) -> dict[str, float]:
    data = _json_object(text)
    schema_ok = data is not None and set(data.keys()) == set(expected.keys())
    fields_ok = data == expected
    return {
        "json_valid": float(data is not None),
        "json_schema_accuracy": float(schema_ok),
        "field_exact_match": float(fields_ok),
        "no_extra_prose": float(data is not None and text.strip().startswith("{") and text.strip().endswith("}")),
        "format_accuracy": float(schema_ok),
        "exact_match": float(fields_ok),
        "length": float(len(text)),
    }


def grade_label_classification(text: str, expected: str, allowed_labels: list[str] | None = None) -> dict[str, float]:
    stripped = text.strip()
    normalized = stripped.lower().strip(" .\n\t")
    expected_norm = expected.lower()
    allowed = {label.lower() for label in (allowed_labels or [])}
    return {
        "label_exact_match": float(normalized == expected_norm),
        "format_accuracy": float((not allowed or normalized in allowed) and len(normalized.split()) == 1),
        "no_extra_prose": float(normalized == stripped.lower().strip(" .\n\t")),
        "length": float(len(text)),
    }


def grade_label_template(text: str, expected: str, allowed_labels: list[str] | None = None) -> dict[str, float]:
    score = grade_label_classification(text, expected, allowed_labels)
    score["exact_match"] = score["label_exact_match"]
    return score


def _topic_words(topic: str) -> set[str]:
    stop = {"a", "an", "the", "and", "who", "to", "about", "short", "story", "tiny"}
    return {word for word in re.findall(r"[a-z]+", topic.lower()) if word not in stop}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def grade_story_template(text: str, row: dict) -> dict[str, float]:
    stripped = text.strip()
    sentences = _sentences(stripped)
    words = [word for word in stripped.replace("\n", " ").split(" ") if word]
    topic = row.get("topic") or row.get("prompt", "")
    topic_words = _topic_words(topic)
    lower = stripped.lower()
    topic_hit = bool(topic_words) and all(word in lower for word in topic_words)
    expected_name = row.get("name")
    first = sentences[0] if sentences else ""
    name_match = re.match(r"^Once there was .+ named ([A-Z][A-Za-z]+)\.$", first)
    name = expected_name or (name_match.group(1) if name_match else "")
    template_ok = (
        len(sentences) == 4
        and bool(name_match)
        and (not expected_name or name == expected_name)
        and len(sentences) > 1
        and sentences[1].startswith(f"{name} wanted to ")
        and len(sentences) > 2
        and sentences[2].startswith(f"One day, {name} ")
        and len(sentences) > 3
        and sentences[3].startswith(f"In the end, {name} felt ")
    )
    sentence_ok = 3 <= len(sentences) <= 4
    word_count_ok = 16 <= len(words) <= 80
    return {
        "template_followed": float(template_ok),
        "topic_match": float(topic_hit),
        "sentence_count_ok": float(sentence_ok),
        "word_count_ok": float(word_count_ok),
        "format_accuracy": float(template_ok and topic_hit),
        "length": float(len(text)),
        "word_count": float(len(words)),
        "sentence_count": float(len(sentences)),
    }


def grade_story(text: str, topic: str) -> dict[str, float]:
    stripped = text.strip()
    words = [word for word in stripped.replace("\n", " ").split(" ") if word]
    sentence_count = sum(stripped.count(mark) for mark in ".!?")
    topic_words = _topic_words(topic)
    lower = stripped.lower()
    topic_hit = bool(topic_words) and any(word in lower for word in topic_words)
    sentence_ok = 2 <= sentence_count <= 4
    return {
        "story_topic_match": float(topic_hit),
        "story_sentence_count_ok": float(sentence_ok),
        "story_instruction_followed": float(bool(stripped) and topic_hit and sentence_count >= 1),
        "length": float(len(text)),
        "word_count": float(len(words)),
        "sentence_count": float(sentence_count),
    }


def grade_sft_response(text: str, row: dict) -> dict[str, float]:
    task = row.get("task_type")
    if task == "arithmetic_json":
        return grade_response(text, int(row["answer"]))
    if task == "story_template":
        return grade_story_template(text, row)
    if task == "label_template":
        return grade_label_template(text, row.get("label") or row.get("expected", ""), row.get("allowed_labels"))
    if task == "json_format":
        expected = row.get("expected_json")
        if expected is None and row.get("expected"):
            expected = json.loads(row["expected"])
        return grade_json_format(text, expected or {})
    if task == "json_field_copy":
        expected = row.get("expected_json")
        if expected is None and row.get("expected"):
            expected = json.loads(row["expected"])
        return grade_json_field_copy(text, expected or {})
    if task == "label_classification":
        return grade_label_classification(text, row.get("label") or row.get("expected", ""), row.get("allowed_labels"))
    if task == "story_instruction":
        return grade_story(text, row.get("topic") or row.get("prompt", ""))
    return {"length": float(len(text)), "nonempty": float(bool(text.strip()))}


def summarize_scores(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"n": 0.0}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for row in rows:
        task = row.get("task_type", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {"answer"}:
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
                totals[f"{task}/{key}"] = totals.get(f"{task}/{key}", 0.0) + float(value)
                counts[f"{task}/{key}"] = counts.get(f"{task}/{key}", 0) + 1
    out = {"n": float(len(rows)), **{f"count/{task}": float(count) for task, count in task_counts.items()}}
    out.update({key: totals[key] / counts[key] for key in sorted(totals)})
    return out


def evaluate_sft(model, tokenizer, path: str | Path, device=None, max_examples: int | None = None) -> dict:
    dataset_rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_examples:
        dataset_rows = dataset_rows[:max_examples]
    outputs = []
    for row in dataset_rows:
        max_new_tokens = 12 if row.get("task_type") in {"label_template", "label_classification"} else 96
        suffix = generate(model, tokenizer, row["prompt"], max_new_tokens=max_new_tokens, temperature=0.0, device=device)
        score = grade_sft_response(suffix, row)
        outputs.append({"id": row.get("id"), "task_type": row.get("task_type"), "prompt": row["prompt"], "output": suffix, **score})
    return {**summarize_scores(outputs), "outputs": outputs}
