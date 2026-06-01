from __future__ import annotations

import json

from adapters import (
    build_model,
    evaluate_arithmetic,
    grade_arithmetic,
    make_config,
    story_format_score,
    summarize_story_samples,
    train_bpe,
)


def test_arithmetic_grade_response_checks_json_format():
    assert grade_arithmetic('{"answer": 5}', 5)["format_accuracy"] == 1.0
    assert grade_arithmetic('{"answer": 5, "extra": true}', 5)["json_valid"] == 1.0
    assert grade_arithmetic('{"answer": 5, "extra": true}', 5)["format_accuracy"] == 0.0
    assert grade_arithmetic("the answer is 5", 5)["exact_match"] == 1.0


def test_story_format_summary_scores_two_sentence_outputs():
    score = story_format_score("A tiny robot helped. The library was calm.")
    assert score["nonempty"] == 1.0
    assert score["two_sentence_like"] == 1.0
    summary = summarize_story_samples([{"nonempty": 1.0, "word_count": 4.0, "sentence_count": 2.0, "two_sentence_like": 1.0}])
    assert summary["n"] == 1.0
    assert summary["sentence_count"] == 2.0


def test_arithmetic_eval_schema(tmp_path):
    rows = [
        {"prompt": "Return JSON. What is 1 + 1?", "response": '{"answer": 2}', "answer": 2},
        {"prompt": "Return JSON. What is 3 + 4?", "response": '{"answer": 7}', "answer": 7},
    ]
    tok = train_bpe([row["prompt"] + row["response"] for row in rows], vocab_size=280)
    cfg = make_config(vocab_size=tok.vocab_size, context_length=32)
    model = build_model(cfg)
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    scores = evaluate_arithmetic(model, tok, path, device="cpu", max_examples=1)
    assert {"json_valid", "format_accuracy", "exact_match", "length", "n", "outputs"} <= scores.keys()
    assert scores["n"] == 1.0
    assert isinstance(scores["outputs"], list)
