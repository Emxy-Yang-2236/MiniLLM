from __future__ import annotations

from adapters import build_sft_release_rows, grade_sft_response, summarize_sft_scores


def test_sft_release_primary_tasks_and_counts():
    rows = build_sft_release_rows(seed=0)
    assert len(rows["train"]) == 6000
    assert len(rows["valid"]) == 1000
    assert len(rows["eval"]) == 600
    assert len(rows["fixed_prompts"]) == 16
    primary_train_tasks = {row["task_type"] for row in rows["train"]}
    primary_eval_tasks = {row["task_type"] for row in rows["eval"]}
    fixed_tasks = {row["task_type"] for row in rows["fixed_prompts"]}
    assert primary_train_tasks == {"story_template", "label_template"}
    assert primary_eval_tasks == {"story_template", "label_template"}
    assert fixed_tasks == {"story_template", "label_template"}
    assert all(row["task_type"] != "arithmetic_json" for split in rows.values() for row in split)
    assert all("Card:" not in row["prompt"] for split in rows.values() for row in split)


def _label_texts(rows):
    texts = set()
    for row in rows:
        if row["task_type"] != "label_template":
            continue
        texts.add(row["prompt"].split("Text: ", 1)[1].split("\nAnswer:", 1)[0])
    return texts


def test_sft_label_text_is_disjoint_across_splits():
    rows = build_sft_release_rows(seed=0)
    split_names = ["train", "valid", "eval", "fixed_prompts"]
    label_texts = {name: _label_texts(rows[name]) for name in split_names}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            assert not (label_texts[left] & label_texts[right])


def test_story_template_grading_checks_shape_and_topic():
    row = {
        "task_type": "story_template",
        "topic": "a brave mouse",
        "name": "Lily",
    }
    text = (
        "Once there was a brave mouse named Lily. "
        "Lily wanted to help a friend. "
        "One day, Lily found a lost button. "
        "In the end, Lily felt proud and happy."
    )
    score = grade_sft_response(text, row)
    assert score["template_followed"] == 1.0
    assert score["topic_match"] == 1.0
    assert score["sentence_count_ok"] == 1.0


def test_label_template_grading_requires_single_allowed_label():
    row = {
        "task_type": "label_template",
        "label": "sad",
        "allowed_labels": ["happy", "sad", "scared", "angry"],
    }
    assert grade_sft_response("sad", row)["exact_match"] == 1.0
    assert grade_sft_response("sad because he cried", row)["format_accuracy"] == 0.0
    assert grade_sft_response("happy", row)["exact_match"] == 0.0


def test_sft_score_summary_reports_new_task_prefixes():
    rows = [
        {
            "task_type": "story_template",
            **grade_sft_response(
                "Once there was a brave mouse named Lily. Lily wanted to help. One day, Lily tried again. In the end, Lily felt happy.",
                {"task_type": "story_template", "topic": "a brave mouse", "name": "Lily"},
            ),
        },
        {"task_type": "label_template", **grade_sft_response("happy", {"task_type": "label_template", "label": "happy", "allowed_labels": ["happy", "sad"]})},
    ]
    summary = summarize_sft_scores(rows)
    assert summary["count/story_template"] == 1.0
    assert summary["count/label_template"] == 1.0
    assert "story_template/template_followed" in summary
    assert "label_template/exact_match" in summary
