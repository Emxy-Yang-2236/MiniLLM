from __future__ import annotations

from pathlib import Path

from minillm.model.generation import generate


def load_story_prompts(path: str | Path) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [prompt for prompt in prompts if prompt]


def story_format_score(text: str) -> dict[str, float]:
    stripped = text.strip()
    words = [word for word in stripped.replace("\n", " ").split(" ") if word]
    sentence_marks = sum(stripped.count(mark) for mark in ".!?")
    return {
        "nonempty": float(bool(stripped)),
        "word_count": float(len(words)),
        "sentence_count": float(sentence_marks),
        "two_sentence_like": float(sentence_marks >= 2),
    }


def generate_story_samples(
    model,
    tokenizer,
    prompts: list[str] | str | Path,
    device=None,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> list[dict]:
    if not isinstance(prompts, list):
        prompts = load_story_prompts(prompts)
    rows: list[dict] = []
    for prompt in prompts:
        suffix = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        rows.append({"prompt": prompt, "output": suffix, **story_format_score(suffix)})
    return rows


def summarize_story_samples(samples: list[dict]) -> dict[str, float]:
    if not samples:
        return {"n": 0.0, "nonempty": 0.0, "word_count": 0.0, "sentence_count": 0.0, "two_sentence_like": 0.0}
    keys = ["nonempty", "word_count", "sentence_count", "two_sentence_like"]
    return {
        "n": float(len(samples)),
        **{key: sum(float(row[key]) for row in samples) / len(samples) for key in keys},
    }
