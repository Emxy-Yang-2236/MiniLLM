from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path


TINYSTORIES_TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
LICENSE = "CDLA-Sharing-1.0"
MANIFEST_VERSION = 10
SFT_SEED = 0
TRAIN_FILENAME = "TinyStoriesV2-GPT4-train.txt"
VALID_FILENAME = "TinyStoriesV2-GPT4-valid.txt"
LARGE_TEXT_VERIFY_BYTES = 100 * 1024 * 1024


def load_manifest(dataset_dir: str | Path) -> dict:
    return json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))


def resolve_dataset_path(dataset_dir: str | Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(dataset_dir) / path).resolve()


def dataset_path_for_split(dataset_dir: str | Path, manifest: dict, split: str) -> Path:
    for entry in manifest["files"]:
        if entry["split"] == split:
            return resolve_dataset_path(dataset_dir, entry["path"])
    raise KeyError(f"split {split!r} not found in {Path(dataset_dir) / 'manifest.json'}")


def file_stats(path: str | Path) -> dict:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    num_examples = 0
    whitespace_tokens = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                num_examples += 1
                whitespace_tokens += len(line.split())
    return {
        "num_examples": num_examples,
        "whitespace_tokens": whitespace_tokens,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def verify_manifest(dataset_dir: str | Path) -> dict:
    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    rows = []
    ok = True
    for entry in manifest["files"]:
        path = resolve_dataset_path(dataset_dir, entry["path"])
        if path.stat().st_size > LARGE_TEXT_VERIFY_BYTES and entry.get("format") == "plain_text":
            bytes_match = path.stat().st_size == int(entry["bytes"])
            actual = {key: entry[key] for key in ["num_examples", "whitespace_tokens", "sha256"]}
            actual["bytes"] = path.stat().st_size
            actual["verification_mode"] = "cached_large_text_stats_size_checked"
            expected = {key: entry[key] for key in ["num_examples", "whitespace_tokens", "bytes", "sha256"]}
            matches = bytes_match
            rows.append({"path": entry["path"], "ok": matches, "expected": expected, "actual": actual})
            ok = ok and matches
            continue
        actual = file_stats(path)
        expected = {key: entry[key] for key in ["num_examples", "whitespace_tokens", "bytes", "sha256"]}
        matches = actual == expected
        rows.append({"path": entry["path"], "ok": matches, "expected": expected, "actual": actual})
        ok = ok and matches
    sft_manifest_path = dataset_dir / "sft" / "manifest.json"
    sft_manifest_ok = True
    if sft_manifest_path.exists():
        sft_manifest = json.loads(sft_manifest_path.read_text(encoding="utf-8"))
        for entry in sft_manifest.get("files", []):
            actual = file_stats(resolve_dataset_path(dataset_dir, entry["path"]))
            expected = {key: entry[key] for key in ["num_examples", "whitespace_tokens", "bytes", "sha256"]}
            if actual != expected:
                sft_manifest_ok = False
                rows.append({"path": entry["path"], "ok": False, "expected": expected, "actual": actual})
    return {"ok": ok and sft_manifest_ok, "name": manifest["name"], "version": manifest["version"], "files": rows}


def require_valid_manifest(dataset_dir: str | Path) -> dict:
    result = verify_manifest(dataset_dir)
    if not result["ok"]:
        bad = [row["path"] for row in result["files"] if not row["ok"]]
        raise RuntimeError(f"dataset manifest verification failed for: {bad}")
    return result


def _valid_utf8_prefix(data: bytes) -> bytes:
    while data:
        try:
            data.decode("utf-8")
            return data
        except UnicodeDecodeError as exc:
            data = data[: exc.start]
    return b""


def _download_prefix_file(url: str, path: Path, num_bytes: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{num_bytes - 1}"})
    written = 0
    tail = b""
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
        while written < num_bytes:
            chunk = response.read(min(1024 * 1024, num_bytes - written))
            if not chunk:
                break
            if written + len(chunk) > num_bytes:
                chunk = chunk[: num_bytes - written]
            tail = (tail + chunk)[-8:]
            out.write(chunk)
            written += len(chunk)
    if written == 0:
        raise RuntimeError(f"downloaded zero bytes from {url}")
    valid_tail = _valid_utf8_prefix(tail)
    trim = len(tail) - len(valid_tail)
    if trim:
        with path.open("rb+") as f:
            f.truncate(written - trim)
        written -= trim
    return written


def _copy_prefix_file(src: Path, dst: Path, num_bytes: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()[:num_bytes]
    data = _valid_utf8_prefix(data)
    dst.write_bytes(data)
    return len(data)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


STORY_TOPICS = [
    "a brave mouse",
    "a kind robot",
    "a lost puppy",
    "a tiny dragon",
    "a shy rabbit",
    "a red bird",
    "a small fox",
    "a sleepy bear",
    "a helpful train",
    "a gentle turtle",
    "a curious kitten",
    "a little sailor",
    "a magic seed",
    "a quiet moon",
    "a blue fish",
    "a happy frog",
    "a soft cloud",
    "a tiny painter",
    "a careful bee",
    "a lonely robot",
    "a green dinosaur",
    "a brave girl",
    "a little duck",
    "a friendly star",
]
STORY_GOALS = [
    "help a friend",
    "find a lost toy",
    "share a snack",
    "try again",
    "make a small gift",
    "sing a soft song",
    "clean the little room",
    "learn to be kind",
    "bring the kite home",
    "plant a tiny seed",
    "fix a red wagon",
    "tell the truth",
]
STORY_EVENTS = [
    "found a lost button",
    "shared a toy with Mia",
    "helped Tom cross the path",
    "picked up a fallen cup",
    "gave Nora a warm hug",
    "found a small door",
    "carried a kite to the park",
    "saw Ben sitting alone",
    "made a bridge from blocks",
    "heard a puppy cry",
    "put the seed in soft dirt",
    "gave a flower to Sara",
    "fixed the wagon wheel",
    "held a lamp in the dark",
    "found a blue ribbon",
    "waited and tried again",
]
STORY_FEELINGS = [
    "proud and happy",
    "kind and glad",
    "safe and warm",
    "brave and calm",
    "thankful and bright",
    "gentle and proud",
    "happy with a smile",
    "less lonely",
    "ready to try again",
    "quiet and safe",
]
NAMES = ["Lily", "Tom", "Mia", "Ben", "Sara", "Nora", "Bim", "Tim"]
LABELS = ["happy", "sad", "scared", "angry", "kind", "lonely"]
LABEL_OBJECTS = ["toy", "kite", "cup", "ball", "book", "wagon", "flower", "button", "block", "snack"]
LABEL_PLACES = ["park", "garden", "school", "forest", "library", "kitchen", "beach", "farm"]
LABEL_NOISES = [
    "thunder",
    "a loud bang",
    "a dark cave sound",
    "a big bark",
    "wind at the window",
    "a creak in the hall",
    "rain on the roof",
    "a faraway horn",
]
LABEL_TIMES = [
    "that morning",
    "after lunch",
    "before bedtime",
    "on a rainy day",
    "during playtime",
    "at sunset",
    "after school",
    "on a sunny day",
    "in the evening",
    "before dinner",
    "at noon",
    "on a windy day",
]


def _task_prompt(task: str, prompt: str) -> str:
    return f"### Task: {task}\n### Prompt:\n{prompt}\n### Response:"


def _story_fields(idx: int) -> dict[str, str]:
    return {
        "topic": STORY_TOPICS[idx % len(STORY_TOPICS)],
        "name": NAMES[idx % len(NAMES)],
        "goal": STORY_GOALS[(idx // len(STORY_TOPICS)) % len(STORY_GOALS)],
        "event": STORY_EVENTS[(idx // (len(STORY_TOPICS) * len(STORY_GOALS))) % len(STORY_EVENTS)],
        "emotion": STORY_FEELINGS[
            (idx // (len(STORY_TOPICS) * len(STORY_GOALS) * len(STORY_EVENTS))) % len(STORY_FEELINGS)
        ],
    }


def _story_response(topic: str, name: str, goal: str, event: str, emotion: str) -> str:
    return (
        f"Once there was {topic} named {name}.\n"
        f"{name} wanted to {goal}.\n"
        f"One day, {name} {event}.\n"
        f"In the end, {name} felt {emotion}."
    )


def _story_prompt(fields: dict[str, str], split: str) -> str:
    openers = {
        "train": "Write a short story",
        "valid": "Write a tiny story",
        "eval": "Write a short children's story",
        "fixed": "Write a short story",
    }
    opener = openers.get(split, openers["train"])
    return (
        f"{opener} about {fields['topic']}.\n"
        f"Goal: {fields['goal']}.\n"
        f"Event: {fields['event']}.\n"
        f"Feeling: {fields['emotion']}."
    )


def _story_rows(split: str, count: int, offset: int = 0) -> list[dict]:
    rows = []
    for i in range(count):
        fields = _story_fields(i + offset)
        plain_prompt = _story_prompt(fields, split)
        rows.append(
            {
                "id": f"story_{split}_{i:05d}",
                "task_type": "story_template",
                "topic": fields["topic"],
                "name": fields["name"],
                "goal": fields["goal"],
                "event": fields["event"],
                "emotion": fields["emotion"],
                "prompt": _task_prompt("story", plain_prompt),
                "response": _story_response(**fields),
                "source": "assignment-authored deterministic story template task",
            }
        )
    return rows


def _label_text(label: str, idx: int) -> str:
    name = NAMES[idx % len(NAMES)]
    friend = NAMES[(idx + 3) % len(NAMES)]
    item = LABEL_OBJECTS[(idx // len(NAMES)) % len(LABEL_OBJECTS)]
    place = LABEL_PLACES[(idx // (len(NAMES) * len(LABEL_OBJECTS))) % len(LABEL_PLACES)]
    noise = LABEL_NOISES[(idx // (len(NAMES) * len(LABEL_OBJECTS) * len(LABEL_PLACES))) % len(LABEL_NOISES)]
    time_phrase = LABEL_TIMES[(idx // (len(NAMES) * len(LABEL_OBJECTS) * len(LABEL_PLACES))) % len(LABEL_TIMES)]
    if label == "happy":
        return f"{name} found a {item} at the {place} {time_phrase} and smiled."
    if label == "sad":
        return f"{name} lost a {item} at the {place} {time_phrase} and cried softly."
    if label == "scared":
        return f"{name} heard {noise} near the {place} {time_phrase} and hid."
    if label == "angry":
        return f"{name} frowned at the {place} because {friend} broke the {item} {time_phrase}."
    if label == "kind":
        return f"{name} helped {friend} carry a {item} at the {place} {time_phrase}."
    if label == "lonely":
        return f"{name} sat alone by the {place} {time_phrase} and wished for a friend."
    raise ValueError(label)


def _label_prompt(text: str, split: str) -> str:
    openers = {
        "train": "Choose one feeling label",
        "valid": "Read the sentence and choose one feeling label",
        "eval": "Pick the best feeling label",
        "fixed": "Choose the best feeling label",
    }
    opener = openers.get(split, openers["train"])
    return f"{opener}: {', '.join(LABELS)}.\nText: {text}\nAnswer:"


def _label_rows(split: str, count: int, offset: int = 0) -> list[dict]:
    rows = []
    for i in range(count):
        label = LABELS[(i + offset) % len(LABELS)]
        text = _label_text(label, (i + offset) // len(LABELS))
        plain_prompt = _label_prompt(text, split)
        rows.append(
            {
                "id": f"label_{split}_{i:05d}",
                "task_type": "label_template",
                "prompt": _task_prompt("label", plain_prompt),
                "response": label,
                "label": label,
                "allowed_labels": LABELS,
                "source": "assignment-authored deterministic label template task",
            }
        )
    return rows


def build_sft_release_rows(seed: int = SFT_SEED) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    train = _story_rows("train", 4200, 0) + _label_rows("train", 1800, 0)
    valid = _story_rows("valid", 700, 10000) + _label_rows("valid", 300, 10000)
    eval_rows = _story_rows("eval", 300, 20000) + _label_rows("eval", 300, 20000)
    fixed = _fixed_prompt_rows()
    rng.shuffle(train)
    rng.shuffle(valid)
    return {"train": train, "valid": valid, "eval": eval_rows, "fixed_prompts": fixed}


def _fixed_prompt_rows() -> list[dict]:
    rows: list[dict] = []
    for i in range(8):
        fields = _story_fields(30000 + i)
        plain_prompt = _story_prompt(fields, "fixed")
        rows.append(
            {
                "id": f"fixed_story_{i:02d}",
                "task_type": "story_template",
                "topic": fields["topic"],
                "name": fields["name"],
                "prompt": _task_prompt("story", plain_prompt),
                "expected": _story_response(**fields),
            }
        )
    for i in range(8):
        label = LABELS[(30000 + i) % len(LABELS)]
        text = _label_text(label, (30000 + i) // len(LABELS))
        plain_prompt = _label_prompt(text, "fixed")
        rows.append(
            {
                "id": f"fixed_label_{i:02d}",
                "task_type": "label_template",
                "prompt": _task_prompt("label", plain_prompt),
                "expected": label,
                "label": label,
                "allowed_labels": LABELS,
            }
        )
    return rows


def _manifest_entry(dataset_dir: Path, path: str | Path, split: str, source: str, file_format: str) -> dict:
    actual_path = resolve_dataset_path(dataset_dir, path)
    stored_path = os.path.relpath(actual_path, dataset_dir)
    stats = file_stats(actual_path)
    return {"path": stored_path, "split": split, "format": file_format, "source": source, **stats}


def _cached_manifest_entry(
    dataset_dir: Path,
    cached: dict[str, dict],
    path: str | Path,
    split: str,
    source: str,
    file_format: str,
) -> dict:
    actual_path = resolve_dataset_path(dataset_dir, path)
    stored_path = os.path.relpath(actual_path, dataset_dir)
    cached_entry = cached.get(split)
    if cached_entry and cached_entry.get("path") == stored_path and actual_path.stat().st_size == int(cached_entry["bytes"]):
        return {**cached_entry, "source": source, "format": file_format}
    return _manifest_entry(dataset_dir, path, split, source, file_format)


def _task_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row.get("task_type", "unknown") for row in rows))


def _label_texts(rows: list[dict]) -> set[str]:
    texts = set()
    for row in rows:
        if row.get("task_type") != "label_template":
            continue
        prompt = row["prompt"]
        marker = "Text: "
        if marker in prompt and "\nAnswer:" in prompt:
            texts.add(prompt.split(marker, 1)[1].split("\nAnswer:", 1)[0])
    return texts


def _check_sft_overlap(rows_by_split: dict[str, list[dict]]) -> dict:
    train_prompts = {row["prompt"] for row in rows_by_split["train"]}
    valid_prompts = {row["prompt"] for row in rows_by_split["valid"]}
    eval_prompts = {row["prompt"] for row in rows_by_split["eval"]}
    fixed_prompts = {row["prompt"] for row in rows_by_split["fixed_prompts"]}
    train_ids = {row["id"] for row in rows_by_split["train"]}
    valid_ids = {row["id"] for row in rows_by_split["valid"]}
    eval_ids = {row["id"] for row in rows_by_split["eval"]}
    fixed_ids = {row["id"] for row in rows_by_split["fixed_prompts"]}
    train_label_texts = _label_texts(rows_by_split["train"])
    valid_label_texts = _label_texts(rows_by_split["valid"])
    eval_label_texts = _label_texts(rows_by_split["eval"])
    fixed_label_texts = _label_texts(rows_by_split["fixed_prompts"])
    label_text_overlap = (
        train_label_texts & valid_label_texts
        or train_label_texts & eval_label_texts
        or train_label_texts & fixed_label_texts
        or valid_label_texts & eval_label_texts
        or valid_label_texts & fixed_label_texts
        or eval_label_texts & fixed_label_texts
    )
    return {
        "train_valid_prompt_overlap_count": len(train_prompts & valid_prompts),
        "train_eval_prompt_overlap_count": len(train_prompts & eval_prompts),
        "train_fixed_prompt_overlap_count": len(train_prompts & fixed_prompts),
        "valid_eval_prompt_overlap_count": len(valid_prompts & eval_prompts),
        "valid_fixed_prompt_overlap_count": len(valid_prompts & fixed_prompts),
        "eval_fixed_prompt_overlap_count": len(eval_prompts & fixed_prompts),
        "train_valid_id_overlap_count": len(train_ids & valid_ids),
        "train_eval_id_overlap_count": len(train_ids & eval_ids),
        "train_fixed_id_overlap_count": len(train_ids & fixed_ids),
        "valid_eval_id_overlap_count": len(valid_ids & eval_ids),
        "valid_fixed_id_overlap_count": len(valid_ids & fixed_ids),
        "eval_fixed_id_overlap_count": len(eval_ids & fixed_ids),
        "train_valid_label_text_overlap_count": len(train_label_texts & valid_label_texts),
        "train_eval_label_text_overlap_count": len(train_label_texts & eval_label_texts),
        "train_fixed_label_text_overlap_count": len(train_label_texts & fixed_label_texts),
        "valid_eval_label_text_overlap_count": len(valid_label_texts & eval_label_texts),
        "valid_fixed_label_text_overlap_count": len(valid_label_texts & fixed_label_texts),
        "eval_fixed_label_text_overlap_count": len(eval_label_texts & fixed_label_texts),
        "ok": not (
            train_prompts & valid_prompts
            or train_prompts & eval_prompts
            or train_prompts & fixed_prompts
            or valid_prompts & eval_prompts
            or valid_prompts & fixed_prompts
            or eval_prompts & fixed_prompts
            or train_ids & valid_ids
            or train_ids & eval_ids
            or train_ids & fixed_ids
            or valid_ids & eval_ids
            or valid_ids & fixed_ids
            or eval_ids & fixed_ids
            or label_text_overlap
        ),
    }


def write_sft_release_data(dataset_dir: str | Path, seed: int = SFT_SEED) -> dict:
    dataset_dir = Path(dataset_dir)
    rows_by_split = build_sft_release_rows(seed=seed)
    _write_jsonl(dataset_dir / "sft" / "train.jsonl", rows_by_split["train"])
    _write_jsonl(dataset_dir / "sft" / "valid.jsonl", rows_by_split["valid"])
    _write_jsonl(dataset_dir / "sft" / "eval.jsonl", rows_by_split["eval"])
    _write_jsonl(dataset_dir / "sft" / "fixed_prompts.jsonl", rows_by_split["fixed_prompts"])
    # Keep eval/ paths as aliases for the canonical pipeline and existing docs.
    _write_jsonl(dataset_dir / "eval" / "sft_eval.jsonl", rows_by_split["eval"])
    _write_jsonl(dataset_dir / "eval" / "fixed_prompts.jsonl", rows_by_split["fixed_prompts"])
    for stale in [dataset_dir / "sft" / "diagnostic_arithmetic.jsonl", dataset_dir / "eval" / "arithmetic_diagnostic.jsonl"]:
        if stale.exists():
            stale.unlink()
    overlap = _check_sft_overlap(rows_by_split)
    manifest = {
        "name": "minillm_sft_template_release_v6",
        "version": 6,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_script": "minillm.data.release.write_sft_release_data",
        "random_seed": seed,
        "split_policy": "Exact prompts and example ids are disjoint between train, eval, and fixed prompts. Main SFT tasks are template-conditioned story generation and single-label classification.",
        "task_types": ["story_template", "label_template"],
        "diagnostic_task_types": [],
        "prompt_templates": {
            "story_template": "natural 4-sentence TinyStories-style prompt with topic, goal, event, and feeling fields",
            "label_template": "natural single TinyStories-style emotion label prompt from a fixed label set",
        },
        "overlap_checks": overlap,
        "counts_by_split": {split: len(rows) for split, rows in rows_by_split.items()},
        "counts_by_task": {split: _task_counts(rows) for split, rows in rows_by_split.items()},
        "files": [
            _manifest_entry(dataset_dir, "sft/train.jsonl", "sft_train", "fixed deterministic template SFT train set", "jsonl"),
            _manifest_entry(dataset_dir, "sft/valid.jsonl", "sft_valid", "fixed deterministic template SFT validation set", "jsonl"),
            _manifest_entry(dataset_dir, "sft/eval.jsonl", "sft_eval", "held-out template SFT eval set", "jsonl"),
            _manifest_entry(dataset_dir, "sft/fixed_prompts.jsonl", "before_after_prompts", "held-out fixed template before/after prompts", "jsonl"),
            _manifest_entry(dataset_dir, "eval/sft_eval.jsonl", "sft_eval_alias", "alias of sft/eval.jsonl", "jsonl"),
            _manifest_entry(dataset_dir, "eval/fixed_prompts.jsonl", "before_after_prompts_alias", "alias of sft/fixed_prompts.jsonl", "jsonl"),
        ],
    }
    (dataset_dir / "sft" / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if not overlap["ok"]:
        raise RuntimeError(f"SFT split overlap check failed: {overlap}")
    return manifest


def _write_release_readme(dataset_dir: Path, train_path: Path, valid_path: Path) -> None:
    text = f"""# MiniLLM Student Release Dataset

This is the fixed dataset used by the main MiniLLM student pipeline.

## Source And Attribution

- Pretraining text: the full TinyStories V2 GPT-4 train/valid text files from Hugging Face.
- Train source: `{TINYSTORIES_TRAIN_URL}`
- Valid source: `{TINYSTORIES_VALID_URL}`
- TinyStories license: `{LICENSE}`.
- SFT/eval text: fixed assignment-authored prompt-response JSONL examples for template-conditioned short stories and single-label TinyStories-style classification. Arithmetic is not part of the main SFT demo. These examples are committed with the assignment and are not generated at runtime.

## Local Files

- train: `{train_path}`
- valid: `{valid_path}`

The authoritative file counts, token counts, byte counts, SHA256 hashes, source URLs, and split names are in `manifest.json`.
"""
    (dataset_dir / "README.md").write_text(text, encoding="utf-8")


def _download_full_file(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as out:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            out.write(chunk)
    tmp_path.replace(path)


def prepare_student_release_dataset(
    dataset_dir: str | Path,
    train_bytes: int = 536_870_912,
    valid_bytes: int = 33_554_432,
    tokenizer_bytes: int = 134_217_728,
    force: bool = False,
) -> dict:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    cached_entries: dict[str, dict] = {}
    if manifest_path.exists() and not force:
        manifest = load_manifest(dataset_dir)
        cached_entries = {entry["split"]: entry for entry in manifest.get("files", [])}
        result = verify_manifest(dataset_dir)
        if result["ok"] and int(manifest.get("version", 0)) >= MANIFEST_VERSION and (dataset_dir / "sft" / "manifest.json").exists():
            return result

    raw_dir = dataset_dir.parent / "raw" / "tinystories"
    train_path = raw_dir / TRAIN_FILENAME
    valid_path = raw_dir / VALID_FILENAME
    _download_full_file(TINYSTORIES_TRAIN_URL, train_path, force=force)
    _download_full_file(TINYSTORIES_VALID_URL, valid_path, force=force)
    sft_manifest = write_sft_release_data(dataset_dir, seed=SFT_SEED)
    _write_release_readme(dataset_dir, train_path, valid_path)
    manifest = {
        "name": "minillm_tinystories_full_release_v1",
        "source": "Full TinyStories V2 GPT-4 train/valid text for pretraining plus fixed assignment-authored SFT/eval JSONL.",
        "license": LICENSE,
        "version": MANIFEST_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sft_generation": {
            "script": sft_manifest["generation_script"],
            "seed": SFT_SEED,
            "manifest": "sft/manifest.json",
            "overlap_checks": sft_manifest["overlap_checks"],
        },
        "legacy_byte_budgets_ignored_for_full_release": {
            "requested_train_bytes": train_bytes,
            "requested_valid_bytes": valid_bytes,
            "requested_tokenizer_bytes": tokenizer_bytes,
        },
        "tokenizer_training": {
            "split": "pretrain_train",
            "note": "Tokenizer training reads this full train file, usually with tokenizer_train_mb set in the tokenizer config to bound BPE training time.",
        },
        "source_urls": {
            "tinystories_train": TINYSTORIES_TRAIN_URL,
            "tinystories_valid": TINYSTORIES_VALID_URL,
            "cs336_assignment1_reference": "https://github.com/stanford-cs336/spring2024-assignment1-basics",
        },
        "files": [
            _cached_manifest_entry(dataset_dir, cached_entries, train_path, "pretrain_train", "Full TinyStoriesV2-GPT4 train split", "plain_text"),
            _cached_manifest_entry(dataset_dir, cached_entries, valid_path, "pretrain_valid", "Full TinyStoriesV2-GPT4 valid split", "plain_text"),
            _manifest_entry(dataset_dir, "sft/train.jsonl", "sft_train", "assignment-authored fixed template SFT train set", "jsonl"),
            _manifest_entry(dataset_dir, "sft/valid.jsonl", "sft_valid", "assignment-authored fixed template SFT validation set", "jsonl"),
            _manifest_entry(dataset_dir, "sft/eval.jsonl", "sft_eval", "held-out template SFT eval set", "jsonl"),
            _manifest_entry(dataset_dir, "sft/fixed_prompts.jsonl", "before_after_prompts", "held-out fixed template SFT before/after prompts", "jsonl"),
            _manifest_entry(dataset_dir, "eval/sft_eval.jsonl", "sft_eval_alias", "alias of sft/eval.jsonl", "jsonl"),
            _manifest_entry(dataset_dir, "eval/fixed_prompts.jsonl", "before_after_prompts_alias", "alias of sft/fixed_prompts.jsonl", "jsonl"),
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return require_valid_manifest(dataset_dir)
