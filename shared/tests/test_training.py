from __future__ import annotations

import json

from adapters import dataset_path_for_split, student_release_dir, train_bpe, train_pretrain, verify_student_release


def test_pretrain_loop_writes_metrics_and_checkpoint(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    train_path = data / "train.txt"
    valid_path = data / "valid.txt"
    text = "\n".join(["Once upon a time a tiny model learned quickly."] * 20)
    train_path.write_text(text, encoding="utf-8")
    valid_path.write_text(text, encoding="utf-8")
    tok = train_bpe(text.splitlines(), vocab_size=280)
    tok_path = data / "tokenizer.json"
    tok.save(tok_path)
    run_dir = tmp_path / "run"
    cfg = {
        "seed": 0,
        "device": "cpu",
        "tokenizer_path": str(tok_path),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "run_dir": str(run_dir),
        "batch_size": 2,
        "max_steps": 2,
        "learning_rate": 1e-3,
        "warmup_steps": 1,
        "eval_interval": 1,
        "model": {
            "vocab_size": tok.vocab_size,
            "num_layers": 1,
            "d_model": 32,
            "d_ff": 64,
            "num_heads": 4,
            "context_length": 16,
            "attention_backend": "naive",
        },
    }
    result = train_pretrain(cfg)
    assert result["step"] == 2
    assert (run_dir / "checkpoint_last.pt").exists()
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows and rows[-1]["step"] == 2


def test_pretraining_smoke_uses_student_release_dataset(tmp_path):
    release = student_release_dir()
    verification = verify_student_release(release)
    assert verification["ok"]
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    release_train = dataset_path_for_split(release, manifest, "pretrain_train")
    release_valid = dataset_path_for_split(release, manifest, "pretrain_valid")
    train_path = tmp_path / "release_train_subset.txt"
    valid_path = tmp_path / "release_valid_subset.txt"
    with release_train.open("r", encoding="utf-8") as f:
        tokenizer_lines = [line.strip() for _, line in zip(range(50), f) if line.strip()]
    train_path.write_text("\n".join(tokenizer_lines[:30]) + "\n", encoding="utf-8")
    with release_valid.open("r", encoding="utf-8") as f:
        valid_lines = [line.strip() for _, line in zip(range(20), f) if line.strip()]
    valid_path.write_text("\n".join(valid_lines or tokenizer_lines[:10]) + "\n", encoding="utf-8")
    tok = train_bpe(tokenizer_lines, vocab_size=300)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)
    run_dir = tmp_path / "release_run"
    cfg = {
        "seed": 0,
        "device": "cpu",
        "tokenizer_path": str(tok_path),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "run_dir": str(run_dir),
        "batch_size": 2,
        "max_steps": 1,
        "learning_rate": 1e-3,
        "warmup_steps": 1,
        "eval_interval": 1,
        "model": {
            "vocab_size": tok.vocab_size,
            "num_layers": 1,
            "d_model": 32,
            "d_ff": 64,
            "num_heads": 4,
            "context_length": 16,
            "attention_backend": "naive",
        },
    }
    result = train_pretrain(cfg)
    assert result["step"] == 1
    assert (run_dir / "checkpoint_last.pt").exists()
