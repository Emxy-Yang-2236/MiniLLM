from __future__ import annotations

import json

import torch

from adapters import build_model, make_config, sft_dataset, tiny_overfit_batch, train_bpe


def test_sft_batch_loss_is_finite(tmp_path):
    tok = train_bpe(["Return JSON. What is 2 + 5?", '{"answer": 7}'], vocab_size=280)
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps({"prompt": "Return JSON. What is 2 + 5?", "response": '{"answer": 7}'}) + "\n",
        encoding="utf-8",
    )
    item = sft_dataset(path, tok, seq_len=32)[0]
    cfg = make_config(vocab_size=tok.vocab_size, context_length=32)
    model = build_model(cfg)
    out = model(item["input_ids"].unsqueeze(0), item["labels"].unsqueeze(0))
    assert torch.isfinite(out["loss"])


def test_tiny_sft_batch_can_overfit_downward(tmp_path):
    torch.manual_seed(0)
    tok = train_bpe(["Return JSON. What is 2 + 5?", '{"answer": 7}'], vocab_size=280)
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps({"prompt": "Return JSON. What is 2 + 5?", "response": '{"answer": 7}'}) + "\n",
        encoding="utf-8",
    )
    item = sft_dataset(path, tok, seq_len=24)[0]
    batch = {"input_ids": item["input_ids"].unsqueeze(0), "labels": item["labels"].unsqueeze(0)}
    cfg = make_config(vocab_size=tok.vocab_size, context_length=24, d_model=24, d_ff=48, num_heads=4)
    result = tiny_overfit_batch(build_model(cfg), batch, steps=15, lr=3e-3, device="cpu")
    assert result["best_loss"] <= result["initial_loss"]
    assert result["final_loss"] < result["initial_loss"] or result["best_loss"] < result["initial_loss"]
