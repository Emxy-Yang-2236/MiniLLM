from __future__ import annotations

import json

import torch

from adapters import sft_collator, sft_dataset, train_bpe


def test_sft_dataset_masks_prompt_and_keeps_response_loss(tmp_path):
    tok = train_bpe(["Return JSON. What is 1 + 2?", '{"answer": 3}'], vocab_size=280)
    path = tmp_path / "sft.jsonl"
    row = {"prompt": "Return JSON. What is 1 + 2?", "response": '{"answer": 3}'}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    item = sft_dataset(path, tok, seq_len=32)[0]
    labels = item["labels"].tolist()
    assert -100 in labels
    assert any(label != -100 for label in labels)
    first_supervised = next(i for i, label in enumerate(labels) if label != -100)
    assert all(label == -100 for label in labels[:first_supervised])


def test_sft_dataset_truncates_prompt_but_preserves_response_label(tmp_path):
    prompt = " ".join(["prompt"] * 80)
    response = "ok"
    tok = train_bpe([prompt, response], vocab_size=280)
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"prompt": prompt, "response": response}) + "\n", encoding="utf-8")
    item = sft_dataset(path, tok, seq_len=16)[0]
    labels = item["labels"].tolist()
    assert item["input_ids"].shape == (16,)
    assert any(label != -100 for label in labels)


def test_sft_collator_pads_to_longest_row_and_masks_labels():
    collate = sft_collator(pad_id=0)
    batch = collate(
        [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 5, 6])},
            {"input_ids": torch.tensor([1, 4]), "labels": torch.tensor([-100, 7])},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3], [1, 4, 0]]
    assert batch["labels"].tolist() == [[-100, 5, 6], [-100, 7, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
