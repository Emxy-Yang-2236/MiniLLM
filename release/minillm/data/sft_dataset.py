from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def tokenizer_padding_id(tokenizer) -> int:
    """Use <|endoftext|> as the padding value; labels/attention_mask hide pads."""
    return tokenizer.special_token_ids.get("<|endoftext|>", 0)


def load_sft_rows(path: str | Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
            raise ValueError(f"SFT row {line_no} must contain string prompt and response fields")
        rows.append(row)
    return rows


def format_prompt_response(row: dict, tokenizer, add_eos: bool = True) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.encode(row["prompt"])
    response_ids = tokenizer.encode(row["response"])
    if add_eos:
        stop_id = getattr(tokenizer, "special_token_ids", {}).get("<|endoftext|>")
        if stop_id is None:
            raise ValueError("add_eos=True requires <|endoftext|> in tokenizer special tokens")
        response_ids = response_ids + [stop_id]
    return prompt_ids, response_ids


def truncate_prompt_response(prompt_ids: list[int], response_ids: list[int], seq_len: int) -> tuple[list[int], list[int]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    total = seq_len + 1
    if not response_ids:
        return prompt_ids[:total], []
    prompt_budget = min(len(prompt_ids), max(1, total - 1))
    prompt_ids = prompt_ids[:prompt_budget]
    response_budget = max(0, total - len(prompt_ids))
    return prompt_ids, response_ids[:response_budget]


def make_sft_tensors(row: dict, tokenizer, seq_len: int) -> dict:
    """Week 3 TODO: build input_ids, response-only labels, and attention_mask.

    Use `format_prompt_response` and `truncate_prompt_response`.
    Then create next-token-prediction pairs from the concatenated sequence:
    `input_ids = seq[:-1]`, `labels = seq[1:]`.

    The important SFT rule is response-only loss:
    labels that predict prompt tokens must be `-100`, and labels that
    predict response tokens should keep their token ids. Pad `input_ids`
    to `seq_len` with `tokenizer_padding_id(tokenizer)`, pad labels with
    `-100`, and set attention_mask to 1 for real tokens and 0 for padding.
    """
    raise NotImplementedError("Week 3 TODO: implement response-only SFT tensors")


class SFTDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, seq_len: int):
        self.rows = load_sft_rows(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return make_sft_tensors(self.rows[idx], self.tokenizer, self.seq_len)


class SFTCollator:
    def __init__(self, pad_id: int, label_pad_id: int = -100):
        self.pad_id = pad_id
        self.label_pad_id = label_pad_id

    def __call__(self, rows: list[dict]) -> dict:
        max_len = max(row["input_ids"].numel() for row in rows)
        input_ids = []
        labels = []
        attention_mask = []
        for row in rows:
            x = row["input_ids"]
            y = row["labels"]
            mask = row.get("attention_mask")
            if mask is None:
                mask = torch.ones(x.numel(), dtype=torch.long)
            pad = max_len - x.numel()
            input_ids.append(torch.cat([x, torch.full((pad,), self.pad_id, dtype=torch.long)]))
            labels.append(torch.cat([y, torch.full((pad,), self.label_pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([mask.to(dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
            "prompt": [row.get("prompt", "") for row in rows],
            "response": [row.get("response", "") for row in rows],
        }
