from __future__ import annotations

import numpy as np
import pytest
import torch

from adapters import encode_text_file, pretrain_dataset, random_block_dataset, split_text_file, train_bpe


def test_encode_text_file_and_memmap_dataset(tmp_path):
    text_path = tmp_path / "train.txt"
    text_path.write_text("Once upon a time\nA tiny story\n", encoding="utf-8")
    tok = train_bpe(text_path.read_text().splitlines(), vocab_size=280)
    bin_path = tmp_path / "tokens.bin"
    count = encode_text_file(text_path, tok, bin_path)
    assert count > 0
    ds = pretrain_dataset(bin_path, tokenizer=None, seq_len=8)
    item = ds[0]
    assert item["input_ids"].shape == (8,)
    assert item["labels"].shape == (8,)


def test_random_block_dataset_is_deterministic_by_index(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(64, dtype=np.uint32).tofile(path)
    a = random_block_dataset(path, seq_len=6, num_samples=4, seed=123)
    b = random_block_dataset(path, seq_len=6, num_samples=4, seed=123)
    assert len(a) == 4
    assert torch.equal(a[2]["input_ids"], b[2]["input_ids"])
    assert torch.equal(a[2]["labels"], a[2]["input_ids"] + 1)


def test_split_text_file_validates_fraction(tmp_path):
    src = tmp_path / "all.txt"
    train = tmp_path / "train.txt"
    valid = tmp_path / "valid.txt"
    src.write_text("abcdefghij", encoding="utf-8")
    stats = split_text_file(src, train, valid, valid_fraction=0.3)
    assert stats == {"train_chars": 7, "valid_chars": 3}
    assert train.read_text(encoding="utf-8") == "abcdefg"
    assert valid.read_text(encoding="utf-8") == "hij"
    with pytest.raises(ValueError):
        split_text_file(src, train, valid, valid_fraction=0.0)
