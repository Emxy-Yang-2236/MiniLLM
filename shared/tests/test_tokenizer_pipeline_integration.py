from __future__ import annotations

import json

import numpy as np
import pytest

from adapters import (
    encode_text_file,
    encoded_manifest_path,
    pretrain_dataset,
    tokenizer_module,
    validate_encoded_manifest,
)


def test_train_bpe_from_file_preserves_newlines_max_bytes_and_stats(tmp_path):
    module = tokenizer_module()
    text_path = tmp_path / "corpus.txt"
    text_path.write_text("hello\nworld\ncafé 😊\n", encoding="utf-8")
    out = tmp_path / "tokenizer.json"
    data = text_path.read_bytes()
    max_bytes = data.index("😊".encode("utf-8")) + 1
    tok = module.train_bpe_from_file(
        text_path,
        vocab_size=300,
        output_path=out,
        min_frequency=1,
        max_bytes=max_bytes,
        pretokenizer="gpt2_like",
    )
    assert out.exists()
    assert tok.decode(tok.encode("hello\nworld"), skip_special=False) == "hello\nworld"
    stats = tok.describe()["stats"]
    assert stats["input_bytes"] <= max_bytes
    assert stats["final_vocab_size"] == tok.vocab_size
    assert stats["num_merges"] == len(tok.merges)
    assert stats["source_sha256"]


def test_encode_bin_manifest_and_dataset_guard(tmp_path):
    module = tokenizer_module()
    text_path = tmp_path / "train.txt"
    text_path.write_text("Once upon a time\nA tiny story\n", encoding="utf-8")
    tok_path = tmp_path / "tokenizer.json"
    tok = module.train_bpe_from_file(text_path, vocab_size=300, output_path=tok_path, min_frequency=1)
    bin_path = tmp_path / "tokens.bin"
    count = encode_text_file(text_path, tok, bin_path)
    manifest_path = encoded_manifest_path(bin_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert count == manifest["token_count"]
    assert manifest["tokenizer_sha256"] == module.tokenizer_file_sha256(tok_path)
    assert manifest["pretokenizer"] == "gpt2_like"
    assert manifest["dtype"] == "uint16"
    ids = np.memmap(bin_path, dtype=np.dtype(manifest["dtype"]), mode="r")
    assert int(ids.max()) < tok.vocab_size
    validate_encoded_manifest(bin_path, tok_path)
    item = pretrain_dataset(bin_path, tokenizer=None, seq_len=8)[0]
    assert item["input_ids"].shape == (8,)

    other_path = tmp_path / "other_tokenizer.json"
    module.train_bpe_from_file(text_path, vocab_size=301, output_path=other_path, min_frequency=1)
    with pytest.raises(RuntimeError, match="tokenizer hash mismatch"):
        validate_encoded_manifest(bin_path, other_path)
