from __future__ import annotations

from adapters import load_tokenizer, tokenizer_module, train_bpe


def test_byte_bpe_roundtrip_and_special_tokens(tmp_path):
    tok = train_bpe(["banana bandana", "banana banana", "你好 banana"], vocab_size=280)
    text = "banana 你好"
    ids = tok.encode(text, add_eos=True)
    assert ids[-1] == tok.endoftext_id
    assert tok.decode(ids, skip_special=True) == text

    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = load_tokenizer(path)
    assert loaded.encode(text, add_eos=True) == ids
    assert loaded.decode(ids, skip_special=True) == text


def test_bpe_training_is_deterministic():
    texts = ["abab abab", "abab abac", "zzzz"]
    a = train_bpe(texts, vocab_size=270)
    b = train_bpe(texts, vocab_size=270)
    assert a.merges == b.merges
    assert a.vocab == b.vocab


def test_tokenizer_batch_empty_unicode_spaces_and_embedded_specials(tmp_path):
    texts = ["", "emoji 😀 café", "two  spaces", "literal <|endoftext|> marker"]
    tok = train_bpe(texts, vocab_size=300, min_frequency=1)
    encoded = tok.batch_encode(texts, add_eos=True)
    plain = tok.batch_encode(texts)
    assert len(encoded) == len(texts)
    assert encoded[0] == [tok.endoftext_id]
    assert tok.batch_decode(plain, skip_special=False) == texts
    assert tok.encode("literal <|endoftext|> marker")[len(tok.encode("literal "))] == tok.endoftext_id

    saved = tmp_path / "tok.json"
    tok.save(saved)
    loaded = load_tokenizer(saved)
    assert loaded.describe()["stats"]["input_texts"] == len(texts)
    assert loaded.batch_decode(loaded.batch_encode(texts), skip_special=False) == texts


def test_bpe_merge_tie_breaks_lexicographically():
    initial_size = tokenizer_module().ByteBPETokenizer.initial().vocab_size
    tok = train_bpe(["za za", "az az"], vocab_size=initial_size + 1, min_frequency=1)
    assert tok.merges[0] == (b"z", b"a")
    assert tok.describe()["num_merges"] == 1
