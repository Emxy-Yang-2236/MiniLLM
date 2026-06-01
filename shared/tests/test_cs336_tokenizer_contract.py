from __future__ import annotations

from adapters import get_tokenizer, load_tokenizer, run_train_bpe, train_bpe


def test_cs336_train_bpe_is_deterministic_and_orders_merges(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("za za\naz az\n", encoding="utf-8")
    vocab_a, merges_a = run_train_bpe(corpus, vocab_size=262, special_tokens=[], min_frequency=1)
    vocab_b, merges_b = run_train_bpe(corpus, vocab_size=262, special_tokens=[], min_frequency=1)
    assert vocab_a == vocab_b
    assert merges_a == merges_b
    assert merges_a[0] == (b"z", b"a")


def test_cs336_get_tokenizer_encode_decode_special_tokens_and_roundtrip():
    vocab = {0: b"a", 1: b"b", 2: b"ab", 3: b" ", 4: b"<|endoftext|>"}
    tok = get_tokenizer(vocab, merges=[(b"a", b"b")], special_tokens=["<|endoftext|>"])
    ids = tok.encode("ab <|endoftext|>")
    assert ids == [2, 3, 4]
    assert tok.decode(ids, skip_special=False) == "ab <|endoftext|>"
    assert tok.decode(ids, skip_special=True) == "ab "


def test_cs336_tokenizer_contract_unicode_repeated_spaces_iterable_and_save_load(tmp_path):
    texts = ["hello  world", "unicode café 😀", "literal <|endoftext|> stays special"]
    tok = train_bpe(texts, vocab_size=310, min_frequency=1)
    for text in texts:
        assert tok.decode(tok.encode(text), skip_special=False) == text

    chunks = ["hello  ", "world", " café"]
    iterable_ids = list(tok.encode_iterable(chunks))
    assert tok.decode(iterable_ids, skip_special=False) == "".join(chunks)

    special_ids = tok.encode("literal <|endoftext|> stays special")
    assert tok.endoftext_id in special_ids

    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = load_tokenizer(path)
    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges
    assert loaded.decode(loaded.encode("unicode café 😀"), skip_special=False) == "unicode café 😀"
