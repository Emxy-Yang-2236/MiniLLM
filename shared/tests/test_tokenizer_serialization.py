from __future__ import annotations

from adapters import load_tokenizer, tokenizer_module, train_bpe


def test_special_tokens_roundtrip_skip_and_longest_match():
    tok = train_bpe(
        "hello <|endoftext|> world",
        vocab_size=280,
        min_frequency=1,
        special_tokens=["<|endoftext|>"],
    )
    text = "hello <|endoftext|> world"
    ids = tok.encode(text)
    assert tok.endoftext_id in ids
    assert tok.decode(ids, skip_special=False) == text
    assert tok.decode(ids, skip_special=True) == "hello  world"
    assert all(b"<|endoftext|>" not in left + right for left, right in tok.merges)

    module = tokenizer_module()
    overlap = module.ByteBPETokenizer.initial(special_tokens=["<x>", "<x><y>"], pretokenizer="gpt2_like")
    assert overlap.encode("<x><y>") == [1]


def test_ascii_unicode_spaces_newlines_empty_batch_and_iterable_roundtrip():
    texts = ["", "ASCII text", "café 😊", "two  spaces", "hello\nworld"]
    tok = train_bpe("\n".join(texts), vocab_size=320, min_frequency=1, pretokenizer="gpt2_like")
    for text in texts:
        assert tok.decode(tok.encode(text), skip_special=False) == text
    encoded = tok.batch_encode(texts)
    assert tok.batch_decode(encoded, skip_special=False) == texts
    iterable_ids = list(tok.encode_iterable(["hello\n", "world"]))
    assert tok.decode(iterable_ids, skip_special=False) == "hello\nworld"


def test_save_load_preserves_vocab_merges_config_and_encode(tmp_path):
    tok = train_bpe("hello hello\ncafé 😊\n", vocab_size=300, min_frequency=1, pretokenizer="gpt2_like")
    before = tok.encode("hello café 😊")
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = load_tokenizer(path)
    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.pretokenizer == "gpt2_like"
    assert loaded.tie_break == "max"
    assert loaded.encode("hello café 😊") == before
    assert path.read_text(encoding="utf-8").count('"pretokenizer"') >= 1
    assert tokenizer_module().tokenizer_file_sha256(path)
