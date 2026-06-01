from __future__ import annotations

import pytest

from adapters import get_tokenizer, tokenizer_module, train_bpe


def test_vocab_size_validation_and_initial_vocab():
    module = tokenizer_module()
    with pytest.raises(ValueError):
        train_bpe("hello", vocab_size=128)
    tok = module.ByteBPETokenizer.initial(pretokenizer="gpt2_like")
    assert tok.vocab_size == len(tok.special_tokens) + 256
    assert tok.special_token_ids["<|endoftext|>"] == 0


def test_training_merges_stats_and_min_frequency_stop():
    tok = train_bpe("banana banana banana", vocab_size=275, min_frequency=2, pretokenizer="gpt2_like")
    assert tok.merges
    assert tok.describe()["num_merges"] == tok.vocab_size - tok.describe()["stats"]["initial_vocab_size"]
    stopped = train_bpe("abcd", vocab_size=300, min_frequency=2, special_tokens=[], pretokenizer="gpt2_like")
    assert stopped.merges == []
    assert stopped.describe()["stats"]["stopped_reason"] == "min_frequency"


def test_deterministic_tie_breaking_uses_cs336_lexicographic_max_pair():
    a = train_bpe("za za az az", vocab_size=257, min_frequency=1, special_tokens=[], pretokenizer="simple")
    b = train_bpe("za za az az", vocab_size=257, min_frequency=1, special_tokens=[], pretokenizer="simple")
    assert a.merges == b.merges
    assert a.merges[0] == (b"z", b"a")


def test_training_compresses_repeated_corpus_better_than_byte_vocab():
    module = tokenizer_module()
    base = module.ByteBPETokenizer.initial(pretokenizer="gpt2_like")
    trained = train_bpe("banana banana banana banana", vocab_size=280, min_frequency=2, pretokenizer="gpt2_like")
    text = "banana banana"
    assert len(trained.encode(text)) < len(base.encode(text))


def test_merge_application_is_greedy_by_merge_rank():
    vocab = {0: b"a", 1: b"b", 2: b"c", 3: b"ab", 4: b"bc", 5: b"abc"}
    tok = get_tokenizer(vocab, merges=[(b"a", b"b"), (b"ab", b"c"), (b"b", b"c")], special_tokens=[])
    assert tok.encode("abc") == [5]
