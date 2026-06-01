from __future__ import annotations

from adapters import tokenizer_module


def parts(text: str) -> list[str]:
    return tokenizer_module().pretokenize(text, pretokenizer="gpt2_like")


def test_gpt2_like_basic_words_and_punctuation():
    assert parts("Hello world!") == ["Hello", " world", "!"]
    assert parts("punctuation?!") == ["punctuation", "?!"]


def test_gpt2_like_contractions_numbers_and_spaces():
    assert parts("I'm here.") == ["I", "'m", " here", "."]
    assert parts("Numbers: 123 456") == ["Numbers", ":", " 123", " 456"]
    leading = parts("  leading spaces")
    assert "".join(leading) == "  leading spaces"
    assert " leading" in leading


def test_gpt2_like_preserves_newlines_and_unicode():
    newline = parts("hello\nworld")
    assert "".join(newline) == "hello\nworld"
    assert "\n" in newline
    unicode_parts = parts("café 😊")
    assert "".join(unicode_parts) == "café 😊"
    assert unicode_parts[0] == "café"
