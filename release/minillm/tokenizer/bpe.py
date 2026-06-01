from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<|endoftext|>"]
GPT2_LIKE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKENIZER_VERSION = 2
DEFAULT_PRETOKENIZER = "gpt2_like"
DEFAULT_TIE_BREAK = "max"


@dataclass
class TokenizerTrainingStats:
    input_texts: int = 0
    input_bytes: int = 0
    training_bytes_read: int = 0
    source_path: str = ""
    source_sha256: str = ""
    initial_vocab_size: int = 0
    final_vocab_size: int = 0
    requested_vocab_size: int = 0
    num_merges: int = 0
    min_frequency: int = 2
    stopped_reason: str = ""
    longest_token_id: int = -1
    longest_token_bytes: int = 0
    longest_token_repr: str = ""
    top_pair_count: int = 0
    pretokenizer: str = DEFAULT_PRETOKENIZER
    tie_break: str = DEFAULT_TIE_BREAK
    tokenizer_version: int = TOKENIZER_VERSION
    elapsed_sec: float = 0.0


@dataclass
class ByteBPETokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]
    pretokenizer: str = DEFAULT_PRETOKENIZER
    tie_break: str = DEFAULT_TIE_BREAK
    stats: TokenizerTrainingStats = field(default_factory=TokenizerTrainingStats)

    @classmethod
    def initial(
        cls,
        special_tokens: list[str] | None = None,
        pretokenizer: str = DEFAULT_PRETOKENIZER,
        tie_break: str = DEFAULT_TIE_BREAK,
    ) -> "ByteBPETokenizer":
        """Build the fixed initial vocabulary: specials first, then all 256 byte tokens.
        """
        if pretokenizer not in {"simple", "gpt2_like"}:
            raise ValueError(f"unknown pretokenizer {pretokenizer!r}")
        if tie_break not in {"min", "max"}:
            raise ValueError(f"unknown tie_break {tie_break!r}")
        specials = list(SPECIAL_TOKENS if special_tokens is None else special_tokens)
        if len(set(specials)) != len(specials):
            raise ValueError("special_tokens must be unique")
        vocab: dict[int, bytes] = {i: tok.encode("utf-8") for i, tok in enumerate(specials)}
        offset = len(vocab)
        for i in range(256):
            vocab[offset + i] = bytes([i])
        stats = TokenizerTrainingStats(
            initial_vocab_size=len(vocab),
            final_vocab_size=len(vocab),
            pretokenizer=pretokenizer,
            tie_break=tie_break,
        )
        return cls(vocab=vocab, merges=[], special_tokens=specials, pretokenizer=pretokenizer, tie_break=tie_break, stats=stats)

    @property
    def endoftext_id(self) -> int:
        if "<|endoftext|>" not in self.special_token_ids:
            raise ValueError('tokenizer has no real "<|endoftext|>" special token')
        return self.special_token_ids["<|endoftext|>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def special_token_ids(self) -> dict[str, int]:
        token_to_id = {value: idx for idx, value in self.vocab.items()}
        return {tok: token_to_id[tok.encode("utf-8")] for tok in self.special_tokens if tok.encode("utf-8") in token_to_id}

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Week 1 TODO: split specials, pretokenize spans, apply BPE ranks, and return token ids.

        Official MiniLLM tokenizers only use <|endoftext|> as a special token.
        If add_eos=True, append <|endoftext|>; do not invent a separate
        stop token.
        """
        raise NotImplementedError

    def encode_iterable(self, texts: Iterable[str]):
        """Week 1 TODO: stream token ids from text chunks without joining everything in memory."""
        raise NotImplementedError

    def batch_encode(self, texts: list[str], add_eos: bool = False) -> list[list[int]]:
        return [self.encode(text, add_eos=add_eos) for text in texts]

    def decode(self, ids: list[int] | tuple[int, ...], skip_special: bool = False) -> str:
        """Week 1 TODO: concatenate token bytes and decode UTF-8; optionally skip specials."""
        raise NotImplementedError

    def batch_decode(self, batch_ids: list[list[int]], skip_special: bool = False) -> list[str]:
        return [self.decode(ids, skip_special=skip_special) for ids in batch_ids]

    def describe(self) -> dict:
        """Week 1 TODO: return tokenizer config and training stats for debugging."""
        raise NotImplementedError

    def stable_hash(self) -> str:
        """Week 1 TODO: compute a stable sha256 over the saved tokenizer payload."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Week 1 TODO: save vocab, merges, special tokens, pretokenizer, tie-break rule, and stats as JSON."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        """Week 1 TODO: load the JSON format exactly enough that encode output is unchanged."""
        raise NotImplementedError


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_file_sha256(path: str | Path) -> str:
    return file_sha256(path)


def pretokenize(text: str, pretokenizer: str = DEFAULT_PRETOKENIZER) -> list[str]:
    """Week 1 TODO: implement 'simple' and GPT-2-like regex pretokenization.

    GPT-2-like mode should use GPT2_LIKE_PATTERN and preserve leading spaces,
    contractions, punctuation, numbers, unicode, whitespace runs, and newlines.
    """
    raise NotImplementedError


def train_bpe(
    texts: list[str] | str,
    vocab_size: int,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
    pretokenizer: str = DEFAULT_PRETOKENIZER,
    tie_break: str = DEFAULT_TIE_BREAK,
) -> ByteBPETokenizer:
    """Week 1 TODO: train byte-level BPE from text.

    Required behavior: GPT-2-like pretokenization, special-token boundaries,
    deterministic CS336-compatible merge order, and no newline loss. You may
    create any private helper functions you want; their names are not part of
    the public API.
    """
    raise NotImplementedError


def train_bpe_from_file(
    input_path: str | Path,
    vocab_size: int,
    output_path: str | Path,
    special_tokens: list[str] | None = None,
    min_frequency: int = 2,
    max_bytes: int | None = None,
    pretokenizer: str = DEFAULT_PRETOKENIZER,
    tie_break: str = DEFAULT_TIE_BREAK,
) -> ByteBPETokenizer:
    """Week 1 TODO: read a UTF-8-safe prefix, train BPE preserving newlines, save tokenizer.json."""
    raise NotImplementedError
