"""Starter for the Week 1 byte-level BPE tokenizer.

Students implement two core pieces in this file:

1. BPE training: learn `vocab` and `merges` from raw text.
2. Tokenizer runtime: use `vocab` and `merges` to encode/decode text.

The remaining save/load/hash helpers exist so the tokenizer can be used by the
pretraining pipeline and artifact manifests.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


# Official MiniLLM tokenizer settings. Do not add special tokens unless you also change the training/generation pipeline.
SPECIAL_TOKENS = ["<|endoftext|>"]
GPT2_LIKE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKENIZER_VERSION = 2
DEFAULT_PRETOKENIZER = "gpt2_like"
DEFAULT_TIE_BREAK = "max"


def _bytes_to_json(value: bytes) -> list[int]:
    return list(value)


def _bytes_from_json(value: list[int]) -> bytes:
    return bytes(value)


@dataclass
class TokenizerTrainingStats:
    """Metadata written to tokenizer.json and later copied into run manifests.

    These fields are for debugging and reproducibility. They do not affect BPE
    encoding after the tokenizer has been trained.

    Students do not need to modify this.
    """

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
    """Byte-level BPE tokenizer.

    `vocab` maps token id -> raw token bytes.
    `merges` is ordered by creation time.
    `special_tokens` are recognized before normal pre-tokenization.

    You may add private cached properties or helper functions.
    """

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

        """Build the initial vocabulary: specials first, then all 256 bytes."""

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

    # ------------------------------------------------------------------
    # Convenience properties used by generation and reporting scripts.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Student TODO Part 2: encoding and decoding.
    # ------------------------------------------------------------------

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Encode text into token ids.

        Required behavior:
        1. Split special tokens first, using longest-match-first.
        2. GPT-2 pre-tokenize normal text spans.
        3. Convert each pre-token to UTF-8 bytes.
        4. Apply BPE merges within each pre-token only.
        5. Convert the final byte tokens to ids.

        If `add_eos=True`, append the real `<|endoftext|>` id. Do not invent a
        separate EOS token.
        """
        raise NotImplementedError

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        """Yield token ids from chunks without joining the full input string."""
        raise NotImplementedError

    def batch_encode(self, texts: list[str], add_eos: bool = False) -> list[list[int]]:
        return [self.encode(text, add_eos=add_eos) for text in texts]

    def decode(self, ids: list[int] | tuple[int, ...], skip_special: bool = False) -> str:
        """Decode token ids by concatenating token bytes.

        Use `bytes.decode("utf-8", errors="replace")` so malformed byte
        sequences turn into the Unicode replacement character instead of
        raising an exception.
        """
        raise NotImplementedError

    def batch_decode(self, batch_ids: list[list[int]], skip_special: bool = False) -> list[str]:
        return [self.decode(ids, skip_special=skip_special) for ids in batch_ids]

    # ------------------------------------------------------------------
    # Provided pipeline helpers: serialization and reproducibility.
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        """Return tokenizer config and training stats for manifests/reports."""
        longest_id, longest = max(self.vocab.items(), key=lambda item: len(item[1]))
        return {
            "tokenizer_version": TOKENIZER_VERSION,
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "num_special_tokens": len(self.special_tokens),
            "special_tokens": list(self.special_tokens),
            "pretokenizer": self.pretokenizer,
            "tie_break": self.tie_break,
            "min_frequency": self.stats.min_frequency,
            "longest_token_id": longest_id,
            "longest_token_bytes": len(longest),
            "longest_token_repr": longest.decode("utf-8", errors="replace"),
            "stats": asdict(self.stats),
        }

    def stable_hash(self) -> str:
        """Compute a stable sha256 over the tokenizer payload."""
        payload = json.dumps(_tokenizer_json_payload(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        """Save vocab, merges, special tokens, config, and stats as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _tokenizer_json_payload(self)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_manifest(self, path: str | Path) -> None:
        """Write tokenizer_manifest.json next to tokenizer.json."""
        path = Path(path)
        manifest = {
            "tokenizer_path": str(path),
            "tokenizer_sha256": tokenizer_file_sha256(path),
            "tokenizer_version": TOKENIZER_VERSION,
            "pretokenizer": self.pretokenizer,
            "regex_pattern": GPT2_LIKE_PATTERN if self.pretokenizer == "gpt2_like" else "",
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "special_tokens": list(self.special_tokens),
            "tie_break": self.tie_break,
            "training_source_path": self.stats.source_path,
            "training_source_sha256": self.stats.source_sha256,
            "training_bytes": self.stats.input_bytes,
            "training_bytes_read": self.stats.training_bytes_read,
            "elapsed_sec": self.stats.elapsed_sec,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tokenizer_manifest_path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        """Load tokenizer.json so encode/decode behavior is unchanged."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        stats = _training_stats_from_dict(data.get("stats", {}))
        return cls(
            vocab={int(k): _bytes_from_json(v) for k, v in data["vocab"].items()},
            merges=[(_bytes_from_json(left), _bytes_from_json(right)) for left, right in data["merges"]],
            special_tokens=list(data["special_tokens"]),
            pretokenizer=data.get("pretokenizer", stats.pretokenizer),
            tie_break=data.get("tie_break", stats.tie_break),
            stats=stats,
        )


def _tokenizer_json_payload(tokenizer: ByteBPETokenizer) -> dict:
    return {
        "tokenizer_version": TOKENIZER_VERSION,
        "pretokenizer": tokenizer.pretokenizer,
        "tie_break": tokenizer.tie_break,
        "vocab_size": tokenizer.vocab_size,
        "min_frequency": tokenizer.stats.min_frequency,
        "special_tokens": list(tokenizer.special_tokens),
        "vocab": {str(k): _bytes_to_json(v) for k, v in sorted(tokenizer.vocab.items())},
        "merges": [[_bytes_to_json(left), _bytes_to_json(right)] for left, right in tokenizer.merges],
        "stats": asdict(tokenizer.stats),
    }


def _training_stats_from_dict(data: dict) -> TokenizerTrainingStats:
    fields = TokenizerTrainingStats.__dataclass_fields__
    return TokenizerTrainingStats(**{key: value for key, value in data.items() if key in fields})


# ----------------------------------------------------------------------
# File hashing helpers used by tokenizer/data manifests.
# ----------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_file_sha256(path: str | Path) -> str:
    return file_sha256(path)


def tokenizer_manifest_path(path: str | Path) -> Path:
    return Path(path).with_name("tokenizer_manifest.json")


# ----------------------------------------------------------------------
# Student TODO Part 1: BPE training entrypoints.
# ----------------------------------------------------------------------


def pretokenize(text: str, pretokenizer: str = DEFAULT_PRETOKENIZER) -> list[str]:
    """Split text before UTF-8 byte conversion.

    Official runs use `gpt2_like`, implemented with the third-party `regex`
    package and `GPT2_LIKE_PATTERN`.
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
    """Train byte-level BPE from in-memory text.

    Required behavior:
    - validate `vocab_size >= len(special_tokens) + 256`;
    - remove special tokens before normal pre-tokenization;
    - preserve newlines and all Unicode text via UTF-8 bytes;
    - choose the highest-frequency pair each round;
    - break frequency ties by lexicographically greater raw bytes when
      `tie_break="max"`;
    - return a `ByteBPETokenizer`.
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
    num_workers: int = 0,
    num_chunks: int | None = None,
) -> ByteBPETokenizer:
    """Train BPE from a file and save tokenizer.json.

    After training, call `tok.save(output_path)` and
    `tok.save_manifest(output_path)`.

    `num_workers` and `num_chunks` are accepted so this starter matches the
    release scripts and reference implementation. A correct Week 1 solution may
    ignore them and run serially.
    """
    raise NotImplementedError
