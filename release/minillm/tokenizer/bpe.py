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
import regex as re

# Official MiniLLM tokenizer settings. Do not add special tokens unless you also change the training/generation pipeline.
SPECIAL_TOKENS = ["<|endoftext|>"]
GPT2_LIKE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKENIZER_VERSION = 2
DEFAULT_PRETOKENIZER = "gpt2_like"
DEFAULT_TIE_BREAK = "max"
SIMPLE_PATTERN = r"\S+|\s+"


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

    # cache some immutable dicts used in fn encode to reduce the cost
    _token_to_id_cache: dict[bytes, int] | None = field(
        default=None, init=False, repr=False, compare=False)
    _merge_rank_cache: dict[tuple[bytes, bytes], int] | None = field(
        default=None, init=False, repr=False, compare=False)
    _special_token_ids_cache: dict[str, int] | None = field(
        default=None, init=False, repr=False, compare=False)

    # lazy update the above dict
    @property
    def _token_to_id_lazy(self) -> dict[bytes, int] | None:
        if self._token_to_id_cache is None or len(self._token_to_id_cache) != len(self.vocab):
            self._token_to_id_cache = {token: idx for idx, token in self.vocab.items()}
        return self._token_to_id_cache

    @property
    def _merge_rank_lazy(self) -> dict[tuple[bytes, bytes], int] | None:
        if self._merge_rank_cache is None or len(self._merge_rank_cache) != len(self.merges):
            self._merge_rank_cache = {pair: rank for rank, pair in enumerate(self.merges)}
        return self._merge_rank_cache

    @property
    def _special_tok_ids_lazy(self) -> dict[str, int] | None :
        if self._special_token_ids_cache is None or len(self._special_token_ids_cache) != len(self.special_tokens) :
            self._special_token_ids_cache = self.special_token_ids
        return self._special_token_ids_cache

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
            raise ValueError("special_tokens must be unique")    # use set to avoid repetition

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
        if "<|endoftext|>" not in self._special_tok_ids_lazy:
            raise ValueError('tokenizer has no real "<|endoftext|>" special token')
        return self._special_tok_ids_lazy["<|endoftext|>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def special_token_ids(self) -> dict[str, int]:
        if self.special_tokens == {} :
            return {}

        token_to_id = {
            value: idx
            for idx, value in self.vocab.items()
        }

        return {
            tok: token_to_id[tok.encode("utf-8")]
            for tok in self.special_tokens
            if tok.encode("utf-8") in token_to_id
            }

    # ------------------------------------------------------------------
    # Week 1 TODO: tokenizer runtime (encoding and decoding).
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
        # pre_store special_token_ids to decrease dict creation cost
        special_tok_ids = self._special_tok_ids_lazy

        # step 1
        split_str: list[str] = []
        if not self.special_tokens:
            split_str = [text]
        else:
            split_str = split_special(self.special_tokens, text)

        #step 2
        pre_tokenized : list[str] = []

        for piece in split_str:
            if piece in special_tok_ids:
                pre_tokenized.append(piece)
                continue

            pre_tokenized_piece = pretokenize(piece, self.pretokenizer)
            pre_tokenized.extend(pre_tokenized_piece)

        # step 3 ~ 5
        encoded : list[int] = []

        # prepare merge rank for step 4
        merge_rank = self._merge_rank_lazy

        # prepare token-id map for step 5
        token_to_id = self._token_to_id_lazy

        for piece in pre_tokenized:
            if piece in special_tok_ids:
                encoded.append(special_tok_ids[piece])
                continue

            #step 3
            piece_bytes = piece.encode("utf-8")
            byte_converted = [
                bytes([byte_value])
                for byte_value in piece_bytes
            ]

            #step 4
            while merge_rank != {} and len(byte_converted) > 1:
                best_pair: tuple[bytes, bytes] | None = None

                for i in range(1,len(byte_converted)):
                    cur_pair = (byte_converted[i-1], byte_converted[i])

                    if (cur_pair in merge_rank and
                        (best_pair is None or
                         merge_rank[best_pair] > merge_rank[cur_pair])):
                        best_pair = cur_pair

                if best_pair is None: break

                new_b_c : list[bytes] = []

                i = 0
                while i < len(byte_converted):
                    cur_pair: tuple[bytes, bytes] | None = None

                    if i < len(byte_converted) - 1:
                        cur_pair = (byte_converted[i], byte_converted[i + 1])

                    if cur_pair is not None and cur_pair == best_pair:
                        merge_piece = byte_converted[i] + byte_converted[i + 1]
                        new_b_c.append(merge_piece)
                        i += 2
                    else:
                        new_b_c.append(byte_converted[i])
                        i += 1

                byte_converted = new_b_c

            # step 5
            for merged_piece in byte_converted:
                encoded.append(token_to_id[merged_piece])

        if add_eos:
            encoded.append(self.endoftext_id)

        return encoded


    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        """Yield token ids from chunks without joining the full input string."""
        for chunk in texts:
            yield from self.encode(chunk)

    def batch_encode(self, texts: list[str], add_eos: bool = False) -> list[list[int]]:
        return [self.encode(text, add_eos=add_eos) for text in texts]

    def decode(self, ids: list[int] | tuple[int, ...], skip_special: bool = False) -> str:
        """Decode token ids by concatenating token bytes.

        Use `bytes.decode("utf-8", errors="replace")` so malformed byte
        sequences turn into the Unicode replacement character instead of
        raising an exception.
        """
        token_bytes : list[bytes] = []

        for token_id in ids:
            if skip_special and token_id in self._special_tok_ids_lazy.values():
                continue

            token_bytes.append(self.vocab[token_id])

        raw_bytes = b"".join(token_bytes)
        raw_tokens = raw_bytes.decode("utf-8", errors= "replace")

        return raw_tokens

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
# Week 1 TODO: BPE training entrypoints.
# ----------------------------------------------------------------------


def pretokenize(text: str, pretokenizer: str = DEFAULT_PRETOKENIZER) -> list[str]:
    """Split text before UTF-8 byte conversion.

    Official runs use `gpt2_like`, implemented with the third-party `regex`
    package and `GPT2_LIKE_PATTERN`.
    """
    if pretokenizer == DEFAULT_PRETOKENIZER :

        gpt2_pattern = re.compile(GPT2_LIKE_PATTERN)

        pre_tokenized: list[str] = pattern_match(text= text, pattern= gpt2_pattern)

        return pre_tokenized

    if pretokenizer == "simple":

        simple_pattern = re.compile(SIMPLE_PATTERN)

        pre_tokenized: list[str] = pattern_match(text= text, pattern= simple_pattern)

        return pre_tokenized

    raise ValueError(f"unknown pretokenizer {pretokenizer!r}")

# match str by regex and split into list[str]
def pattern_match(text: str,
                  pattern: re.Pattern[str]) -> list[str]:
    split_str: list[str] = []
    last_iter = 0

    for match in pattern.finditer(text):
        if last_iter < match.start():
            split_str.append(text[last_iter: match.start()])

        split_str.append(match.group())
        last_iter = match.end()

    if last_iter < len(text):
        split_str.append(text[last_iter:])

    return split_str

def pattern_match_except(text: str,
                  pattern: re.Pattern[str]) -> list[str]:
    split_str: list[str] = []
    last_iter = 0

    for match in pattern.finditer(text):
        if last_iter < match.start():
            split_str.append(text[last_iter: match.start()])

        #split_str.append(match.group())
        last_iter = match.end()

    if last_iter < len(text):
        split_str.append(text[last_iter:])

    return split_str


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

    # init some state info
    start_time = time.perf_counter()
    input_texts : int = 1
    input_bytes : int = 0
    if isinstance(texts, str):
        input_bytes = len(texts.encode("utf-8"))
    else:
        input_texts = len(texts)
        for text in texts:
            input_bytes += len(text.encode("utf-8"))

    # validate vocab_size
    if special_tokens is None :
        special_tokens = SPECIAL_TOKENS
    if vocab_size < len(special_tokens) + 256:
        raise ValueError(f"Invalid vocab list: length < {len(special_tokens)  + 256}")

    split_str: list[str] = []

    # remove special tokens and pretokenize
    if not special_tokens:
        if isinstance(texts, str):
            split_str = [texts]
        else:
            split_str = texts
    else:
        if isinstance(texts, str):
            split_str = split_special_except(special_tokens, texts)
        else:
            for text in texts:
                split_str.extend(split_special_except(special_tokens, text))

    pre_tokenized: list[str] = []
    for text in split_str:
        pre_tokenized.extend(pretokenize(text, pretokenizer))

    # turn ["low" "low" "lower"] into {("low": 2), ("lower": 1)}
    chunk_counter: dict[str, int] = {}
    for chunks in pre_tokenized:
        chunk_counter[chunks] = chunk_counter.get(chunks, 0) + 1

    # turn {("low": 2), ("lower": 1)} into {((b"l",b"o",b"w"): 2), ((...): 1)}
    chunk_counter_bytes: dict[tuple[bytes,...], int] = {}
    for chunks, count in chunk_counter.items():
        chunk_bytes = chunks.encode("utf-8")
        byte_tuple = tuple(bytes([byte_value]) for byte_value in chunk_bytes)
        chunk_counter_bytes[byte_tuple] = count

    # get initial pair count
    pair_count: dict[tuple[bytes,bytes], int] = {}
    for byte_tuple, count in chunk_counter_bytes.items():
        if len(byte_tuple) <= 1: continue

        for i in range (0, len(byte_tuple)-1):
            cur_tuple = (byte_tuple[i],byte_tuple[i + 1])
            pair_count[cur_tuple] = pair_count.get(cur_tuple, 0) + count

    # split chunk_counter_bytes into two list to maintain synchronization during update
    words = list(chunk_counter_bytes.keys())
    frequencies = [
        chunk_counter_bytes[word]
        for word in words
    ]

    # find the positon where each pair appears (in which word)
    pair_to_word_ids : dict[tuple[bytes, bytes], set[int]] = {}

    for word_id, word in enumerate(words):
        word_pairs = single_pair_count(word)

        for pair in word_pairs:
            pair_to_word_ids.setdefault(pair, set()).add(word_id)

    # create tokenizer
    tokenizer = ByteBPETokenizer.initial(
        special_tokens,
        pretokenizer,
        tie_break,
    )

    # init state, part 1
    tokenizer.stats.input_texts = input_texts
    tokenizer.stats.input_bytes = input_bytes
    tokenizer.stats.training_bytes_read = input_bytes
    tokenizer.stats.requested_vocab_size = vocab_size
    tokenizer.stats.min_frequency = min_frequency
    tokenizer.stats.stopped_reason = "vocab_size"  # default
    tokenizer.stats.pretokenizer = pretokenizer
    tokenizer.stats.tie_break = tie_break
    tokenizer.stats.top_pair_count = max(pair_count.values())


    #start training
    while len(tokenizer.vocab) < vocab_size:
        if pair_count == {} :
            tokenizer.stats.stopped_reason = "no_pairs"
            break
        top_count = max(pair_count.values(), default= 0)
        if top_count < min_frequency :
            tokenizer.stats.stopped_reason = "min_frequency"
            break

        top_pairs = [
            pair
            for pair, count in pair_count.items()
            if count == top_count
        ]
        best_pair: tuple[bytes, bytes]
        if tie_break == "max":
            best_pair = max(top_pairs)
        else:
            best_pair = min(top_pairs)

        affected_word_ids = list(
            pair_to_word_ids.get(best_pair,set())
        )
        for word_id in affected_word_ids:
            old_word = words[word_id]
            frequency = frequencies[word_id]
            old_word_count = single_pair_count(old_word)

            # subtract the influence of all pairs, update the word and add new pairs back
            for pair, count in old_word_count.items():
                pair_count[pair] -= count * frequency
                if pair_count[pair] == 0:
                    pair_count.pop(pair)
                pair_to_word_ids[pair].discard(word_id)
                if not pair_to_word_ids[pair] :
                    pair_to_word_ids.pop(pair)

            i = 0
            new_word: list[bytes] = []
            while i < len(old_word) :
                if i < len(old_word) - 1 and (old_word[i], old_word[i+1]) == best_pair:
                    new_word.append(old_word[i] + old_word[i+1])
                    i += 2
                else :
                    new_word.append(old_word[i])
                    i += 1

            new_word_tuple = tuple(byte_value for  byte_value in new_word)
            new_word_count = single_pair_count(new_word_tuple)
            for pair, count in new_word_count.items():
                pair_count[pair] = (
                        pair_count.get(pair, 0)
                        + count * frequency )
                pair_to_word_ids.setdefault(pair, set()).add(word_id)

            words[word_id] = new_word_tuple

        new_token_bytes = best_pair[0] + best_pair[1]
        tokenizer.vocab[len(tokenizer.vocab)] = new_token_bytes
        tokenizer.merges.append(best_pair)

    # update training state, part 2
    longest_id, longest_token = max(
        tokenizer.vocab.items(),
        key=lambda item: len(item[1]),
    )
    tokenizer.stats.longest_token_id = longest_id
    tokenizer.stats.longest_token_bytes = len(longest_token)
    tokenizer.stats.longest_token_repr = longest_token.decode(
        "utf-8",
        errors="replace",
    )

    # update training state, part 3
    tokenizer.stats.final_vocab_size = len(tokenizer.vocab)
    tokenizer.stats.num_merges = len(tokenizer.merges)
    tokenizer.stats.elapsed_sec = (
            time.perf_counter() - start_time
    )

    return tokenizer


def split_special(
        special_tokens: list[str],
        text: str,
) -> list[str]:
    split_str: list[str] = []

    ordered_special_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )
    special_tokens_re_lst = [re.escape(i) for i in ordered_special_tokens]
    special_tokens_re: str = '|'.join(special_tokens_re_lst)

    pattern = re.compile(special_tokens_re)

    split_str = pattern_match(text=text, pattern=pattern)

    return split_str

def split_special_except(
        special_tokens: list[str],
        text: str,
) -> list[str]:
    split_str: list[str] = []

    ordered_special_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )
    special_tokens_re_lst = [re.escape(i) for i in ordered_special_tokens]
    special_tokens_re: str = '|'.join(special_tokens_re_lst)

    pattern = re.compile(special_tokens_re)

    split_str = pattern_match_except(text=text, pattern=pattern)

    return split_str


def single_pair_count(
       byte_tuple : tuple[bytes,...]
) -> dict[tuple[bytes,bytes], int] :
    pair_count: dict[tuple[bytes,bytes], int] = {}

    if len(byte_tuple) > 1 :
        for i in range (0, len(byte_tuple)-1):
            cur_tuple = (byte_tuple[i],byte_tuple[i + 1])
            pair_count[cur_tuple] = pair_count.get(cur_tuple, 0) + 1

    return pair_count

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
    input_path = Path(input_path)
    output_path = Path(output_path)
    raw_bytes: bytes

    with input_path.open("rb") as file:
        if max_bytes is None:
            raw_bytes = file.read()
        else:
            raw_bytes = file.read(max_bytes)

    training_bytes_read = len(raw_bytes)

    text = raw_bytes.decode(
        "utf-8",
        errors="ignore",
    )

    tokenizer = train_bpe(
        text,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        pretokenizer=pretokenizer,
        tie_break=tie_break,
    )

    tokenizer.stats.source_path = str(input_path)
    tokenizer.stats.source_sha256 = file_sha256(input_path)
    tokenizer.stats.training_bytes_read = training_bytes_read

    tokenizer.save(output_path)
    tokenizer._source_path = str(output_path)
    tokenizer.save_manifest(output_path)

    return tokenizer