from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from minillm.tokenizer.bpe import file_sha256, tokenizer_file_sha256


def get_batch(dataset, batch_size: int, context_length: int, device):
    """next-token batch sampling.

    Given a one-dimensional token sequence, return `(x, y)` tensors on `device`
    with shape `(batch_size, context_length)`, where `y` is `x` shifted one
    token to the right.
    """
    raise NotImplementedError("Week 2 TODO: implement get_batch")


def encoded_manifest_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix:
        return path.with_suffix(".manifest.json")
    return path.with_name(path.name + ".manifest.json")


def load_encoded_manifest(path: str | Path) -> dict:
    return json.loads(encoded_manifest_path(path).read_text(encoding="utf-8"))


def validate_encoded_manifest(path: str | Path, tokenizer_path: str | Path) -> dict:
    manifest_path = encoded_manifest_path(path)
    if not manifest_path.exists():
        raise RuntimeError(f"missing encoded dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = tokenizer_file_sha256(tokenizer_path)
    expected = manifest.get("tokenizer_sha256")
    if expected != actual:
        raise RuntimeError(
            f"tokenizer hash mismatch for {path}: manifest has {expected}, loaded tokenizer has {actual}. "
            "Regenerate tokenizer.json and encoded .bin files together."
        )
    if int(manifest.get("tokenizer_version", -1)) < 2:
        raise RuntimeError(f"unsupported tokenizer spec version in {manifest_path}: {manifest.get('tokenizer_version')}")
    if manifest.get("dtype") not in {"uint16", "uint32"}:
        raise RuntimeError(f"unsupported encoded dtype in {manifest_path}: {manifest.get('dtype')}")
    return manifest


def _manifest_dtype(path: Path) -> np.dtype:
    manifest_path = encoded_manifest_path(path)
    if not manifest_path.exists():
        return np.dtype(np.uint32)
    dtype = json.loads(manifest_path.read_text(encoding="utf-8")).get("dtype", "uint32")
    if dtype == "uint16":
        return np.dtype(np.uint16)
    if dtype == "uint32":
        return np.dtype(np.uint32)
    raise RuntimeError(f"unsupported encoded dtype {dtype!r} in {manifest_path}")


class PretrainDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer=None, seq_len: int = 128):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.path = Path(path)
        self.seq_len = seq_len
        self.pad_id = tokenizer.special_token_ids.get("<|endoftext|>", 0) if tokenizer is not None else 0
        if self.path.suffix == ".bin":
            self.ids = np.memmap(self.path, dtype=_manifest_dtype(self.path), mode="r")
        else:
            if tokenizer is None:
                raise ValueError("tokenizer is required for text datasets")
            text = self.path.read_text(encoding="utf-8")
            ids = tokenizer.encode(text, add_eos=True)
            if len(ids) < seq_len + 2:
                ids = ids * ((seq_len + 2) // max(1, len(ids)) + 1)
            self.ids = np.asarray(ids, dtype=np.uint32)

    def __len__(self) -> int:
        return max(1, len(self.ids) - self.seq_len)

    def __getitem__(self, idx: int):
        idx = idx % len(self)
        chunk = np.asarray(self.ids[idx : idx + self.seq_len + 1], dtype=np.int64)
        if len(chunk) < self.seq_len + 1:
            chunk = np.pad(chunk, (0, self.seq_len + 1 - len(chunk)), constant_values=self.pad_id)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}


class RandomBlockDataset(Dataset):
    def __init__(self, path: str | Path, seq_len: int, num_samples: int, seed: int = 0):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        path = Path(path)
        self.ids = np.memmap(path, dtype=_manifest_dtype(path), mode="r")
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed
        if len(self.ids) < seq_len + 2:
            raise ValueError("token file is too small for requested sequence length")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        idx = int(idx) % self.num_samples
        rng = np.random.default_rng(self.seed + idx)
        start = int(rng.integers(0, len(self.ids) - self.seq_len - 1))
        chunk = np.asarray(self.ids[start : start + self.seq_len + 1], dtype=np.int64)
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long),
        }


def encode_text_file(input_path: str | Path, tokenizer, output_path: str | Path, add_eos: bool = True) -> int:
    return encode_text_file_with_manifest(input_path, tokenizer, output_path, add_eos=add_eos)["token_count"]


def encode_text_file_with_manifest(
    input_path: str | Path,
    tokenizer,
    output_path: str | Path,
    add_eos: bool = True,
    tokenizer_path: str | Path | None = None,
    max_bytes: int | None = None,
    encode_command: str = "",
) -> dict:
    start = time.perf_counter()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    bytes_read = 0
    dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32
    dtype_name = "uint16" if dtype == np.uint16 else "uint32"
    with input_path.open("r", encoding="utf-8") as src, output_path.open("wb") as dst:
        for line in src:
            encoded = line.encode("utf-8")
            if max_bytes is not None and bytes_read + len(encoded) > int(max_bytes):
                keep = int(max_bytes) - bytes_read
                if keep <= 0:
                    break
                encoded = encoded[:keep]
                while encoded:
                    try:
                        line = encoded.decode("utf-8")
                        break
                    except UnicodeDecodeError as exc:
                        encoded = encoded[: exc.start]
                else:
                    break
            bytes_read += len(encoded)
            arr = np.asarray(tokenizer.encode(line, add_eos=add_eos), dtype=dtype)
            arr.tofile(dst)
            total += int(arr.size)
            if max_bytes is not None and bytes_read >= int(max_bytes):
                break
    tok_path_text = str(tokenizer_path or getattr(tokenizer, "_source_path", "") or "")
    tok_path = Path(tok_path_text) if tok_path_text else None
    tok_sha = tokenizer_file_sha256(tok_path) if tok_path is not None and tok_path.is_file() else tokenizer.stable_hash()
    manifest = {
        "path": str(output_path),
        "bin_sha256": file_sha256(output_path),
        "dtype": dtype_name,
        "token_count": total,
        "bytes": output_path.stat().st_size,
        "source_text_path": str(input_path),
        "source_text_sha256": file_sha256(input_path),
        "source_bytes_read": bytes_read,
        "tokenizer_path": str(tok_path) if tok_path is not None else "",
        "tokenizer_sha256": tok_sha,
        "vocab_size": tokenizer.vocab_size,
        "num_merges": len(tokenizer.merges),
        "special_tokens": list(tokenizer.special_tokens),
        "pretokenizer": tokenizer.pretokenizer,
        "tokenizer_version": tokenizer.describe()["tokenizer_version"],
        "add_eos": add_eos,
        "elapsed_sec": time.perf_counter() - start,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "encode_command": encode_command,
    }
    encoded_manifest_path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def split_text_file(input_path: str | Path, train_path: str | Path, valid_path: str | Path, valid_fraction: float = 0.1) -> dict:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    text = Path(input_path).read_text(encoding="utf-8")
    split = int(len(text) * (1.0 - valid_fraction))
    train_path = Path(train_path)
    valid_path = Path(valid_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(text[:split], encoding="utf-8")
    valid_path.write_text(text[split:], encoding="utf-8")
    return {"train_chars": split, "valid_chars": len(text) - split}
