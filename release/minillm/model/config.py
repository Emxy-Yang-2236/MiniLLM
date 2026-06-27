from __future__ import annotations

from dataclasses import asdict, dataclass


def _default_d_ff(d_model: int) -> int:
    raw = int(8 * d_model / 3)
    return max(64, ((raw + 63) // 64) * 64)


@dataclass
class TransformerConfig:
    vocab_size: int = 512
    context_length: int = 128
    d_model: int = 128
    d_ff: int | None = None
    num_layers: int = 2
    num_heads: int = 4
    rope_theta: float = 10000.0
    attention_backend: str = "naive"
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.context_length <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("model sizes must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.attention_backend not in {"naive", "sdpa"}:
            raise ValueError("attention_backend must be 'naive' or 'sdpa'")
        if self.d_ff is None:
            self.d_ff = _default_d_ff(self.d_model)
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TransformerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
