from __future__ import annotations

import torch
import torch.nn as nn

from .config import TransformerConfig


class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig | int | None = None,
        num_heads: int | None = None,
        d_ff: int | None = None,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        *,
        d_model: int | None = None,
        context_length: int | None = None,
        dropout: float = 0.0,
        attention_backend: str = "naive",
        mlp_type: str = "swiglu",
    ):
        super().__init__()
        raise NotImplementedError("Week 2 TODO: implement CS336-style pre-norm Transformer block")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerLM(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig | None = None,
        *,
        vocab_size: int | None = None,
        context_length: int | None = None,
        d_model: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        d_ff: int | None = None,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        attention_backend: str = "naive",
        tie_embeddings: bool = False,
        mlp_type: str = "swiglu",
    ):
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError("Week 2 TODO: build CS336-style TransformerLM from from-scratch components")

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    raise NotImplementedError("Week 2 TODO: count model parameters")


def model_summary(model: TransformerLM) -> dict:
    raise NotImplementedError("Week 2 TODO: return model summary for reports")
