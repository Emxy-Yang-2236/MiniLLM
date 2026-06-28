from __future__ import annotations

import torch
import torch.nn as nn

from .config import TransformerConfig


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block.

    Structure:
        `x = x + MHA(RMSNorm(x))`
        `x = x + SwiGLU(RMSNorm(x))`
    """

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
        attention_backend: str = "naive",
    ):
        """Accept either `TransformerConfig` or scalar dimensions for tests."""
        super().__init__()
        raise NotImplementedError("Week 2 TODO: implement CS336-style pre-norm Transformer block")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block on hidden states.

        Args:
            x: Tensor with shape `(batch_size, seq_len, d_model)`.

        Returns:
            Tensor with the same shape `(batch_size, seq_len, d_model)`.
        """
        raise NotImplementedError


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

    `input_ids -> embeddings -> blocks -> final RMSNorm -> LM head`.
    Return `{"logits": logits}` and add `loss` when `labels` are provided.
    The dataset/collator prepares next-token labels; do not shift labels again.
    """

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
        attention_backend: str = "naive",
        tie_embeddings: bool = False,
    ):
        """Accept `TransformerConfig` or scalar dimensions; most students can ignore `attention_backend`."""
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError("Week 2 TODO: build TransformerLM from from-scratch components")

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return logits with shape `(batch_size, seq_len, vocab_size)` and optional loss."""
        raise NotImplementedError


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Provided helper: count scalar parameters for reports."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def model_summary(model: TransformerLM) -> dict:
    """Provided helper: summarize model config and parameter count for reports."""
    cfg = model.cfg
    return {
        "parameters": count_parameters(model),
        "vocab_size": cfg.vocab_size,
        "context_length": cfg.context_length,
        "d_model": cfg.d_model,
        "d_ff": cfg.d_ff,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "attention_backend": cfg.attention_backend,
        "tie_embeddings": cfg.tie_embeddings,
    }
