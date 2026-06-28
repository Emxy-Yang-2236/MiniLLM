from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Shapes: `Q (..., n_queries, d_k)`, `K (..., n_keys, d_k)`,
    `V (..., n_keys, d_v) -> (..., n_queries, d_v)`.
    If `mask` is provided, `False` entries are blocked before softmax.
    """
    raise NotImplementedError("Week 2 TODO: implement scaled dot-product attention")


def causal_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Provided causal PyTorch SDPA backend. Not a student TODO."""

    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Input/output shape: `(batch_size, seq_len, d_model)`.
    Use Q/K/V projections, split into heads, apply causal attention, then project
    back to `d_model`. Apply RoPE only to Q and K.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        backend: str = "naive",
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        use_rope: bool = True,
    ):
        """Use `head_dim = d_model // num_heads`; `backend="sdpa"` is provided."""
        super().__init__()
        raise NotImplementedError("Week 2 TODO: implement causal multi-head self-attention")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Apply causal MHA. If `token_positions` is omitted, use `0..seq_len-1`."""
        raise NotImplementedError("Week 2 TODO: implement causal attention; SDPA helper is provided")
