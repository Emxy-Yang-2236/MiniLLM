from __future__ import annotations

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    raise NotImplementedError("Week 2 TODO: implement scaled dot-product attention")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        backend: str = "naive",
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        use_rope: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("Week 2 TODO: implement causal multi-head self-attention")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("Week 2 TODO: run causal attention with naive or SDPA backend")


class CausalSelfAttention(MultiHeadSelfAttention):
    pass
