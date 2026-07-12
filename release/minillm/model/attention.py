from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from einops import rearrange, reduce
from .layers import softmax, Linear, RotaryPositionalEmbedding


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

    # calculate Q * K_T
    scores = einsum(
        Q, K,
        "... query d_k, ... key d_k -> ... query key",
    )
    d_k = Q.shape[-1]
    scaled_scores = scores / (d_k ** 0.5)

    # apply mask matrix : masked_fill pos where mask[pos] = False to -inf
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    attention_weights = softmax(
        scaled_scores,
        dim=-1,
    )

    # (d_queries, d_keys) @ (d_keys, d_v)
    return attention_weights @ V



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

        # save basic model config
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.backend = backend
        self.use_rope = use_rope

        # init attention matrix
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        # handle RoPE use
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
            )
        else:
            self.rope = None


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Apply causal MHA. If `token_positions` is omitted, use `0..seq_len-1`."""

        # apply q,k,v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # rearrange
        q = rearrange(
            q,
            "batch seq (num_heads d_head) -> batch num_heads seq d_head",
            num_heads = self.num_heads,
        )
        k = rearrange(
            k,
            "batch seq (num_heads d_head) -> batch num_heads seq d_head",
            num_heads=self.num_heads,
        )
        v = rearrange(
            v,
            "batch seq (num_heads d_head) -> batch num_heads seq d_head",
            num_heads=self.num_heads,
        )

        # apply RoPE
        if token_positions is None:
            token_positions = torch.arange(
                x.shape[-2],     # seq_length
                device=x.device,
            ) # (0,...,seq_length-1) if not provided

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # causal mask
        seq_len = x.shape[-2]
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )

        # apply attention
        if self.backend == "naive":
            attention_output = scaled_dot_product_attention(q,k,v, mask= causal_mask)
        else:
            attention_output = causal_sdpa_attention(q,k,v)

        attention_output = rearrange(
            attention_output,
            "batch num_heads seq d_head -> batch seq (num_heads d_head)",
        )
        
        return self.output_proj(attention_output)