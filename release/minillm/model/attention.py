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
    """attention(Q, K, V).

    Inputs:
        Q: Query tensor with shape `(..., n_queries, d_k)`.
        K: Key tensor with shape `(..., n_keys, d_k)`.
        V: Value tensor with shape `(..., n_keys, d_v)`.
        mask: Optional boolean tensor broadcastable to `(..., n_queries, n_keys)`.
            `True` means the query may attend to that key;
            `False` means that score should be masked out before softmax.

    Output:
        Tensor with shape `(..., n_queries, d_v)`.

    Requirements:
        - Compute `softmax(Q @ K.T / sqrt(d_k)) @ V`.
        - Apply the mask by setting disallowed pre-softmax scores to `-inf`.
        - Treat all leading dimensions as batch dimensions. This is what lets
          the same function work for both single-head and multi-head attention.

    Public tests call this through `adapters.run_scaled_dot_product_attention`.
    """
    raise NotImplementedError("Week 2 TODO: implement scaled dot-product attention")


def causal_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Provided SDPA backend for faster training and measurement.

    Args:
        q: Query tensor with shape `(batch_size, num_heads, seq_len, head_dim)`.
        k: Key tensor with shape `(batch_size, num_heads, seq_len, head_dim)`.
        v: Value tensor with shape `(batch_size, num_heads, seq_len, head_dim)`.

    Returns:
        Tensor with shape `(batch_size, num_heads, seq_len, head_dim)`.

    This function is not a student TODO. Use it when `backend == "sdpa"`.
    Students implement the naive attention path in `scaled_dot_product_attention`
    and in `MultiHeadSelfAttention.forward`.
    """

    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class MultiHeadSelfAttention(nn.Module):
    """causal multi-head self-attention.

    This module implements the attention sublayer used inside a decoder-only
    Transformer block:

        MHA(x) = W_o MultiHead(W_q x, W_k x, W_v x)

    with a causal mask so token position `i` may attend only to positions
    `j <= i`. RoPE, when enabled, is applied to query and key vectors, not to
    value vectors.
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
        """Construct causal multi-head self-attention.

        Args:
            d_model: Dimensionality of Transformer block inputs and outputs.
            num_heads: Number of attention heads.
                `d_model` must be divisible by `num_heads`;
                each head then has dimension `head_dim = d_model // num_heads`,
                where `d_k = d_v = d_model / num_heads`.
            backend: `"naive"` for the student-implemented attention path, or
                `"sdpa"` for the provided PyTorch scaled-dot-product attention
                helper `causal_sdpa_attention`.
            max_seq_len: Maximum sequence length used to precompute RoPE sin/cos values.
            theta: RoPE base frequency parameter.
            use_rope: Whether to apply RoPE to Q and K before attention.

        Learnable parameters:
            - query projection `W_q`
            - key projection `W_k`
            - value projection `W_v`
            - output projection `W_o`

        Public tests call this through:
            - `adapters.run_multihead_self_attention`
            - `adapters.run_multihead_self_attention_with_rope`
        """
        super().__init__()
        raise NotImplementedError("Week 2 TODO: implement causal multi-head self-attention")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Apply causal multi-head self-attention.

        Args:
            x: Input tensor with shape `(batch_size, seq_len, d_model)`.
            token_positions: Optional tensor of token positions for RoPE. If
                provided, it should identify the absolute positions of the
                `seq_len` tokens. If omitted, use positions `0, ..., seq_len-1`.

        Returns:
            Tensor with shape `(batch_size, seq_len, d_model)`.

        Required behavior:
            - Project `x` into Q, K, and V with three matrix multiplies.
            - Reshape/split the model dimension into `num_heads` independent
              heads of size `head_dim`.
            - Apply RoPE to Q and K only when RoPE is enabled.
            - Use a causal mask so future tokens cannot affect earlier outputs.
            - For `backend == "sdpa"`, call the provided
              `causal_sdpa_attention(q, k, v)`.
              You do not need to implement PyTorch SDPA yourself.
            - Concatenate heads and apply the output projection.
        """
        raise NotImplementedError("Week 2 TODO: implement causal attention; SDPA helper is provided")
