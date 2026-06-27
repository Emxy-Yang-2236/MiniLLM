from __future__ import annotations

import torch
import torch.nn as nn


class Linear(nn.Module):
    """bias-free linear transformation.

    Implements y = W x, following the `nn.Linear` interface except there is no bias argument and no bias parameter.

    Args:
        in_features: Final dimension of the input.
        out_features: Final dimension of the output.
        device: Device used to store the parameter.
        dtype: Data type of the parameter.

    Forward:
        x: Tensor with shape `(..., in_features)`.

    Returns:
        Tensor with shape `(..., out_features)`.

    Requirements:
        - Subclass `nn.Module` and call `super().__init__()`.
        - Store the weight as an `nn.Parameter` with shape
          `(out_features, in_features)`.
        - Do not use `nn.Linear` or `nn.functional.linear`.
        - Initialize with `torch.nn.init.trunc_normal_`.
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement bias-free Linear without nn.Linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to `x`."""
        raise NotImplementedError


class Embedding(nn.Module):
    """token embedding lookup.

    Maps integer token IDs to vectors by indexing into an embedding matrix.
    This follows the `nn.Embedding` interface, but students implement the lookup
    directly.

    Args:
        num_embeddings: Vocabulary size.
        embedding_dim: Dimension of each embedding vector, i.e. `d_model`.
        device: Device used to store the parameter.
        dtype: Data type of the parameter.

    Forward:
        token_ids: Integer tensor with shape `(...)`, commonly
            `(batch_size, sequence_length)`.

    Returns:
        Tensor with shape `(..., embedding_dim)`.

    Requirements:
        - Subclass `nn.Module` and call `super().__init__()`.
        - Store the embedding matrix as an `nn.Parameter` with shape
          `(num_embeddings, embedding_dim)`.
        - Do not use `nn.Embedding` or `nn.functional.embedding`.
        - Initialize with `torch.nn.init.trunc_normal_`.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement Embedding without nn.Embedding")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embedding vectors for `token_ids`."""
        raise NotImplementedError


class RMSNorm(nn.Module):
    """root mean square layer normalization.

    RMSNorm rescales each activation vector along the final dimension using its root mean square and a learnable gain parameter.

    Args:
        d_model: Hidden dimension of the model.
        eps: Epsilon value for numerical stability.
        device: Device used to store the gain parameter.
        dtype: Data type of the gain parameter.

    Forward:
        x: Tensor with shape `(batch_size, sequence_length, d_model)` or any shape whose final dimension is `d_model`.

    Returns:
        Tensor with the same shape and original dtype as `x`.

    Requirements:
        - Store the gain parameter as an `nn.Parameter` initialized to ones.
        - Upcast `x` to `torch.float32` before squaring to avoid overflow.
        - Cast the result back to the original input dtype before returning.
        - Do not use `nn.RMSNorm`.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Week 1 TODO: compute RMSNorm in fp32 to avoid overflow when squaring x.
        # The returned tensor must be cast back to in_dtype.
        # result = ...
        # return result.to(in_dtype)
        raise NotImplementedError("Week 1 TODO: implement RMSNorm without nn.RMSNorm")


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Implements the position-wise feed-forward network used inside each
    Transformer block:

        FFN(x) = W2(SiLU(W1 x) * W3 x)

    Args:
        d_model: Hidden dimension of the Transformer block input/output.
        d_ff: Inner feed-forward dimension. CS336 recommends approximately `(8 / 3) * d_model`, rounded to a nearby multiple of 64.
        device: Device used to store parameters.
        dtype: Data type of parameters.

    Forward:
        x: Tensor with shape `(..., d_model)`.

    Returns:
        Tensor with shape `(..., d_model)`.

    Requirements:
        - Implement three bias-free linear projections `w1`, `w2`, and `w3`.
        - Use SiLU/Swish on the gate branch.
        - It is OK to use `torch.sigmoid` when implementing SiLU.
        - Do not replace the module with a ready-made high-level MLP.
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement SwiGLU with weights w1, w2, w3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the position-wise SwiGLU feed-forward network."""
        raise NotImplementedError


class RotaryPositionalEmbedding(nn.Module):
    """rotary positional embedding.

    Applies RoPE to query/key vectors by rotating pairs of hidden dimensions according to explicit token positions. 
    RoPE has no learnable parameters.

    Args:
        theta: RoPE base value `Theta`.
        d_k: Query/key head dimension.
        max_seq_len: Maximum sequence length that will be input.
        device: Device used to store optional precomputed buffers.

    Forward:
        x: Tensor with shape `(..., seq_len, d_k)`. The leading dimensions are arbitrary batch-like dimensions.
        token_positions: Tensor with shape `(..., seq_len)` specifying the token position for each vector in `x`.

    Returns:
        Tensor with the same shape as `x`.

    Requirements:
        - Use `token_positions` to select the correct sin/cos values.
        - If precomputing sin/cos, store them with `register_buffer`, not as learnable parameters.
        - Do not rotate value vectors; attention code should apply RoPE only to queries and keys.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement CS336-style RoPE")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to `x` at the provided `token_positions`."""
        raise NotImplementedError


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """numerically stable softmax.

    Args:
        x: Input tensor.
        dim: Dimension along which to normalize.

    Returns:
        Tensor with the same shape as `x`. Values along `dim` form a normalized
        probability distribution.

    Requirements:
        - Subtract the maximum value along `dim` before exponentiating.
        - Handle arbitrary leading batch-like dimensions.
        - Do not rely on a numerically unstable direct `exp(x) / exp(x).sum()`.

    Adapter/test:
        `adapters.run_softmax`; `pytest -k test_softmax_matches_pytorch`.
    """
    raise NotImplementedError("Week 1 TODO: implement numerically stable softmax")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """next-token cross-entropy loss.

    Args:
        logits: Predicted logits with shape `(..., vocab_size)`.
        targets: Integer target token IDs with shape `(...)`, matching the batch-like dimensions of `logits`.
        ignore_index: Target value that should be excluded from the average loss, used later for response-only SFT labels.

    Returns:
        Scalar tensor: the mean cross-entropy over non-ignored targets.

    Requirements:
        - Subtract the largest logit for numerical stability.
        - Cancel `log` and `exp` where possible, using the log-sum-exp trick.
        - Handle arbitrary leading batch-like dimensions.
        - Do not use `torch.nn.functional.cross_entropy`.
    """
    raise NotImplementedError("Week 2 TODO: implement cross-entropy without F.cross_entropy")
