from __future__ import annotations

import torch
import torch.nn as nn


class Linear(nn.Module):
    """Bias-free linear layer.

    Forward shape: `(..., in_features) -> (..., out_features)`.
    Store weight as `(out_features, in_features)`.
    Do not use `nn.Linear` or `nn.functional.linear`.
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement bias-free Linear without nn.Linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to `x`."""
        raise NotImplementedError("Week 1 TODO: implement Linear.forward")


class Embedding(nn.Module):
    """Token embedding lookup.

    Forward shape: token ids `(...) -> (..., embedding_dim)`.
    Store the embedding matrix as `(num_embeddings, embedding_dim)`.
    Do not use `nn.Embedding` or `nn.functional.embedding`.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement Embedding without nn.Embedding")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embedding vectors for `token_ids`."""
        raise NotImplementedError("Week 1 TODO: implement Embedding.forward")


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Forward shape: `(..., d_model) -> (..., d_model)`.
    Upcast `x` to fp32 before squaring, then cast back to the input dtype.
    Do not use `nn.RMSNorm`.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        raise NotImplementedError("Week 1 TODO: implement RMSNorm without nn.RMSNorm")


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Formula: `W2(SiLU(W1 x) * W3 x)`.
    Forward shape: `(..., d_model) -> (..., d_model)`.
    Use three bias-free projections; do not replace this with a ready-made MLP.
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement SwiGLU with weights w1, w2, w3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the position-wise SwiGLU feed-forward network."""
        raise NotImplementedError("Week 1 TODO: implement SwiGLU.forward")


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding.

    Forward shape: `x (..., seq_len, d_k)` with `token_positions (..., seq_len)`.
    Return the same shape as `x`. Store sin/cos caches as buffers, not parameters.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement RoPE")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to `x` at the provided `token_positions`."""
        raise NotImplementedError("Week 1 TODO: implement RoPE forward pass")


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.

    Normalize along `dim` and return the same shape as `x`.
    Subtract the max before exponentiating.
    """
    raise NotImplementedError("Week 1 TODO: implement numerically stable softmax")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Next-token cross-entropy loss.

    `logits`: `(..., vocab_size)`, `targets`: `(...)`.
    Return mean loss over targets not equal to `ignore_index`.
    Do not use `torch.nn.functional.cross_entropy`.
    """
    raise NotImplementedError("Week 2 TODO: implement cross-entropy without F.cross_entropy")
