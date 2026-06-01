from __future__ import annotations

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement bias-free Linear without nn.Linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement Embedding without nn.Embedding")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RMSNorm(nn.Module):
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
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement SwiGLU with weights w1, w2, w3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GELUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement GELU MLP baseline")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        raise NotImplementedError("Week 1 TODO: implement CS336-style RoPE")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    raise NotImplementedError("Week 1 TODO: implement numerically stable softmax")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    raise NotImplementedError("Week 1 TODO: implement cross-entropy without F.cross_entropy")
