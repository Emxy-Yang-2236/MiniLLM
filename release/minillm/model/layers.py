from __future__ import annotations

from webbrowser import register

import torch
import torch.nn as nn
from einops import rearrange, reduce, pack
from einops import einsum


class Linear(nn.Module):
    """Bias-free linear layer.

    Forward shape: `(..., in_features) -> (..., out_features)`.
    Store weight as `(out_features, in_features)`.
    Do not use `nn.Linear` or `nn.functional.linear`.
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        std = (2.0 / (in_features + out_features)) ** 0.5

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device= device,
                dtype= dtype,
            )
        )

        torch.nn.init.trunc_normal_(
            self.weight,
            mean= 0.0,
            std= std,
            a= -3.0 * std,
            b=  3.0 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to `x`."""
        # x in row vector form
        return x @ self.weight.T


class Embedding(nn.Module):
    """Token embedding lookup.

    Forward shape: token ids `(...) -> (..., embedding_dim)`.
    Store the embedding matrix as `(num_embeddings, embedding_dim)`.
    Do not use `nn.Embedding` or `nn.functional.embedding`.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device= device,
                dtype= dtype,
            )
        )

        torch.nn.init.trunc_normal_(
            self.weight,
            mean= 0.0,
            std= 1.0,
            a= -3.0,
            b=  3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embedding vectors for `token_ids`."""
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square normalization.

    Forward shape: `(..., d_model) -> (..., d_model)`.
    Upcast `x` to fp32 before squaring, then cast back to the input dtype.
    Do not use `nn.RMSNorm`.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square = reduce(
            x.pow(2),
            "... d_model -> ... 1",
            "mean",
        )

        rms_norm_res = x * torch.rsqrt(
            mean_square + self.eps
        )

        # multiply learnable g_i
        final_res = (rms_norm_res * self.weight.to(torch.float32)).to(in_dtype)
        return final_res

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Formula: `W2(SiLU(W1 x) * W3 x)`.
    Forward shape: `(..., d_model) -> (..., d_model)`.
    Use three bias-free projections; do not replace this with a ready-made MLP.
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        # using linear layer to init w1~3
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the position-wise SwiGLU feed-forward network."""

        val_silu = torch.nn.functional.silu(
            self.w1(x)
        )
        val_glu = val_silu * self.w3(x)
        val_ffn = self.w2(val_glu)

        return val_ffn

class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding.

    Forward shape: `x (..., seq_len, d_k)` with `token_positions (..., seq_len)`.
    Return the same shape as `x`. Store sin/cos caches as buffers, not parameters.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # theta_i,r = i * Theta(param) ^ (-2r/d_k)   (r from 0 to d_k/2 - 1)
        trav = torch.arange(
            0, d_k, 2,
            dtype= torch.float32,
            device= device,
            )
        inv_freq = theta ** (-trav / d_k)

        positions = torch.arange(
            max_seq_len,
            dtype= torch.float32,
            device= device,
        )

        angles = einsum(
            positions,
            inv_freq,
            "pos, freq -> pos freq",
        )

        #store cos and sin cache
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        self.register_buffer(
            "cos_cache",
            cos_cache,
            persistent=False,
        )

        self.register_buffer(
            "sin_cache",
            sin_cache,
            persistent=False,
        )


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to `x` at the provided `token_positions`."""
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # x may have head dimension (... seq d_k) -> (... head seq d_k)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # x'_even = x_even * cos - x_odd * sin
        # x'_odd = x_odd * cos + x_even * sin
        x_even, x_odd = rearrange(
            x,
            "... (pair component) -> ... pair component",
            component = 2,
        ).unbind(dim= -1)

        x_prime_even = x_even * cos - x_odd * sin
        x_prime_odd = x_odd * cos + x_even * sin
        x_prime = rearrange(
            torch.stack(
                (x_prime_even, x_prime_odd),
                dim=-1,
            ),
            "... pair component -> ... (pair component)",
            component = 2,
        )

        return x_prime

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.

    Normalize along `dim` and return the same shape as `x`.
    Subtract the max before exponentiating.
    """
    max_value = x.max(dim=dim, keepdim=True).values
    x_prime = x - max_value

    exp_x_prime = torch.exp(x_prime)
    sum_exp = exp_x_prime.sum(dim = dim, keepdim= True)

    return exp_x_prime / sum_exp

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Next-token cross-entropy loss.

    `logits`: `(..., vocab_size)`, `targets`: `(...)`.
    Return mean loss over targets not equal to `ignore_index`.
    Do not use `torch.nn.functional.cross_entropy`.
    """

    flat_logits, _ = pack(
        [logits],
        "* vocab",
    )
    flat_targets, _ = pack(
        [targets],
        "*",
    )

    # handle ignore_idx
    valid_mask = flat_targets != ignore_index
    valid_logits = flat_logits[valid_mask]
    valid_targets = flat_targets[valid_mask]

    log_normalizer = torch.logsumexp(
        valid_logits,
        dim=-1,
    )

    target_indices = rearrange(
        valid_targets,
        "token -> token 1",
    )

    # apply targets to valid_logits
    target_logits = valid_logits.gather(
        dim=-1,
        index=target_indices,
    )
    # delete last dim -> val for calculating CE
    target_logits = rearrange(
        target_logits,
        "token 1 -> token",
    )

    per_token_loss = log_normalizer - target_logits
    loss = reduce(
        per_token_loss,
        "token ->",
        "mean",
    )

    return loss


