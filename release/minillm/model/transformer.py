from __future__ import annotations

import torch
import torch.nn as nn

from .config import TransformerConfig


class TransformerBlock(nn.Module):
    """pre-norm Transformer block.

    A decoder-only Transformer block has two residual sublayers:

        x = x + MultiHeadSelfAttention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    The attention sublayer must be causal, so earlier token positions cannot
    attend to later token positions.
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
        """Construct a Transformer block.

        Args:
            cfg: Optional `TransformerConfig`. If provided, read model sizes and attention options from it. 
            The adapter tests may also construct this class by passing the scalar arguments below.

            d_model: Dimensionality of the block input and output.
            num_heads: Number of heads in causal multi-head self-attention.
            d_ff: Inner dimensionality of the position-wise SwiGLU feed-forward network.
            max_seq_len/context_length: Maximum sequence length used by RoPE.
            theta: RoPE base frequency parameter.
            attention_backend: Most students can ignore this argument; it is used by configs and benchmarking.
        """
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
    """decoder-only Transformer language model.

    Core model structure:

        input_ids
          -> token embeddings
          -> `num_layers` pre-norm Transformer blocks
          -> final RMSNorm
          -> LM head
          -> logits with shape `(batch_size, seq_len, vocab_size)`

    Training interface:

        out = {"logits": logits}
        if labels is not None:
            out["loss"] = cross_entropy(logits, labels, ignore_index=-100)

    The dataset prepares labels for next-token prediction. This module only
    computes logits and, when labels are provided, the optional CE loss.
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
        """Construct a Transformer language model.

        Args:
            cfg: Optional `TransformerConfig`. If omitted, build a config from the scalar arguments below.
            vocab_size: Vocabulary size, which determines the token embedding and LM-head output dimensions.
            context_length: Maximum context length, also used for RoPE buffers. 
                            context_length should be passed down as max_seq_len
            d_model: Hidden size of embeddings and Transformer blocks.
            num_layers: Number of Transformer blocks.
            num_heads: Number of attention heads per block.
            d_ff: Inner size of the SwiGLU feed-forward network.
            rope_theta: RoPE base frequency parameter.
            attention_backend: Most students can ignore this argument; it is used by configs and benchmarking.
            tie_embeddings: Whether the LM head reuses the token embedding weight matrix.
        """
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError("Week 2 TODO: build TransformerLM from from-scratch components")

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Run the language model.

        Args:
            input_ids: Integer token ids with shape `(batch_size, seq_len)`.
            labels: Optional integer token ids with shape `(batch_size, seq_len)`.
                Label entries equal to `-100` should be ignored by the loss,
                matching PyTorch language-modeling convention.

        Returns:
            A dictionary containing:
                `logits`: Tensor with shape `(batch_size, seq_len, vocab_size)`.
                `loss`: Scalar tensor, only when `labels` is provided.
        """
        raise NotImplementedError


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Provided helper: count parameters for reports and sanity checks.

    Args:
        model: Any PyTorch module.
        trainable_only: If true, count only parameters with `requires_grad=True`; otherwise count all parameters.

    Returns:
        Total number of scalar parameters matching the filter.
    """
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def model_summary(model: TransformerLM) -> dict:
    """Provided helper: return a small model summary used by pipeline reports.

    Expected fields include the model config, total/trainable parameter counts,
    and whether embeddings are tied. Do not include removed architecture knobs
    such as `mlp_type`; this project uses SwiGLU.
    """
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
