from __future__ import annotations

import torch
import torch.nn as nn

from .config import TransformerConfig
from .attention import MultiHeadSelfAttention
from .layers import RMSNorm, SwiGLU, Embedding, Linear, cross_entropy


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

        # case 1: TransformerConfig is provided
        if isinstance(cfg, TransformerConfig):
            d_model = cfg.d_model
            num_heads = cfg.num_heads
            d_ff = cfg.d_ff
            max_seq_len = cfg.context_length
            theta = cfg.rope_theta
            attention_backend = cfg.attention_backend
        # case 2: scalar dimensions
        elif isinstance(cfg, int):
            if d_model is not None:
                raise ValueError("d_model was provided twice")
            d_model = cfg

        # check if param not provided (for case 2 but also check case 1)
        if d_model is None:
            raise ValueError("d_model must be provided")
        if num_heads is None:
            raise ValueError("num_heads must be provided")
        if d_ff is None:
            raise ValueError("d_ff must be provided")
        if max_seq_len is None:
            raise ValueError("max_seq_len or context_length must be provided")

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            backend=attention_backend,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=True,
        )
        # two layer norm
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block on hidden states.

        Args:
            x: Tensor with shape `(batch_size, seq_len, d_model)`.

        Returns:
            Tensor with the same shape `(batch_size, seq_len, d_model)`.
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

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

        # init cfg
        if cfg is None:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided")
            if context_length is None:
                raise ValueError("context_length must be provided")
            if d_model is None:
                raise ValueError("d_model must be provided")
            if num_layers is None:
                raise ValueError("num_layers must be provided")
            if num_heads is None:
                raise ValueError("num_heads must be provided")
            if d_ff is None:
                raise ValueError("d_ff must be provided")

            cfg = TransformerConfig(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                attention_backend=attention_backend,
                tie_embeddings=tie_embeddings,
            )

        self.cfg = cfg

        # init embedding
        self.token_embeddings = Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.d_model,
        )
        # stack num_layers layer of transformer_block
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )

        self.lm_head = Linear(
            in_features=cfg.d_model,
            out_features=cfg.vocab_size,
        )
        self.ln_final = RMSNorm(cfg.d_model)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return logits with shape `(batch_size, seq_len, vocab_size)` and optional loss."""
        # Embedding
        x = self.token_embeddings(input_ids)

        # transformer_layer
        for layer in self.layers:
            x = layer(x)

        # final norm
        x = self.ln_final(x)
        logits = self.lm_head(x)  # d_model -> d_vocab

        out = {"logits": logits}

        if labels is not None:
            out["loss"] = cross_entropy(
                logits,
                labels,
                ignore_index=-100,
            )

        return out

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
