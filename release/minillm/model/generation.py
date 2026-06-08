from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GenerationConfig:
    max_new_tokens: int = 40
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    stop_ids: list[int] | None = None


def default_stop_ids(tokenizer) -> list[int]:
    raise NotImplementedError("Week 2 TODO: return the real <|endoftext|> stop id without fallback values")


def generate_ids(model, input_ids, cfg: GenerationConfig):
    raise NotImplementedError("Week 2 TODO: implement greedy/top-k/top-p autoregressive token generation")


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    device=None,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_ids: list[int] | None = None,
    return_full_text: bool = False,
) -> str:
    raise NotImplementedError("Week 2 TODO: tokenize a prompt, call generate_ids, and decode completion text")


def batch_generate(model, tokenizer, prompts: list[str], device=None, **kwargs) -> list[str]:
    raise NotImplementedError("Week 2 TODO: generate samples for multiple prompts")


@torch.inference_mode()
def next_token_options(model, tokenizer, input_ids, top_k: int = 10, temperature: float = 1.0) -> list[dict]:
    """Return top-k next-token probabilities for a single prompt prefix.

    This helper is provided for the optional inspection script. Students still
    implement the model forward pass and tokenizer decode path it relies on.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must have shape [seq_len] or [1, seq_len]")
    if input_ids.size(1) == 0:
        raise ValueError("input_ids must contain at least one token")
    top_k = max(1, int(top_k))

    was_training = model.training
    model.eval()
    x = input_ids.to(next(model.parameters()).device)
    logits = model(x[:, -model.cfg.context_length :])["logits"][:, -1, :]
    top_k = min(top_k, logits.size(-1))
    scaled = logits if temperature <= 0 else logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
    if was_training:
        model.train()

    rows = []
    for rank, (prob, token_id) in enumerate(zip(top_probs[0], top_ids[0]), start=1):
        tid = int(token_id.item())
        rows.append(
            {
                "rank": rank,
                "token_id": tid,
                "token_text": tokenizer.decode([tid], skip_special=False),
                "prob": float(prob.item()),
            }
        )
    return rows
