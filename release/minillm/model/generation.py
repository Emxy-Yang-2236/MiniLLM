from __future__ import annotations

from dataclasses import dataclass


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
