from __future__ import annotations

from dataclasses import dataclass

import torch
from .layers import softmax

from einops import rearrange
from minillm.tokenizer.bpe import ByteBPETokenizer

@dataclass
class GenerationConfig:
    """Configuration for autoregressive decoding.

    `temperature <= 0` means greedy decoding. Otherwise, divide logits by
    `temperature`, optionally keep only the `top_k` largest logits, optionally
    apply nucleus sampling with `top_p`, and sample one token from the resulting
    distribution. For top-p, keep the smallest set of highest-probability tokens
    whose cumulative probability is at least `top_p`.
    """

    max_new_tokens: int = 40
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    stop_ids: list[int] | None = None


def default_stop_ids(tokenizer) -> list[int]:
    """Return generation stop ids for the tokenizer.

    MiniLLM's official tokenizer uses `<|endoftext|>` as the only special token.
    Do not invent separate BOS/EOS ids or fall back to byte-token ids.
    """
    token_id = getattr(tokenizer, "special_token_ids", {}).get("<|endoftext|>")
    return [] if token_id is None else [int(token_id)]

@torch.inference_mode()
def generate_ids(model: torch.nn.Module,
                 input_ids,
                 cfg: GenerationConfig):
    """Generate token ids from an already-tokenized prompt.

    Expected behavior:
    - preserve and return the prompt prefix in `input_ids`;
    - call the model on at most the last `model.cfg.context_length` tokens;
    - append one token at a time for up to `cfg.max_new_tokens`;
    - if `cfg.temperature <= 0`, use greedy argmax;
    - otherwise apply temperature, then optional top-k, then optional top-p;
    - stop early when the generated token is in `cfg.stop_ids`;
    - restore the model's training/eval mode before returning.

    Top-p detail: sort probabilities from largest to smallest and keep the
    smallest prefix whose cumulative probability is >= `cfg.top_p`; renormalize
    that kept distribution before sampling.
    """

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(
            input_ids,
            dtype= torch.long,
        )
    if input_ids.ndim == 1:
        input_ids = torch.unsqueeze(input_ids, 0)  # if only contains one batch , set batch dimension

    if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) == 0:
        raise ValueError("Invalid input_ids")

    was_training = model.training
    model.eval()

    result_ids = input_ids.clone()
    stop_ids = set(cfg.stop_ids or [])

    for _ in range (cfg.max_new_tokens):
        context_ids = result_ids[:, -model.cfg.context_length:]
        out = model(context_ids)
        # (batch_size, seq_length, d_vocab) -> (batch_size, d_vocab) (last column left)
        logits = out["logits"][0,-1,:]

        """
        logits = rearrange(
            logits,
            "batch_size 1 vocab_size -> (batch_size 1) vocab_size"
        )
        """
        next_token: torch.Tensor
        if cfg.temperature <= 0 :
            next_token = torch.argmax(
                logits,
                keepdim= True
            )
        else:
            scaled_logits = logits / cfg.temperature
            k = cfg.top_k
            p = cfg.top_p

            filtered_logits = scaled_logits

            if k is not None:
                max_k_values, max_k_indices = torch.topk(
                    scaled_logits,
                    k= k,
                    dim= -1,
                )
                filtered_logits = torch.full_like(
                    scaled_logits,
                    -torch.inf,
                )

                filtered_logits.scatter_(
                    dim= -1,
                    index= max_k_indices,
                    src= max_k_values,
                )

            filtered_logits = topp(
                filtered_logits,
                p,
            )

            probs = softmax(filtered_logits, dim=-1)

            # choose next token according to probs
            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

        # join new_token to result_ids
        next_token_reshape = next_token.reshape(1, 1)
        result_ids = torch.cat(
            [result_ids, next_token_reshape],
            dim=1,
        )

        # stop check
        generated_id = int(next_token.item())
        if generated_id in stop_ids:
            break

    if was_training:
        model.train()

    return result_ids

@torch.inference_mode()
def topp(
        logits: torch.Tensor,
        p: float | None,
) -> torch.Tensor:
    if p is None:
        return logits

    tmp_prob = softmax(logits, dim= -1)
    sorted_p, sorted_indice = torch.sort(
        tmp_prob,
        dim= -1,
        descending= True,
    )
    filter_logits = torch.full_like(
        logits,
        -torch.inf
    )
    tmp_sum: float = 0.0
    tmp_idx = 0
    for val in sorted_p:
        tmp_sum += val
        original_idx = sorted_indice[tmp_idx]
        filter_logits[original_idx] = logits[original_idx]
        tmp_idx += 1
        if tmp_sum >= p:
            break
    return filter_logits

def generate(
    model: torch.nn.Module,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    device=None,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_ids: list[int] | None = None,
    return_full_text: bool = False,
) -> str:
    """Generate text from a raw prompt.

    Tokenize `prompt`, call `generate_ids`, then decode either only the newly
    generated completion or the full prompt+completion depending on
    `return_full_text`. By default, use `<|endoftext|>` as the stop token.
    """
    # tokenizer return list
    prompt_ids = tokenizer.encode(
        prompt,
        add_eos=False,
    )

    prompt_length = len(prompt_ids)

    # empty prompt will cause size mismatch for model
    if prompt_length == 0:
        raise ValueError("prompt must contain at least one token")

    if device is None:
        device = next(model.parameters()).device   # same as model
    else:
        device = torch.device(device)

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    if stop_ids is None:
        gen_stop_ids = default_stop_ids(tokenizer)
    else:
        gen_stop_ids = stop_ids

    # init cfg
    generation_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_ids= gen_stop_ids,
    )

    result_ids = generate_ids(
        model,
        input_ids,
        generation_cfg,
    )

    result_ids = rearrange(
        result_ids,
        "b_s logits -> (b_s logits)"
    )

    if return_full_text:
        ids_to_decode = result_ids
    else:
        ids_to_decode = result_ids[ prompt_length:]

    token_ids = ids_to_decode.detach().cpu().tolist()

    text = tokenizer.decode(
        token_ids,
        skip_special=True,
    )

    return text

def batch_generate(model, tokenizer, prompts: list[str], device=None, **kwargs) -> list[str]:
    """Generate one completion for each prompt using the same decoding options."""
    results: list[str] = []

    for prompt in prompts:
        text = generate(
            model,
            tokenizer,
            prompt,
            device=device,
            **kwargs,
        )
        results.append(text)

    return results

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
