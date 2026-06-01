#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.model.config import TransformerConfig
from minillm.model.generation import batch_generate
from minillm.model.transformer import TransformerLM
from minillm.tokenizer.bpe import ByteBPETokenizer
from minillm.train.checkpoint import load_checkpoint
from minillm.utils.device import get_device


def load_model_and_tokenizer(path: str | Path, device):
    payload = load_checkpoint(path, map_location="cpu")
    model = TransformerLM(TransformerConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    tokenizer = ByteBPETokenizer.load(payload["tokenizer_path"])
    return model, tokenizer


def load_prompts(path: str | Path | None, inline: list[str]) -> list[str]:
    prompts: list[str] = []
    if path is not None:
        prompts.extend(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    prompts.extend(inline)
    if not prompts:
        prompts = [
            "Return JSON only. animal=cat color=blue.",
            "Classify the feeling as one of: happy, sad, scared, angry. Text: Tim lost his toy and cried. Answer:",
            "Write a two sentence story about a tiny robot in a library.",
        ]
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    prompts = load_prompts(args.prompts, args.prompt)
    before_model, before_tok = load_model_and_tokenizer(args.before, device)
    after_model, after_tok = load_model_and_tokenizer(args.after, device)
    before = batch_generate(
        before_model,
        before_tok,
        prompts,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    after = batch_generate(
        after_model,
        after_tok,
        prompts,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    rows = [
        {"prompt": prompt, "before_completion": b, "after_completion": a}
        for prompt, b, a in zip(prompts, before, after)
    ]
    text = json.dumps(rows, indent=2, ensure_ascii=False)
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
