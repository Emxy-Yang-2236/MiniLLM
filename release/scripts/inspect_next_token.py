#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minillm.model.config import TransformerConfig
from minillm.model.generation import next_token_options
from minillm.model.transformer import TransformerLM
from minillm.tokenizer.bpe import ByteBPETokenizer
from minillm.train.checkpoint import load_checkpoint
from minillm.utils.config import load_yaml
from minillm.utils.device import get_device


def _display_token(text: str) -> str:
    escaped = text.encode("unicode_escape").decode("ascii")
    return escaped or "<empty>"


def _load_checkpoint(path: str | Path, device: torch.device):
    payload = load_checkpoint(path, map_location="cpu")
    tokenizer = ByteBPETokenizer.load(payload["tokenizer_path"])
    model = TransformerLM(TransformerConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), tokenizer


def _endoftext_id(tokenizer) -> int | None:
    special_ids = getattr(tokenizer, "special_token_ids", {})
    token_id = special_ids.get("<|endoftext|>") if isinstance(special_ids, dict) else None
    return None if token_id is None else int(token_id)


def _choose(options: list[dict]) -> dict | None:
    while True:
        raw = input(f"Choose 1-{len(options)}, or q to stop: ").strip().lower()
        if raw in {"q", "quit", "stop"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Invalid choice.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect top-k next-token probabilities from a checkpoint.")
    parser.add_argument("--ckpt", "--checkpoint", dest="ckpt", default=None)
    parser.add_argument("--config", default=None, help="Infer checkpoint_last.pt from this config's run_dir.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mode", choices=["greedy", "choose"], default="greedy")
    args = parser.parse_args()

    ckpt = args.ckpt
    if ckpt is None and args.config is not None:
        ckpt = str(Path(load_yaml(args.config)["run_dir"]) / "checkpoint_last.pt")
    if ckpt is None:
        parser.error("one of --ckpt/--checkpoint or --config is required")

    device = get_device(args.device)
    model, tokenizer = _load_checkpoint(ckpt, device)
    ids = tokenizer.encode(args.prompt)
    if not ids:
        seed_id = _endoftext_id(tokenizer)
        if seed_id is None:
            parser.error("prompt encoded to zero tokens and tokenizer has no <|endoftext|> token")
        ids = [seed_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    eot_id = _endoftext_id(tokenizer)
    stop_ids = set() if eot_id is None else {eot_id}

    print("Prompt:")
    print(args.prompt)
    print("\nProbabilities are full-softmax probabilities over the whole vocabulary.")
    for step in range(1, args.steps + 1):
        options = next_token_options(model, tokenizer, input_ids, top_k=args.top_k, temperature=args.temperature)
        print(f"\nStep {step}")
        print("Current text:")
        print(tokenizer.decode(input_ids[0].tolist(), skip_special=True))
        print(f"\nTop-{len(options)} next tokens:")
        for row in options:
            print(f"[{row['rank']}] {_display_token(row['token_text']):>14}  id={row['token_id']:<6} p={row['prob']:.6f}")

        selected = options[0] if args.mode == "greedy" else _choose(options)
        if selected is None:
            break
        print(f"Selected: {_display_token(selected['token_text'])}")
        next_id = int(selected["token_id"])
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id in stop_ids:
            print("Stopped on <|endoftext|>.")
            break

    print("\nFinal text:")
    print(tokenizer.decode(input_ids[0].tolist(), skip_special=True))


if __name__ == "__main__":
    main()
