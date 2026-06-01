#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minillm.model.config import TransformerConfig
from minillm.model.generation import generate
from minillm.model.transformer import TransformerLM
from minillm.tokenizer.bpe import ByteBPETokenizer
from minillm.train.checkpoint import load_checkpoint
from minillm.utils.config import load_yaml
from minillm.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "--checkpoint", dest="ckpt", default=None)
    parser.add_argument("--config", default=None, help="Infer checkpoint_last.pt from this config's run_dir.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    ckpt = args.ckpt
    if ckpt is None and args.config is not None:
        ckpt = str(Path(load_yaml(args.config)["run_dir"]) / "checkpoint_last.pt")
    if ckpt is None:
        parser.error("one of --ckpt/--checkpoint or --config is required")
    payload = load_checkpoint(ckpt, map_location="cpu")
    tok = ByteBPETokenizer.load(payload["tokenizer_path"])
    model = TransformerLM(TransformerConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state"])
    device = get_device(args.device)
    model.to(device)
    with torch.inference_mode():
        completion = generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, device=device)
    print("Prompt:")
    print(args.prompt)
    print()
    print("Completion:")
    print(completion)


if __name__ == "__main__":
    main()
