#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.eval.sft import evaluate_sft
from minillm.model.config import TransformerConfig
from minillm.model.transformer import TransformerLM
from minillm.tokenizer.bpe import ByteBPETokenizer
from minillm.train.checkpoint import load_checkpoint
from minillm.utils.config import load_yaml
from minillm.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "--checkpoint", dest="ckpt", default=None)
    parser.add_argument("--config", default=None, help="Infer checkpoint_last.pt and eval_path from this config.")
    parser.add_argument("--eval_path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    parser.add_argument("--include_outputs", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config) if args.config is not None else {}
    ckpt = args.ckpt or (str(Path(cfg["run_dir"]) / "checkpoint_last.pt") if cfg else None)
    eval_path = args.eval_path or cfg.get("eval_path")
    if eval_path is None:
        parser.error("one of --eval_path or --config with eval_path is required")
    if ckpt is None:
        parser.error("one of --ckpt/--checkpoint or --config is required")
    payload = load_checkpoint(ckpt, map_location="cpu")
    tok = ByteBPETokenizer.load(payload["tokenizer_path"])
    model = TransformerLM(TransformerConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state"])
    device = get_device(args.device)
    model.to(device)
    scores = evaluate_sft(model, tok, eval_path, device=device)
    result = scores if args.include_outputs else {k: v for k, v in scores.items() if k != "outputs"}
    text = json.dumps(result, indent=2)
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
