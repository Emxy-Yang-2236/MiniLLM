from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minillm.data.sft_dataset import SFTCollator, SFTDataset, tokenizer_padding_id
from minillm.eval.sft import evaluate_sft
from minillm.model.config import TransformerConfig
from minillm.model.generation import batch_generate
from minillm.model.transformer import TransformerLM
from minillm.tokenizer.bpe import ByteBPETokenizer, tokenizer_file_sha256
from minillm.train.checkpoint import load_checkpoint, save_checkpoint
from minillm.train.optim import clip_grad_norm_
from minillm.train.seed import set_seed
from minillm.train.state import JsonlLogger, TrainState, apply_lr, autocast_context, create_optimizer, create_scheduler
from minillm.utils.device import get_device


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _eval_sft_loss(model, loader, device, max_batches: int = 4, amp: bool = False, amp_dtype: str = "bf16") -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            with autocast_context(device, amp, amp_dtype):
                loss = model(batch["input_ids"].to(device), batch["labels"].to(device))["loss"]
            losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / max(1, len(losses))


def train_sft(cfg: dict, base_ckpt: str, max_steps: int | None = None) -> dict:
    """Run the provided SFT training loop from a student's pretrained checkpoint.

    Students are responsible for the SFT dataset/masking code. This function is
    intentionally provided as course infrastructure so Week 3 focuses on the
    core SFT idea: supervised response-only loss from the student's own model.
    """
    if not Path(base_ckpt).exists():
        raise FileNotFoundError(f"missing base checkpoint {base_ckpt}; run train_pretrain.py first")

    set_seed(int(cfg.get("seed", 0)))
    out_dir = Path(cfg["run_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_checkpoint(base_ckpt, map_location="cpu")
    tokenizer = ByteBPETokenizer.load(payload["tokenizer_path"])
    if payload.get("tokenizer_sha256") and payload.get("tokenizer_sha256") != tokenizer_file_sha256(payload["tokenizer_path"]):
        raise RuntimeError("base checkpoint tokenizer_sha256 does not match tokenizer file")

    model_cfg = TransformerConfig.from_dict(payload["config"])
    device = get_device(cfg.get("device", "auto"))
    model = TransformerLM(model_cfg).to(device)
    model.load_state_dict(payload["model_state"])

    train_ds = SFTDataset(cfg["train_path"], tokenizer, model_cfg.context_length)
    valid_ds = SFTDataset(cfg["valid_path"], tokenizer, model_cfg.context_length)
    collator = SFTCollator(tokenizer_padding_id(tokenizer))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, collate_fn=collator)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], collate_fn=collator)

    steps = max_steps or int(cfg["max_steps"])
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(cfg, steps)
    state = TrainState()
    if cfg.get("resume_from"):
        load_checkpoint(cfg["resume_from"], model=model, optimizer=optimizer, scheduler=scheduler, train_state=state, map_location=device)

    logger = JsonlLogger(out_dir / "metrics.jsonl")
    amp = bool(cfg.get("amp", False))
    amp_dtype = cfg.get("amp_dtype", "bf16")
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    checkpoint_interval = int(cfg.get("checkpoint_interval", 0) or 0)
    early_stop = bool(cfg.get("early_stop_on_valid", False))
    patience = int(cfg.get("patience", 0) or 0)
    best_valid = float("inf")
    best_step = 0
    bad_evals = 0

    sample_prompts = cfg.get("sample_prompts") or [
        cfg.get(
            "sample_prompt",
            "Choose one feeling label: happy, sad, scared, angry, kind, lonely.\n"
            "Text: Tim lost his toy and cried.\nAnswer:",
        )
    ]
    before = batch_generate(model, tokenizer, sample_prompts, temperature=0.0, device=device)

    start = time.perf_counter()
    model.train()
    train_iter = _cycle(train_loader)
    while state.step < steps:
        lr = scheduler.get_lr(state.step)
        apply_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        tokens_this_step = 0

        for _ in range(grad_accum):
            batch = next(train_iter)
            tokens_this_step += int(batch["input_ids"].numel())
            with autocast_context(device, amp, amp_dtype):
                loss = model(batch["input_ids"].to(device), batch["labels"].to(device))["loss"] / grad_accum
            loss.backward()
            total_loss += float(loss.item())

        grad_norm = clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        optimizer.step()
        scheduler.step()
        state.step += 1
        state.tokens_seen += tokens_this_step

        if state.step % cfg.get("eval_interval", 25) == 0 or state.step == steps:
            val = _eval_sft_loss(model, valid_loader, device, amp=amp, amp_dtype=amp_dtype)
            is_best = val < best_valid
            if is_best:
                best_valid = val
                best_step = state.step
                bad_evals = 0
            else:
                bad_evals += 1
            logger.log(
                {
                    "step": state.step,
                    "train_loss": total_loss,
                    "valid_loss": val,
                    "lr": lr,
                    "grad_norm": float(grad_norm.item()),
                    "tokens": state.tokens_seen,
                    "elapsed_sec": time.perf_counter() - start,
                }
            )
            if is_best:
                save_checkpoint(
                    out_dir / "checkpoint_best.pt",
                    model,
                    optimizer,
                    state.step,
                    model_cfg.to_dict(),
                    payload["tokenizer_path"],
                    scheduler=scheduler,
                    train_state=state,
                    tokenizer_metadata=payload.get("tokenizer_metadata", {}),
                )
            if early_stop and patience > 0 and bad_evals >= patience:
                break

        if checkpoint_interval and (state.step % checkpoint_interval == 0 or state.step == steps):
            save_checkpoint(
                out_dir / "checkpoint_last.pt",
                model,
                optimizer,
                state.step,
                model_cfg.to_dict(),
                payload["tokenizer_path"],
                scheduler=scheduler,
                train_state=state,
                tokenizer_metadata=payload.get("tokenizer_metadata", {}),
            )

    save_checkpoint(
        out_dir / "checkpoint_last.pt",
        model,
        optimizer,
        state.step,
        model_cfg.to_dict(),
        payload["tokenizer_path"],
        scheduler=scheduler,
        train_state=state,
        tokenizer_metadata=payload.get("tokenizer_metadata", {}),
    )

    after = batch_generate(model, tokenizer, sample_prompts, temperature=0.0, device=device)
    samples = "\n\n".join(f"PROMPT\n{p}\nBEFORE\n{b}\nAFTER\n{a}" for p, b, a in zip(sample_prompts, before, after))
    (out_dir / "samples.txt").write_text(samples + "\n", encoding="utf-8")

    if cfg.get("eval_path") and cfg.get("write_eval", True):
        best_path = out_dir / "checkpoint_best.pt"
        eval_model = model
        if best_path.exists():
            best_payload = load_checkpoint(best_path, map_location="cpu")
            eval_model = TransformerLM(model_cfg).to(device)
            eval_model.load_state_dict(best_payload["model_state"])
            eval_model.eval()
        scores = evaluate_sft(eval_model, tokenizer, cfg["eval_path"], device=device)
        (out_dir / "eval.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

    return {
        "run_dir": str(out_dir),
        "step": state.step,
        "best_step": best_step,
        "best_valid_loss": best_valid if best_valid < float("inf") else None,
        "best_checkpoint": str(out_dir / "checkpoint_best.pt") if (out_dir / "checkpoint_best.pt").exists() else str(out_dir / "checkpoint_last.pt"),
        "before": before,
        "after": after,
    }
