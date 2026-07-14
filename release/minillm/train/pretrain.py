from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minillm.data.pretrain_dataset import PretrainDataset, RandomBlockDataset, encoded_manifest_path, validate_encoded_manifest
from minillm.model.config import TransformerConfig
from minillm.model.generation import generate
from minillm.model.transformer import TransformerLM, model_summary
from minillm.tokenizer.bpe import ByteBPETokenizer
from minillm.train.checkpoint import load_checkpoint, save_checkpoint
from minillm.train.optim import clip_grad_norm_
from minillm.train.schedules import WarmupScheduler
from minillm.train.seed import set_seed
from minillm.train.state import JsonlLogger, TrainState, apply_lr, create_optimizer, create_scheduler, autocast_context
from minillm.utils.device import get_device


def evaluate_loss(
        model: torch.nn.Module,
        loader: DataLoader,
        device,
        max_batches: int = 10,
        amp: bool = False,
        amp_dtype: str = "bf16") -> float:
    """Week 3 TODO: compute mean validation loss.

    This is the evaluation-side version of the basic training loop:
    switch to eval mode, disable gradients, run up to `max_batches`, and return the average `out["loss"]`.
    """

    was_training = model.training
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            # move two tenser in batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast_context(device, amp, amp_dtype):
                # logits and loss
                out = model(input_ids, labels)
                loss = out["loss"]

                # initial loss is zero dim tensor
                loss_value = float(loss.item())
                losses.append(loss_value)

    if was_training:
        model.train()

    avg_loss = sum(losses) / max(1, len(losses))
    return avg_loss



def train_one_step(
    model: torch.nn.Module,
    train_iter,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupScheduler,
    state: TrainState,
    device,
    *,
    grad_accum: int = 1,
    grad_clip: float = 1.0,
    amp: bool = False,
    amp_dtype: str = "bf16",
) -> dict:
    """Week 3 TODO: run one optimizer step.

    This should look like the SGD example in the notes, but using your
    TransformerLM, AdamW, cosine scheduler, and gradient clipping:
    set the learning rate, zero gradients, get a batch, compute loss, call
    backward, clip gradients, step the optimizer/scheduler, update `state`, and
    return a metrics row with at least train_loss, lr, grad_norm, step, and
    tokens.
    """

    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")

    model.train()
    lr = scheduler.get_lr(state.step)
    apply_lr(optimizer, lr)

    # clear old grad
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    tokens_this_step = 0    # token processed during this step

    # grad accumulation loop
    for _ in range(grad_accum):
        batch = next(train_iter)

        tokens_this_step += batch["input_ids"].numel()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast_context(device, amp, amp_dtype):
            out = model(input_ids, labels)
            raw_loss = out["loss"]
            scaled_loss = raw_loss / grad_accum         # linearity of derivatives

        scaled_loss.backward()
        total_loss += float(scaled_loss.detach().item())

    grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()

    state.step += 1
    state.tokens_seen += int(tokens_this_step)

    return {
        "train_loss": total_loss,
        "lr": float(lr),
        "grad_norm": float(grad_norm.detach().cpu().item()),
        "step": state.step,
        "tokens": state.tokens_seen,
    }


def _loader(path, tokenizer, context_length: int, batch_size: int, shuffle: bool):
    ds = PretrainDataset(path, tokenizer, context_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def _train_loader(path, tokenizer, context_length: int, batch_size: int, num_samples: int, seed: int):
    if Path(path).suffix == ".bin":
        ds = RandomBlockDataset(path, seq_len=context_length, num_samples=max(batch_size, num_samples), seed=seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return _loader(path, tokenizer, context_length, batch_size, shuffle=True)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _maybe_resume(cfg: dict, model, optimizer, scheduler, state: TrainState, device) -> None:
    resume = cfg.get("resume_from")
    if not resume:
        return
    load_checkpoint(resume, model=model, optimizer=optimizer, scheduler=scheduler, train_state=state, map_location=device)


def _save_training_checkpoint(out_dir, model, optimizer, state, model_cfg, tokenizer, cfg, scheduler, name: str) -> None:
    tokenizer_metadata = tokenizer.describe()
    tokenizer_metadata["encoded_train_manifest"] = str(encoded_manifest_path(cfg["train_path"]))
    tokenizer_metadata["encoded_valid_manifest"] = str(encoded_manifest_path(cfg["valid_path"]))
    save_checkpoint(
        Path(out_dir) / name,
        model,
        optimizer,
        state.step,
        model_cfg.to_dict(),
        cfg["tokenizer_path"],
        scheduler=scheduler,
        train_state=state,
        tokenizer_metadata=tokenizer_metadata,
    )


def train_pretrain(cfg: dict, max_steps: int | None = None) -> dict:
    """Provided orchestration for TinyStories pretraining.

    Students complete `evaluate_loss` and `train_one_step` above. This wrapper
    handles config, tokenizer/model construction, data loaders, checkpoint file
    names, metrics files, and sample generation.
    """
    set_seed(int(cfg.get("seed", 0)))
    out_dir = Path(cfg["run_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteBPETokenizer.load(cfg["tokenizer_path"])
    if Path(cfg["train_path"]).suffix == ".bin":
        validate_encoded_manifest(cfg["train_path"], cfg["tokenizer_path"])
    if Path(cfg["valid_path"]).suffix == ".bin":
        validate_encoded_manifest(cfg["valid_path"], cfg["tokenizer_path"])
    model_cfg = TransformerConfig.from_dict({**cfg["model"], "vocab_size": tokenizer.vocab_size})
    device = get_device(cfg.get("device", "auto"))
    model = TransformerLM(model_cfg).to(device)
    steps = max_steps or int(cfg["max_steps"])
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(cfg, steps)
    state = TrainState()
    _maybe_resume(cfg, model, optimizer, scheduler, state, device)

    remaining_steps = max(0, steps - state.step)
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    train_samples = remaining_steps * grad_accum * int(cfg["batch_size"]) + int(cfg["batch_size"])
    train_loader = _train_loader(
        cfg["train_path"],
        tokenizer,
        model_cfg.context_length,
        int(cfg["batch_size"]),
        train_samples,
        int(cfg.get("seed", 0)) + state.step * grad_accum,
    )
    train_iter = _cycle(train_loader)
    valid_loader = _loader(cfg["valid_path"], tokenizer, model_cfg.context_length, cfg["batch_size"], shuffle=False)
    logger = JsonlLogger(out_dir / "metrics.jsonl")
    amp = bool(cfg.get("amp", False))
    amp_dtype = cfg.get("amp_dtype", "bf16")
    eval_batches = int(cfg.get("eval_batches", 10))
    checkpoint_interval = int(cfg.get("checkpoint_interval", 0) or 0)
    start = time.perf_counter()
    model.train()

    while state.step < steps:
        metrics = train_one_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            state,
            device,
            grad_accum=grad_accum,
            grad_clip=cfg.get("grad_clip", 1.0),
            amp=amp,
            amp_dtype=amp_dtype,
        )
        if state.step % cfg.get("eval_interval", 25) == 0 or state.step == steps:
            val = evaluate_loss(model, valid_loader, device, max_batches=eval_batches, amp=amp, amp_dtype=amp_dtype)
            if state.best_valid_loss is None or val < state.best_valid_loss:
                state.best_valid_loss = val
            metrics.update(
                {
                    "valid_loss": val,
                    "valid_ppl": math.exp(min(20.0, val)),
                    "elapsed_sec": time.perf_counter() - start,
                }
            )
            logger.log(metrics)
        if cfg.get("sample_interval") and state.step % int(cfg["sample_interval"]) == 0:
            prompt = cfg.get("sample_prompt", "Once upon a time")
            sample = generate(model, tokenizer, prompt, device=device)
            (out_dir / f"sample_step_{state.step}.txt").write_text(
                f"PROMPT\n{prompt}\n\nCOMPLETION\n{sample}\n",
                encoding="utf-8",
            )
        if checkpoint_interval and (state.step % checkpoint_interval == 0 or state.step == steps):
            _save_training_checkpoint(out_dir, model, optimizer, state, model_cfg, tokenizer, cfg, scheduler, "checkpoint_last.pt")
            _save_training_checkpoint(
                out_dir,
                model,
                optimizer,
                state,
                model_cfg,
                tokenizer,
                cfg,
                scheduler,
                f"checkpoint_step_{state.step}.pt",
            )
    _save_training_checkpoint(out_dir, model, optimizer, state, model_cfg, tokenizer, cfg, scheduler, "checkpoint_last.pt")
    prompt = cfg.get("sample_prompt", "Once upon a time")
    sample = generate(model, tokenizer, prompt, device=device)
    (out_dir / "samples.txt").write_text(f"PROMPT\n{prompt}\n\nCOMPLETION\n{sample}\n", encoding="utf-8")
    (out_dir / "model_summary.json").write_text(__import__("json").dumps(model_summary(model), indent=2), encoding="utf-8")
    return {"run_dir": str(out_dir), "step": state.step, "sample": sample, "best_valid_loss": state.best_valid_loss}
