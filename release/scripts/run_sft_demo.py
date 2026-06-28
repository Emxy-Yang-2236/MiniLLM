#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.data.release import dataset_path_for_split, load_manifest, require_valid_manifest
from minillm.eval.sft import evaluate_sft, grade_sft_response
from minillm.model.config import TransformerConfig
from minillm.model.generation import generate
from minillm.model.transformer import TransformerLM
from minillm.tokenizer.bpe import ByteBPETokenizer
from minillm.train.checkpoint import load_checkpoint
from minillm.train.metrics import load_jsonl_metrics
from minillm.train.sft import train_sft
from minillm.utils.config import load_yaml
from minillm.utils.device import get_device


def _jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_model(path: str | Path, device):
    payload = load_checkpoint(path, map_location="cpu")
    tokenizer = ByteBPETokenizer.load(payload["tokenizer_path"])
    model = TransformerLM(TransformerConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def _loss_movement(path: str | Path) -> dict:
    rows = load_jsonl_metrics(path)
    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    return {
        "initial_train_loss": first.get("train_loss"),
        "final_train_loss": last.get("train_loss"),
        "initial_valid_loss": first.get("valid_loss"),
        "final_valid_loss": last.get("valid_loss"),
        "rows": len(rows),
    }


def _write_before_after(pre_ckpt: Path, sft_ckpt: Path, prompt_path: Path, out_path: Path, device) -> list[dict]:
    prompts = _jsonl(prompt_path)
    pre_model, pre_tok = _load_model(pre_ckpt, device)
    sft_model, sft_tok = _load_model(sft_ckpt, device)
    rows = []
    lines = [
        "# SFT Before/After Samples",
        "",
        "SFT here is a template-conditioned behavior demo. Arithmetic is not part of the main SFT task.",
        "",
        "# Story Template",
        "",
    ]
    current_section = "story_template"
    for i, item in enumerate(prompts, start=1):
        if current_section == "story_template" and item.get("task_type") == "label_template":
            lines += ["# Label Template", ""]
            current_section = "label_template"
        prompt = item["prompt"]
        max_new_tokens = 12 if item.get("task_type") == "label_template" else 96
        before = generate(pre_model, pre_tok, prompt, max_new_tokens=max_new_tokens, temperature=0.0, device=device)
        after = generate(sft_model, sft_tok, prompt, max_new_tokens=max_new_tokens, temperature=0.0, device=device)
        row = {
            "task_type": item.get("task_type", ""),
            "prompt": prompt,
            "expected": item.get("expected", item.get("response", "")),
            "before": before,
            "after": after,
            "before_score": grade_sft_response(before, item),
            "after_score": grade_sft_response(after, item),
        }
        rows.append(row)
        lines += [
            f"## Example {i}",
            "",
            f"**Task:** `{row['task_type']}`",
            "",
            "**Prompt**",
            "",
            prompt,
            "",
            "**Expected/reference style**",
            "",
            row["expected"],
            "",
            "**Pretrained completion**",
            "",
            before.strip() or "(empty)",
            "",
            "**SFT completion**",
            "",
            after.strip() or "(empty)",
            "",
            "**Automatic score**",
            "",
            "```json",
            json.dumps({"before": row["before_score"], "after": row["after_score"]}, indent=2),
            "```",
            "",
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return rows


def _release_verdict(sft_loss: dict, eval_sft: dict) -> tuple[str, dict]:
    sft_loss_ok = (
        sft_loss.get("initial_train_loss") is not None
        and sft_loss.get("final_train_loss") is not None
        and sft_loss["final_train_loss"] < sft_loss["initial_train_loss"]
        and sft_loss.get("final_valid_loss", float("inf")) <= sft_loss.get("initial_valid_loss", float("inf"))
    )
    checks = {
        "sft_loss_ok": sft_loss_ok,
        "story_template_followed_ok": eval_sft.get("story_template/template_followed", 0.0) >= 0.90,
        "story_template_topic_ok": eval_sft.get("story_template/topic_match", 0.0) >= 0.90,
        "label_template_exact_ok": eval_sft.get("label_template/exact_match", 0.0) >= 0.80,
    }
    return ("PASS" if all(checks.values()) else "WEAK PASS" if sft_loss_ok else "FAIL"), checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_student.yaml")
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--dataset_dir", default="../data/full_release")
    parser.add_argument("--run_dir", default="runs/student_pipeline/student/sft_template")
    parser.add_argument("--output_dir", default="outputs/sft_template")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (ROOT / dataset_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt = Path(args.base_ckpt)
    if not base_ckpt.is_absolute():
        base_ckpt = base_ckpt.resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    require_valid_manifest(dataset_dir)
    manifest = load_manifest(dataset_dir)
    cfg = load_yaml(args.config)
    cfg.update(
        {
            "device": args.device,
            "run_dir": str(run_dir),
            "train_path": str(dataset_path_for_split(dataset_dir, manifest, "sft_train")),
            "valid_path": str(dataset_path_for_split(dataset_dir, manifest, "sft_valid")),
            "eval_path": str(dataset_path_for_split(dataset_dir, manifest, "sft_eval")),
            "write_eval": False,
        }
    )
    start = time.perf_counter()
    result = train_sft(cfg, base_ckpt=str(base_ckpt))
    metrics_path = run_dir / "metrics.jsonl"
    shutil.copyfile(metrics_path, output_dir / "metrics_sft.jsonl")

    device = get_device(args.device)
    eval_ckpt = Path(result.get("best_checkpoint") or (run_dir / "checkpoint_last.pt"))
    fixed_prompt_path = dataset_path_for_split(dataset_dir, manifest, "before_after_prompts")
    before_after = _write_before_after(base_ckpt, eval_ckpt, fixed_prompt_path, output_dir / "sft_before_after.md", device)
    sft_model, sft_tok = _load_model(eval_ckpt, device)
    eval_sft = evaluate_sft(sft_model, sft_tok, dataset_path_for_split(dataset_dir, manifest, "sft_eval"), device=device)
    eval_sft_public = {k: v for k, v in eval_sft.items() if k != "outputs"}
    (output_dir / "eval_sft.json").write_text(json.dumps(eval_sft_public, indent=2), encoding="utf-8")
    sft_loss = _loss_movement(output_dir / "metrics_sft.jsonl")
    verdict, checks = _release_verdict(sft_loss, eval_sft_public)
    summary = {
        "verdict": verdict,
        "checks": checks,
        "base_checkpoint": str(base_ckpt),
        "sft_checkpoint": str(eval_ckpt),
        "sft_best_checkpoint": result.get("best_checkpoint"),
        "sft_loss": sft_loss,
        "eval_sft": eval_sft_public,
        "arithmetic_removed_from_main_demo": True,
        "before_after_count": len(before_after),
        "wall_clock_sec": time.perf_counter() - start,
    }
    (output_dir / "sft_demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    analysis_text = (
        "# SFT Template Demo Analysis\n\n"
        f"Verdict: {verdict}\n\n"
        "This SFT run is a template-conditioned generation demo: the model learns to enter fixed response templates, not arithmetic, reasoning, or general assistant behavior. Arithmetic has been removed from the main SFT train/valid/eval splits, fixed before/after prompts, and PASS criteria.\n\n"
        "## Loss Movement\n\n"
        f"- train loss: `{sft_loss.get('initial_train_loss')}` -> `{sft_loss.get('final_train_loss')}`\n"
        f"- valid loss: `{sft_loss.get('initial_valid_loss')}` -> `{sft_loss.get('final_valid_loss')}`\n"
        f"- logged eval rows: `{sft_loss.get('rows')}`\n\n"
        "## Held-out Metrics\n\n"
        f"- story_template/template_followed: `{eval_sft_public.get('story_template/template_followed')}`\n"
        f"- story_template/topic_match: `{eval_sft_public.get('story_template/topic_match')}`\n"
        f"- story_template/sentence_count_ok: `{eval_sft_public.get('story_template/sentence_count_ok')}`\n"
        f"- label_template/exact_match: `{eval_sft_public.get('label_template/exact_match')}`\n"
        f"- label_template/no_extra_prose: `{eval_sft_public.get('label_template/no_extra_prose')}`\n\n"
        "## Artifacts\n\n"
        f"- before/after: `{output_dir / 'sft_before_after.md'}`\n"
        f"- eval: `{output_dir / 'eval_sft.json'}`\n"
        f"- checkpoint: `{eval_ckpt}`\n\n"
        "## Raw Summary\n\n"
        "```json\n"
        + json.dumps(summary, indent=2)
        + "\n```\n"
    )
    (output_dir / "sft_analysis.md").write_text(analysis_text, encoding="utf-8")
    if verdict != "PASS":
        (output_dir / "failure_analysis.md").write_text(analysis_text, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
