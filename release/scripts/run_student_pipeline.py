#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.data.pretrain_dataset import encode_text_file_with_manifest, encoded_manifest_path, validate_encoded_manifest
from minillm.data.release import dataset_path_for_split, load_manifest, prepare_student_release_dataset, require_valid_manifest
from minillm.eval.sft import evaluate_sft, grade_sft_response
from minillm.model.config import TransformerConfig
from minillm.model.generation import generate
from minillm.model.transformer import TransformerLM
from minillm.systems.analysis import benchmark_report
from minillm.systems.benchmark import benchmark_sweep, write_csv, write_markdown_summary
from minillm.tokenizer.bpe import ByteBPETokenizer, GPT2_LIKE_PATTERN, tokenizer_file_sha256, train_bpe_from_file
from minillm.train.checkpoint import load_checkpoint
from minillm.train.metrics import load_jsonl_metrics, summarize_metrics
from minillm.train.pretrain import train_pretrain
from minillm.train.sft import train_sft
from minillm.utils.config import load_yaml
from minillm.utils.device import get_device


def _jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _mode_configs(root: Path, mode: str) -> tuple[Path, Path, Path]:
    tokenizer_cfg = root / "configs" / f"tokenizer_{mode}.yaml"
    if not tokenizer_cfg.exists():
        tokenizer_cfg = root / "configs" / "tokenizer_student_release.yaml"
    return (
        tokenizer_cfg,
        root / "configs" / f"pretrain_{mode}.yaml",
        root / "configs" / f"sft_{mode}.yaml",
    )


def _load_model(path: str | Path, device: torch.device):
    payload = load_checkpoint(path, map_location="cpu")
    tokenizer = ByteBPETokenizer.load(payload["tokenizer_path"])
    model = TransformerLM(TransformerConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, payload


def _prepare_metric_file(run_dir: str | Path) -> None:
    metrics = Path(run_dir) / "metrics.jsonl"
    if metrics.exists():
        metrics.unlink()


def _bin_token_count(path: str | Path) -> int:
    manifest_path = encoded_manifest_path(path)
    if manifest_path.exists():
        return int(json.loads(manifest_path.read_text(encoding="utf-8"))["token_count"])
    return Path(path).stat().st_size // 4


def _mb_to_bytes(cfg: dict, key: str) -> int | None:
    if key not in cfg or cfg[key] is None:
        return None
    return int(float(cfg[key]) * 1024 * 1024)


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


def _generate_before_after(pre_ckpt: Path, sft_ckpt: Path, prompt_path: Path, out_path: Path, device) -> list[dict]:
    prompts = _jsonl(prompt_path)
    pre_model, pre_tok, _ = _load_model(pre_ckpt, device)
    sft_model, sft_tok, _ = _load_model(sft_ckpt, device)
    rows: list[dict] = []
    for item in prompts:
        prompt = item["prompt"]
        max_new_tokens = 12 if item.get("task_type") == "label_template" else 96
        before = generate(pre_model, pre_tok, prompt, max_new_tokens=max_new_tokens, temperature=0.0, device=device)
        after = generate(sft_model, sft_tok, prompt, max_new_tokens=max_new_tokens, temperature=0.0, device=device)
        row = {
            "prompt": prompt,
            "expected": item.get("expected", item.get("response", "")),
            "before": before,
            "after": after,
            "task_type": item.get("task_type", ""),
        }
        row["before_score"] = grade_sft_response(before, item)
        row["after_score"] = grade_sft_response(after, item)
        rows.append(row)
    _write_before_after_md(rows, out_path)
    return rows


def _write_before_after_md(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SFT Before/After Samples",
        "",
        "SFT here is a template-conditioned behavior demo. Arithmetic is not part of the main SFT task.",
        "",
        "# Story Template",
        "",
    ]
    current_section = "story_template"
    for i, row in enumerate(rows, start=1):
        if current_section == "story_template" and row["task_type"] == "label_template":
            lines.extend(["# Label Template", ""])
            current_section = "label_template"
        lines.extend(
            [
                f"## Example {i}",
                "",
                f"**Task:** `{row['task_type']}`",
                "",
                "**Prompt**",
                "",
                row["prompt"],
                "",
                "**Expected/reference style**",
                "",
                row["expected"],
                "",
                "**Pretrained completion**",
                "",
                row["before"].strip() or "(empty)",
                "",
                "**SFT completion**",
                "",
                row["after"].strip() or "(empty)",
                "",
                "**Automatic score**",
                "",
                "```json",
                json.dumps({"before": row["before_score"], "after": row["after_score"]}, indent=2),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_run_summary(path: Path, data: dict) -> None:
    benchmark_rows = data["benchmark_rows"]
    lines = [
        "# MiniLLM Student Pipeline Summary",
        "",
        f"- Mode: `{data['mode']}`",
        f"- Dataset: `{data['dataset']['name']}` at `{data['dataset_dir']}`",
        f"- Dataset manifest: `{data['manifest_path']}`",
        f"- Device: `{data['device']}`",
        f"- Wall-clock seconds: `{data['wall_clock_sec']:.2f}`",
        f"- Tokenizer: `{data['tokenizer_path']}`",
        f"- Tokenizer vocab size: `{data['tokenizer_vocab_size']}`",
        f"- Pretrain checkpoint: `{data['pretrain_checkpoint']}`",
        f"- SFT checkpoint: `{data['sft_checkpoint']}`",
        f"- Pretrain steps: `{data['pretrain_steps']}`",
        f"- SFT steps: `{data['sft_steps']}`",
        "",
        "## Model",
        "",
        "```json",
        json.dumps(data["model_config"], indent=2),
        "```",
        "",
        "## Training Loss Movement",
        "",
        "```json",
        json.dumps({"pretrain": data["pretrain_loss"], "sft": data["sft_loss"]}, indent=2),
        "```",
        "",
        "## SFT Evaluation",
        "",
        "```json",
        json.dumps(data["eval_sft"], indent=2),
        "```",
        "",
        "## Before/After Examples",
        "",
    ]
    for row in data["before_after"][:5]:
        lines.extend([f"- `{row['prompt']}`", f"  - before: {row['before'].strip() or '(empty)'}", f"  - after: {row['after'].strip() or '(empty)'}"])
    lines.extend(
        [
            "",
            "## Training Measurement Sanity Benchmark",
            "",
            "| attention | precision | batch | seq | full_step_mean_sec | tokens/sec |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in benchmark_rows:
        lines.append(
            f"| {row.get('attention')} | {row.get('precision')} | {row.get('batch_size')} | {row.get('context_length')} | {row.get('full_step_mean_sec')} | {row.get('tokens_per_sec')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_pretrain_samples(pre_ckpt: Path, out_path: Path, device) -> list[dict]:
    prompts = [
        "Once upon a time, there was a little dragon",
        "Lily found a tiny box under her bed",
        "The small robot wanted to make a friend",
        "Tom and Mia went to the forest",
        "The cat",
        "One day",
        "In the morning",
        "A little girl",
    ]
    model, tok, _ = _load_model(pre_ckpt, device)
    rows = []
    lines = ["# Pretraining Fixed-Prompt Samples", "", "Sampling: greedy, temperature=0.0, max_new_tokens=80.", ""]
    for i, prompt in enumerate(prompts, 1):
        completion = generate(model, tok, prompt, max_new_tokens=80, temperature=0.0, device=device)
        rows.append({"prompt": prompt, "output": completion})
        lines += [f"## Sample {i}", "", "**Prompt**", "", prompt, "", "**Pretrained completion**", "", completion.strip() or "(empty)", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return rows


def _write_artifact_reports(output_dir: Path, summary: dict, pretrain_samples: list[dict]) -> None:
    pre_loss = summary["pretrain_loss"]
    sft_loss = summary["sft_loss"]
    eval_sft = summary["eval_sft"]
    pre_rows = load_jsonl_metrics(output_dir / "metrics_pretrain.jsonl")
    sft_rows = load_jsonl_metrics(output_dir / "metrics_sft.jsonl")
    tokenizer_manifest = json.loads(Path(summary["tokenizer_manifest"]).read_text(encoding="utf-8"))
    encoded_manifest_data = {
        name: json.loads(Path(path).read_text(encoding="utf-8")) for name, path in summary["encoded_manifests"].items()
    }
    sft_manifest_path = Path(summary["dataset_dir"]) / "sft" / "manifest.json"
    sft_manifest = json.loads(sft_manifest_path.read_text(encoding="utf-8")) if sft_manifest_path.exists() else {}
    overlap = sft_manifest.get("overlap_checks", {})
    benchmark_rows = summary["benchmark_rows"]
    has_mixed_precision = any(row.get("precision") in {"bf16", "fp16"} and row.get("status") == "ok" for row in benchmark_rows)
    cuda_run = str(summary["device"]).startswith("cuda")
    pretrain_ok = (
        pre_loss.get("initial_train_loss") is not None
        and pre_loss.get("final_train_loss") is not None
        and pre_loss["final_train_loss"] < pre_loss["initial_train_loss"]
        and pre_loss["final_valid_loss"] <= pre_loss["initial_valid_loss"]
    )
    sft_ok = (
        sft_loss.get("initial_train_loss") is not None
        and sft_loss.get("final_train_loss") is not None
        and sft_loss["final_train_loss"] < sft_loss["initial_train_loss"]
        and sft_loss.get("initial_valid_loss") is not None
        and sft_loss.get("final_valid_loss") is not None
        and sft_loss["final_valid_loss"] <= sft_loss["initial_valid_loss"]
        and overlap.get("ok", False)
    )
    story_template_ok = eval_sft.get("story_template/template_followed", 0.0) >= 0.90
    story_topic_ok = eval_sft.get("story_template/topic_match", 0.0) >= 0.90
    label_ok = eval_sft.get("label_template/exact_match", 0.0) >= 0.85
    training_measurement_ok = (not cuda_run) or has_mixed_precision
    primary_sft_ok = story_template_ok and story_topic_ok and label_ok
    verdict = (
        "PASS"
        if pretrain_ok and sft_ok and primary_sft_ok and training_measurement_ok
        else "WEAK PASS"
        if pretrain_ok and sft_ok and training_measurement_ok
        else "FAIL"
    )
    metrics_summary = {
        "verdict": verdict,
        "pretrain": pre_loss,
        "sft": {**sft_loss, "best_step": summary.get("sft_best_step"), "best_valid_loss": summary.get("sft_best_valid_loss")},
        "eval_sft": eval_sft,
        "sft_overlap": overlap,
        "benchmark": summary["benchmark_report"],
        "release_checks": {
            "pretrain_ok": pretrain_ok,
            "sft_loss_ok": sft_ok,
            "story_template_followed_ok": story_template_ok,
            "story_template_topic_ok": story_topic_ok,
            "label_template_exact_ok": label_ok,
            "arithmetic_removed_from_main_demo": True,
            "training_measurement_ok": training_measurement_ok,
        },
        "tokenizer": {
            "path": summary["tokenizer_path"],
            "sha256": summary["tokenizer_sha256"],
            "pretokenizer": summary["tokenizer_pretokenizer"],
            "vocab_size": summary["tokenizer_vocab_size"],
        },
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "tokenizer": {
            "path": summary["tokenizer_path"],
            "sha256": summary["tokenizer_sha256"],
            "manifest": summary["tokenizer_manifest"],
            "vocab_size": summary["tokenizer_vocab_size"],
            "num_merges": summary["tokenizer_num_merges"],
            "pretokenizer": summary["tokenizer_pretokenizer"],
        },
        "raw_data": {"dataset_dir": summary["dataset_dir"], "manifest": summary["manifest_path"], "sha256": _sha256(summary["manifest_path"])},
        "encoded_data": encoded_manifest_data,
        "checkpoints": {
            "pretrain": summary["pretrain_checkpoint"],
            "sft_last": summary["sft_checkpoint"],
            "sft_best": summary["sft_best_checkpoint"],
            "sft_eval": summary["sft_eval_checkpoint"],
        },
        "metrics": summary["outputs"],
        "training_measurement": {
            "benchmark_sanity_csv": summary["outputs"]["benchmark_sanity_csv"],
            "benchmark_sanity_md": summary["outputs"]["benchmark_sanity_md"],
        },
        "sha256": {
            "pretrain_checkpoint": _sha256(summary["pretrain_checkpoint"]),
            "sft_checkpoint": _sha256(summary["sft_eval_checkpoint"]),
        },
    }
    (output_dir / "run_artifacts_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    pre_table = pre_rows[:1] + pre_rows[max(0, len(pre_rows) // 4 - 1): len(pre_rows) // 4] + pre_rows[max(0, len(pre_rows) // 2 - 1): len(pre_rows) // 2] + pre_rows[max(0, 3 * len(pre_rows) // 4 - 1): 3 * len(pre_rows) // 4] + pre_rows[-1:]
    seen = set()
    pre_table = [row for row in pre_table if not (row.get("step") in seen or seen.add(row.get("step")))]
    sft_table = sft_rows[:1] + sft_rows[-4:] if len(sft_rows) > 4 else sft_rows
    model_cfg = summary["model_config"]
    scaling = summary["scaling"]
    benchmark_sample = benchmark_rows[: min(8, len(benchmark_rows))]
    report = [
        "# MiniLLM Release-Candidate Run Report",
        "",
        "## Project Goal",
        "",
        "Expected pipeline: Raw TinyStories text -> train BPE tokenizer -> encode pretraining train/valid text to .bin -> train tiny Transformer LM -> generate pretrained samples -> run tiny-SFT from the pretrained checkpoint -> generate fixed before/after SFT samples -> evaluate SFT -> run training measurement sanity benchmark -> summarize artifacts.",
        "",
        "Educational goal: students should see that their tokenizer, Transformer, training loop, checkpointing, generation, SFT, and training measurement code form one coherent project rather than isolated scripts.",
        "",
        "## Command And Environment",
        "",
        f"- command: `{summary['command']}`",
        f"- start_time_utc: `{summary['start_time_utc']}`",
        f"- end_time_utc: `{summary['end_time_utc']}`",
        f"- wall_clock_sec: `{summary['wall_clock_sec']:.2f}`",
        f"- mode: `{summary['mode']}`",
        f"- device: `{summary['device']}`",
        f"- python: `{summary['environment']['python']}`",
        f"- torch: `{summary['environment']['torch']}`",
        f"- cuda_available: `{summary['environment']['cuda_available']}`",
        f"- gpu: `{summary['environment']['gpu']}`",
        f"- peak_gpu_memory_bytes: `{summary['environment']['peak_gpu_memory_bytes']}`",
        f"- git_commit: `{summary['environment']['git_commit']}`",
        f"- configs: `{summary['config_paths']}`",
        f"- output_dir: `{output_dir}`",
        "",
        "## Tokenizer",
        "",
        f"- tokenizer: `{summary['tokenizer_path']}`",
        f"- tokenizer sha256: `{summary['tokenizer_sha256']}`",
        f"- pretokenizer: `{summary['tokenizer_pretokenizer']}`",
        f"- regex: `{summary['tokenizer_regex']}`",
        f"- vocab size: `{summary['tokenizer_vocab_size']}`",
        f"- merges: `{summary['tokenizer_num_merges']}`",
        f"- tie_break: `{tokenizer_manifest.get('tie_break')}`",
        f"- training_bytes_read: `{tokenizer_manifest.get('training_bytes_read')}`",
        f"- newlines_preserved: `yes`",
        f"- tokenizer manifest: `{summary['tokenizer_manifest']}`",
        "",
        "## Dataset / manifests",
        "",
        f"- dataset manifest: `{summary['manifest_path']}`",
        f"- dataset verification: `{summary['dataset']['verification']}`",
        f"- encoded manifest train: `{summary['encoded_manifests']['train']}`",
        f"- encoded manifest valid: `{summary['encoded_manifests']['valid']}`",
        f"- encoded dtype train/valid: `{encoded_manifest_data['train']['dtype']}` / `{encoded_manifest_data['valid']['dtype']}`",
        f"- encoded tokenizer sha match: `{encoded_manifest_data['train']['tokenizer_sha256'] == summary['tokenizer_sha256'] and encoded_manifest_data['valid']['tokenizer_sha256'] == summary['tokenizer_sha256']}`",
        "",
        "## Model and scaling sanity check",
        "",
        "```json",
        json.dumps(model_cfg, indent=2),
        "```",
        "",
        f"- total_parameters: `{scaling['total_parameters']}`",
        f"- trainable_parameters: `{scaling['trainable_parameters']}`",
        f"- training_tokens_consumed: `{scaling['training_tokens_consumed']}`",
        f"- tokens_per_parameter: `{scaling['tokens_per_parameter']:.3f}`",
        f"- flops_estimate_6ND: `{scaling['flops_estimate_6ND']:.3e}`",
        "",
        "## Pretraining",
        "",
        f"- train loss: `{pre_loss.get('initial_train_loss')}` -> `{pre_loss.get('final_train_loss')}`",
        f"- valid loss: `{pre_loss.get('initial_valid_loss')}` -> `{pre_loss.get('final_valid_loss')}`",
        f"- steps: `{summary['pretrain_steps']}`",
        f"- checkpoint: `{summary['pretrain_checkpoint']}`",
        f"- samples: `{output_dir / 'pretrain_samples.md'}`",
        "",
        "| step | train_loss | valid_loss | lr | tokens | elapsed_sec |",
        "|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row.get('step')} | {row.get('train_loss')} | {row.get('valid_loss')} | {row.get('lr')} | {row.get('tokens')} | {row.get('elapsed_sec')} |"
            for row in pre_table
        ],
        "",
        "## Pretraining fixed samples",
        "",
        *[f"- `{row['prompt']}` -> {row['output'].strip()[:180] or '(empty)'}" for row in pretrain_samples],
        "",
        "## SFT dataset split",
        "",
        f"- SFT counts: `{sft_manifest.get('counts_by_split', {})}`",
        f"- SFT counts by task: `{sft_manifest.get('counts_by_task', {})}`",
        f"- SFT overlap: `{overlap}`",
        "",
        "## SFT",
        "",
        f"- train loss: `{sft_loss.get('initial_train_loss')}` -> `{sft_loss.get('final_train_loss')}`",
        f"- valid loss: `{sft_loss.get('initial_valid_loss')}` -> `{sft_loss.get('final_valid_loss')}`",
        f"- steps: `{summary['sft_steps']}`",
        f"- eval checkpoint: `{summary['sft_eval_checkpoint']}`",
        f"- best-valid checkpoint: `{summary['sft_best_checkpoint']}` at step `{summary.get('sft_best_step')}`",
        f"- held-out eval: `{eval_sft}`",
        f"- before/after: `{output_dir / 'sft_before_after.md'}`",
        "",
        "| step | train_loss | valid_loss | lr | tokens | elapsed_sec |",
        "|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row.get('step')} | {row.get('train_loss')} | {row.get('valid_loss')} | {row.get('lr')} | {row.get('tokens')} | {row.get('elapsed_sec')} |"
            for row in sft_table
        ],
        "",
        "## Before/After Summary",
        "",
        *[
            f"- `{row['prompt']}` | before: {row['before'].strip()[:100] or '(empty)'} | after: {row['after'].strip()[:100] or '(empty)'}"
            for row in summary["before_after"][:10]
        ],
        "",
        "## Training Measurement Mini-lab",
        "",
        f"- rows: `{len(benchmark_rows)}`",
        f"- precisions: `{sorted({row.get('precision') for row in benchmark_rows})}`",
        f"- sanity benchmark csv: `{output_dir / 'benchmark_sanity.csv'}`",
        "- This sanity benchmark only proves that the measurement path runs. For the final report, run `scripts/benchmark_sweep.py` and answer `docs/Training Measurement Mini-lab.md`.",
        "",
        "| backend | dtype | batch | seq_len | full_step_mean_sec | full_step_median_sec | full_step_p90_sec | tokens/sec | peak_memory |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row.get('backend')} | {row.get('dtype')} | {row.get('batch_size')} | {row.get('seq_len')} | {row.get('full_step_mean_sec')} | {row.get('full_step_median_sec')} | {row.get('full_step_p90_sec')} | {row.get('tokens_per_sec')} | {row.get('peak_memory_bytes')} |"
            for row in benchmark_sample
        ],
        "",
        "## Release verdict",
        "",
        verdict,
        "",
        "## Main Issues And Next Steps",
        "",
        "- Arithmetic has been removed from the main SFT train/valid/eval/fixed prompts and PASS criteria.",
        "- The official SFT demo focuses on template-conditioned story generation and single-label classification.",
    ]
    (output_dir / "pipeline_run_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if verdict != "PASS":
        (output_dir / "failure_analysis.md").write_text(
            "# Failure / Weakness Analysis\n\n"
            f"Verdict: {verdict}\n\n"
            f"- pretrain_ok: {pretrain_ok}\n"
            f"- sft_ok: {sft_ok}\n"
            f"- story_template_followed_ok: {story_template_ok}\n"
            f"- story_template_topic_ok: {story_topic_ok}\n"
            f"- label_template_exact_ok: {label_ok}\n"
            f"- training_measurement_ok: {training_measurement_ok}\n"
            "- If the verdict is WEAK PASS, the usual causes are weak SFT main-task generalization or insufficient mixed-precision evidence in the training measurement run.\n",
            encoding="utf-8",
        )
    (output_dir / "train.log").write_text(
        "\n".join(
            [
                f"command: {summary['command']}",
                f"start_time_utc: {summary['start_time_utc']}",
                f"end_time_utc: {summary['end_time_utc']}",
                f"dataset: {summary['dataset_dir']}",
                f"tokenizer: {summary['tokenizer_path']}",
                f"pretrain_checkpoint: {summary['pretrain_checkpoint']}",
                f"sft_eval_checkpoint: {summary['sft_eval_checkpoint']}",
                f"pretrain_loss: {pre_loss}",
                f"sft_loss: {sft_loss}",
                f"eval_sft: {eval_sft}",
                f"benchmark_rows: {len(benchmark_rows)}",
                f"verdict: {verdict}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_pipeline(args) -> dict:
    root = Path(__file__).resolve().parents[1]
    if args.debug_toy_data:
        raise SystemExit("debug toy data mode has been removed; the canonical pipeline always uses data/full_release")

    mode = args.mode
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (root / dataset_dir).resolve()
    output_name = args.output_dir or ("outputs/release_candidate" if mode == "student" else "outputs/smoke")
    output_dir = Path(output_name)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    start_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.prepare_dataset:
        print(f"[pipeline] preparing dataset at {dataset_dir}", flush=True)
        prepare_student_release_dataset(dataset_dir, force=args.force_prepare_dataset)
    print(f"[pipeline] verifying dataset at {dataset_dir}", flush=True)
    verification = require_valid_manifest(dataset_dir)
    manifest = load_manifest(dataset_dir)
    train_text_path = dataset_path_for_split(dataset_dir, manifest, "pretrain_train")
    valid_text_path = dataset_path_for_split(dataset_dir, manifest, "pretrain_valid")
    sft_train_path = dataset_path_for_split(dataset_dir, manifest, "sft_train")
    sft_valid_path = dataset_path_for_split(dataset_dir, manifest, "sft_valid")
    sft_eval_path = dataset_path_for_split(dataset_dir, manifest, "sft_eval")
    fixed_prompt_path = dataset_path_for_split(dataset_dir, manifest, "before_after_prompts")
    (output_dir / "dataset_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    tok_cfg_path, pre_cfg_path, sft_cfg_path = _mode_configs(root, mode)
    tok_cfg = load_yaml(tok_cfg_path)
    pre_cfg = load_yaml(pre_cfg_path)
    sft_cfg = load_yaml(sft_cfg_path)
    device_name = args.device or pre_cfg.get("device", "auto")
    pre_cfg["device"] = device_name
    sft_cfg["device"] = device_name

    run_root = root / "runs" / "student_pipeline" / mode
    tokenizer_path = run_root / "tokenizer" / "tokenizer.json"
    train_bin = run_root / "tokenizer" / "pretrain_train.bin"
    valid_bin = run_root / "tokenizer" / "pretrain_valid.bin"
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and tokenizer_path.exists() and train_bin.exists() and valid_bin.exists():
        print("[pipeline] reusing tokenizer and encoded pretraining binaries", flush=True)
        tok = ByteBPETokenizer.load(tokenizer_path)
        validate_encoded_manifest(train_bin, tokenizer_path)
        validate_encoded_manifest(valid_bin, tokenizer_path)
        train_tokens = _bin_token_count(train_bin)
        valid_tokens = _bin_token_count(valid_bin)
    else:
        print("[pipeline] training tokenizer", flush=True)
        tok = train_bpe_from_file(
            train_text_path,
            int(tok_cfg["vocab_size"]),
            tokenizer_path,
            special_tokens=tok_cfg.get("special_tokens"),
            min_frequency=int(tok_cfg.get("min_frequency", 2)),
            max_bytes=tok_cfg.get("max_bytes", _mb_to_bytes(tok_cfg, "tokenizer_train_mb")),
            pretokenizer=tok_cfg.get("pretokenizer", "gpt2_like"),
            tie_break=tok_cfg.get("tie_break", "max"),
        )
        print("[pipeline] encoding pretraining data", flush=True)
        train_manifest = encode_text_file_with_manifest(
            train_text_path,
            tok,
            train_bin,
            tokenizer_path=tokenizer_path,
            max_bytes=_mb_to_bytes(tok_cfg, "pretrain_train_mb"),
            encode_command="scripts/run_student_pipeline.py",
        )
        valid_manifest = encode_text_file_with_manifest(
            valid_text_path,
            tok,
            valid_bin,
            tokenizer_path=tokenizer_path,
            max_bytes=_mb_to_bytes(tok_cfg, "pretrain_valid_mb"),
            encode_command="scripts/run_student_pipeline.py",
        )
        validate_encoded_manifest(train_bin, tokenizer_path)
        validate_encoded_manifest(valid_bin, tokenizer_path)
        train_tokens = train_manifest["token_count"]
        valid_tokens = valid_manifest["token_count"]

    pre_cfg["tokenizer_path"] = str(tokenizer_path)
    pre_cfg["train_path"] = str(train_bin)
    pre_cfg["valid_path"] = str(valid_bin)
    pre_cfg["run_dir"] = str(run_root / "pretrain")
    pre_ckpt = Path(pre_cfg["run_dir"]) / "checkpoint_last.pt"
    if args.resume and pre_ckpt.exists():
        print(f"[pipeline] resuming pretraining from {pre_ckpt}", flush=True)
        pre_cfg["resume_from"] = str(pre_ckpt)
    else:
        _prepare_metric_file(pre_cfg["run_dir"])
    print("[pipeline] running pretraining", flush=True)
    pretrain_result = train_pretrain(pre_cfg)

    sft_cfg["train_path"] = str(sft_train_path)
    sft_cfg["valid_path"] = str(sft_valid_path)
    sft_cfg["eval_path"] = str(sft_eval_path)
    sft_cfg["write_eval"] = False
    sft_cfg["run_dir"] = str(run_root / "sft")
    sft_ckpt = Path(sft_cfg["run_dir"]) / "checkpoint_last.pt"
    if args.resume and sft_ckpt.exists():
        print(f"[pipeline] resuming SFT from {sft_ckpt}", flush=True)
        sft_cfg["resume_from"] = str(sft_ckpt)
    else:
        _prepare_metric_file(sft_cfg["run_dir"])
    print("[pipeline] running SFT", flush=True)
    sft_result = train_sft(sft_cfg, base_ckpt=str(pre_ckpt))

    shutil.copyfile(Path(pre_cfg["run_dir"]) / "metrics.jsonl", output_dir / "metrics_pretrain.jsonl")
    shutil.copyfile(Path(sft_cfg["run_dir"]) / "metrics.jsonl", output_dir / "metrics_sft.jsonl")

    device = get_device(device_name)
    print("[pipeline] generating before/after samples and SFT eval", flush=True)
    demo_sft_ckpt = Path(sft_result.get("best_checkpoint", sft_ckpt))
    before_after = _generate_before_after(
        pre_ckpt,
        demo_sft_ckpt,
        fixed_prompt_path,
        output_dir / "sft_before_after.md",
        device,
    )
    sft_model, sft_tok, payload = _load_model(demo_sft_ckpt, device)
    eval_sft = evaluate_sft(sft_model, sft_tok, sft_eval_path, device=device)
    eval_sft_public = {k: v for k, v in eval_sft.items() if k != "outputs"}
    (output_dir / "eval_sft.json").write_text(json.dumps(eval_sft_public, indent=2), encoding="utf-8")

    base_cfg = TransformerConfig.from_dict(payload["config"])
    print("[pipeline] running training measurement sanity benchmark", flush=True)
    seq_lens = sorted(set([min(32, base_cfg.context_length), min(128, base_cfg.context_length)])) if mode == "smoke" else sorted(set([min(32, base_cfg.context_length), min(128, base_cfg.context_length), base_cfg.context_length]))
    dev = get_device(device_name)
    precisions = ["fp32"]
    if dev.type == "cuda":
        if torch.cuda.is_bf16_supported():
            precisions.append("bf16")
        else:
            precisions.append("fp16")
    benchmark_rows = benchmark_sweep(
        base_cfg,
        seq_lens=seq_lens,
        batch_sizes=[1] if mode == "smoke" else [1, 2],
        attentions=["naive", "sdpa"],
        precisions=precisions,
        device=device_name,
        warmup=0 if mode == "smoke" else 1,
        steps=1 if mode == "smoke" else 3,
    )
    write_csv(benchmark_rows, output_dir / "benchmark_sanity.csv")
    write_markdown_summary(benchmark_rows, output_dir / "benchmark_sanity.md")
    total_params = sum(param.numel() for param in sft_model.parameters())
    trainable_params = sum(param.numel() for param in sft_model.parameters() if param.requires_grad)
    final_pre_rows = load_jsonl_metrics(output_dir / "metrics_pretrain.jsonl")
    training_tokens_consumed = int(final_pre_rows[-1].get("tokens", 0)) if final_pre_rows else 0
    git_commit = "not available"
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        pass
    dev_for_env = get_device(device_name)
    gpu_name = torch.cuda.get_device_name(dev_for_env) if dev_for_env.type == "cuda" else "not available"
    peak_gpu_memory = torch.cuda.max_memory_allocated(dev_for_env) if dev_for_env.type == "cuda" else "not available"
    end_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_clock = time.perf_counter() - start

    summary = {
        "mode": mode,
        "command": " ".join([sys.executable, *sys.argv]),
        "start_time_utc": start_wall,
        "end_time_utc": end_wall,
        "config_paths": {"tokenizer": str(tok_cfg_path), "pretrain": str(pre_cfg_path), "sft": str(sft_cfg_path)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu_name,
            "peak_gpu_memory_bytes": peak_gpu_memory,
            "git_commit": git_commit,
        },
        "dataset": {"name": manifest["name"], "verification": verification["ok"], "files": manifest["files"]},
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(dataset_dir / "manifest.json"),
        "device": str(device),
        "wall_clock_sec": wall_clock,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_file_sha256(tokenizer_path),
        "tokenizer_manifest": str(tokenizer_path.with_name("tokenizer_manifest.json")),
        "tokenizer_vocab_size": tok.vocab_size,
        "tokenizer_num_merges": len(tok.merges),
        "tokenizer_pretokenizer": tok.pretokenizer,
        "tokenizer_regex": GPT2_LIKE_PATTERN if tok.pretokenizer == "gpt2_like" else "",
        "encoded_tokens": {"train": train_tokens, "valid": valid_tokens},
        "encoded_manifests": {
            "train": str(encoded_manifest_path(train_bin)),
            "valid": str(encoded_manifest_path(valid_bin)),
        },
        "pretrain_checkpoint": str(pre_ckpt),
        "sft_checkpoint": str(sft_ckpt),
        "sft_best_checkpoint": sft_result.get("best_checkpoint", str(sft_ckpt)),
        "sft_eval_checkpoint": str(demo_sft_ckpt),
        "pretrain_steps": pretrain_result["step"],
        "sft_steps": sft_result["step"],
        "sft_best_step": sft_result.get("best_step"),
        "sft_best_valid_loss": sft_result.get("best_valid_loss"),
        "pretrain_loss": _loss_movement(output_dir / "metrics_pretrain.jsonl"),
        "sft_loss": _loss_movement(output_dir / "metrics_sft.jsonl"),
        "pretrain_metrics": summarize_metrics(output_dir / "metrics_pretrain.jsonl"),
        "sft_metrics": summarize_metrics(output_dir / "metrics_sft.jsonl"),
        "eval_sft": eval_sft_public,
        "arithmetic_removed_from_main_demo": True,
        "before_after": before_after,
        "model_config": payload["config"],
        "scaling": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "training_tokens_consumed": training_tokens_consumed,
            "tokens_per_parameter": training_tokens_consumed / max(1, total_params),
            "flops_estimate_6ND": 6.0 * float(total_params) * float(training_tokens_consumed),
        },
        "benchmark_rows": benchmark_rows,
        "benchmark_report": benchmark_report(benchmark_rows),
        "outputs": {
            "metrics_pretrain": str(output_dir / "metrics_pretrain.jsonl"),
            "metrics_sft": str(output_dir / "metrics_sft.jsonl"),
            "eval_sft": str(output_dir / "eval_sft.json"),
            "sft_before_after": str(output_dir / "sft_before_after.md"),
            "benchmark_sanity_csv": str(output_dir / "benchmark_sanity.csv"),
            "benchmark_sanity_md": str(output_dir / "benchmark_sanity.md"),
            "run_summary": str(output_dir / "run_summary.md"),
            "dataset_verification": str(output_dir / "dataset_verification.json"),
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_run_summary(output_dir / "run_summary.md", summary)
    pretrain_samples = _write_pretrain_samples(pre_ckpt, output_dir / "pretrain_samples.md", device)
    _write_artifact_reports(output_dir, summary, pretrain_samples)
    print(json.dumps({k: summary[k] for k in ["mode", "dataset", "tokenizer_path", "pretrain_checkpoint", "sft_checkpoint", "outputs"]}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "student"], default="smoke")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset_dir", default="../data/full_release")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare_dataset", action="store_true")
    parser.add_argument("--force_prepare_dataset", action="store_true")
    parser.add_argument("--debug-toy-data", action="store_true", help="Rejected compatibility flag; the canonical pipeline always uses data/full_release.")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
