# MiniLLM Release

This is the student implementation directory.

Work here unless the course staff explicitly tells you otherwise. The instructor implementation has the same overall layout under `../reference/`.

Important directories:

- `minillm/`: the package you complete.
- `configs/`: YAML configs for tokenizer training, pretraining, SFT, and benchmark runs.
- `scripts/`: command-line entrypoints for training, evaluation, generation, and reports.
- `runs/`: generated checkpoints, tokenizer files, encoded data, and metrics.
- `outputs/`: generated samples, evaluation results, benchmark summaries, and report artifacts.

Useful commands:

```bash
python -m pytest -q ../shared/tests
python -m pytest -q ../shared/tests/cs336_a1_exact
python scripts/verify_student_release_data.py --dataset_dir ../data/full_release
python scripts/run_student_pipeline.py --mode smoke --device cpu
```

After you have a checkpoint, inspect the next-token distribution with the provided script (optional but interesting!):

```bash
python scripts/inspect_next_token.py \
  --ckpt runs/student_pipeline/student/pretrain/checkpoint_last.pt \
  --prompt "Once upon a time" \
  --top_k 8 \
  --steps 20 \
  --mode choose \
  --device auto
```

`--mode greedy` automatically chooses the highest-probability token. `--mode choose` lets you pick from the displayed top-k tokens at each step.

On Apple Silicon, use `--device mps` for a local MPS run. `--device auto` chooses CUDA first, then MPS, then CPU. MPS runs are fp32-only in this project; CUDA mixed-precision benchmark rows should be marked skipped.

The canonical data lives outside this directory in `../data/`.
Do not edit dataset files by hand.

Config paths are written relative to this directory.
Inputs usually point to `../data/...`; generated artifacts usually go under `runs/...` and `outputs/...`.
The pipeline command is expected to work after you complete the TODOs in the starter implementation.

Provided infrastructure includes release-data verification, manifest handling, full pipeline orchestration, SFT train/eval/report glue, run summaries, and code-usage audit scripts. Your main implementation work is in tokenizer, model, optimizer/training, pretraining data, SFT response-only data masking, and generation.
