# Training Measurement Mini-lab

By Week 4, your Transformer LM should already train and produce samples.
This mini-lab asks a different question:

```text
How expensive is one training step of my model, and what changes that cost?
```

This is a measurement and interpretation task.
You will run the provided scripts, read the tables, and explain what the numbers mean.
You are not expected to write kernels, use Triton, add distributed training, or tune for a speed contest.

## What You Are Measuring

The mini-lab measures the model compute path on synthetic random token batches.
It focuses on the cost of a Transformer training step:

```text
forward pass
  -> backward pass
  -> optimizer step
  -> full train step time
```

The table reports both forward-only timing and full-step timing.
Use the questions below to decide which number best supports a claim about training cost.

The pipeline automatically writes a small sanity benchmark:

```text
outputs/release_candidate/benchmark_sanity.csv
outputs/release_candidate/benchmark_sanity.md
```

That sanity benchmark proves the measurement path runs.
For your final report, run the standalone sweep below.


## A Small Amount Of Accounting

Most Transformer compute comes from matrix multiplications.
For a matrix product:

```text
A has shape (m, n)
B has shape (n, p)
A @ B costs about 2mnp floating-point operations
```

This is only a rough accounting rule, but it helps explain the benchmark.
In attention, the model forms scores with:

```text
Q @ K^T
```

If the sequence length is `T`, this score matrix has shape roughly `(T, T)` for each head.
So increasing sequence length usually increases both time and memory more sharply than increasing a small constant elsewhere.

You do not need to derive full model FLOPs in this mini-lab.


## What To Run

First, run the full student pipeline so you have a trained checkpoint, final samples, and the sanity benchmark:

```bash
cd release
python scripts/run_student_pipeline.py --mode student --device cuda
python scripts/plot_training_curves.py
```

If CUDA is unavailable, use `--device mps` on Apple Silicon or `--device cpu` elsewhere. Tokenizer training is CPU-bound. MPS can run smoke and low-resource training, but it is fp32-only in this project and may be much slower than CUDA for the full student-mode run. If you use MPS, clearly state that CUDA mixed-precision and CUDA memory rows were skipped.

The full pipeline writes the main artifacts under `outputs/release_candidate/`:

```text
run_summary.md
pipeline_run_report.md
metrics_summary.json
training_curves.svg
pretrain_samples.md
sft_before_after.md
eval_sft.json
benchmark_sanity.csv
benchmark_sanity.md
run_artifacts_manifest.json
```

Then run the final measurement sweep:

```bash
python scripts/benchmark_sweep.py \
  --config configs/pretrain_student.yaml \
  --seq_lens 32,128,256 \
  --batch_sizes 1,2 \
  --attentions naive,sdpa \
  --precisions fp32,bf16 \
  --device cuda \
  --warmup 2 \
  --steps 5 \
  --csv outputs/final_systems.csv \
  --md outputs/final_systems.md
```

On Apple Silicon:

```bash
python scripts/benchmark_sweep.py \
  --config configs/pretrain_student.yaml \
  --seq_lens 32,128,256 \
  --batch_sizes 1,2 \
  --attentions naive,sdpa \
  --precisions fp32 \
  --device mps \
  --warmup 2 \
  --steps 5 \
  --csv outputs/final_systems.csv \
  --md outputs/final_systems.md
```

You can also check that naive attention and SDPA produce close outputs:

```bash
python scripts/check_attention_backends.py \
  --config configs/pretrain_student.yaml \
  --seq_len 128 \
  --batch_size 2 \
  --device cuda
```

Use `--device mps` or `--device cpu` if needed.

## What Files You Get

The final sweep writes:

```text
outputs/final_systems.csv
outputs/final_systems.md
```

Use these together with the report template:

```text
../reports/templates/training_measurement_report.md
```

Your report should include the command, hardware, a small benchmark table, and answers to the questions below.

## Questions To Answer

You should answer Q1-Q6 clearly in your report.

### Q1. Why do CUDA benchmarks need warmup and synchronization?

Discuss CUDA's execution model, first-iteration overhead, and what can go wrong if timing code does not wait for GPU work to finish.

### Q2. Why is full step time more useful than forward time alone?

Discuss which parts of training are missing from a forward-only measurement.
Use your benchmark rows to explain why `full_step_*` is the better number for training cost.

### Q3. How do sequence length and batch size affect tokens/sec and memory?

Compare rows where only one variable changes.
Report the trend you observe for tokens/sec and peak memory, and connect it to the attention shapes in your model.

### Q4. How do naive attention and SDPA compare under matched configs?

Only compare rows with the same precision, batch size, sequence length, model size, and number of heads.
State whether SDPA was faster/slower or used more/less memory in your run, and give a plausible explanation.

### Q5. What changes under fp32 vs bf16/fp16?

Compare the precision rows that your hardware supports.
If your run has no CUDA mixed-precision rows, say so explicitly and explain what was skipped.

### Q6. What does this benchmark not measure?
