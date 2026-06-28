# Beginner Guide

> This document introduces the general layout of the project.

**Only tutorials about specific implementation tasks will be provided.**
Other prerequisites, such as "What is deep learning?" or "Why does a Transformer look like this?", are left for your own exploration.
Discussions and questions about related topics are welcome.

For each task, you will be given some questions to think about.
These questions mainly focus on why each component matters and why it was designed in that way.
You **do not** need to answer these questions formally.
They are prompts for reflection, intended to help you build a broader view of the field and develop some intuition.

This is only an introduction.

- See [assignment_spec.md](assignment_spec.md) for weekly tasks, tests, grading policy, and submission requirements.
- See [Index_tutorial.md](Index_tutorial.md) for the recommended tutorial order.

Now, let's look at this repository more carefully.

## About This Repo

This repository is organized around two parallel implementations:

- `release/` is the version you work in. It contains the starter `minillm` package, runnable scripts, YAML configs, and TODOs.
- `reference/` is the instructor implementation. It mirrors the release layout, but students do not need to use it unless course staff explicitly says so.
- `shared/` contains public tests and fixtures. The tests call adapter functions so the same tests can run against either `release/` or `reference/`.
- `data/` contains TinyStories raw files and the fixed MiniLLM SFT/eval release data. Do not edit dataset files by hand.
- `docs/` contains the assignment specification and tutorial notes.

Most commands are run from inside `release/`. For example, public tests are usually run as:

```bash
cd release
python -m pytest -q ../shared/tests
```

The main generated directories are `release/runs/` and `release/outputs/`. They are created by training, evaluation, generation, and benchmark scripts. They are not source code.

## Environment Setup

The repository includes two conda environment files:

- `environment-cpu.yml`: use this on laptops, CPU machines, or Apple Silicon Macs using MPS.
- `environment-gpu.yml`: use this on Linux machines with an NVIDIA GPU and compatible CUDA driver.

Create one environment from the repository root:

```bash
conda env create -f environment-cpu.yml
conda activate cs336-minillm
```

or, on a CUDA machine (**recommended**):

```bash
conda env create -f environment-gpu.yml
conda activate cs336-minillm
```

On Apple Silicon, keep the CPU environment and run with `--device mps` to force MPS, or `--device auto` to choose CUDA, then MPS, then CPU. MPS runs should use fp32; CUDA-only mixed-precision and CUDA memory results should be marked skipped.

Then run commands from `release/`:

```bash
cd release
python -m pytest -q ../shared/tests
```

The provided plotting script uses only the Python standard library, so no extra plotting package is required.

## Paths And Configs

YAML configs in `release/configs/` use paths relative to the `release/` directory.
The matching reference configs use the same relative layout from `reference/`.
That is why dataset paths usually start with `../data/`, while generated paths usually start with `runs/` or `outputs/`.

Common path conventions:

- `../data/raw/tinystories/`: full TinyStories train/valid text files.
- `../data/full_release/`: fixed SFT/eval files and dataset manifest.
- `runs/...`: generated tokenizer files, encoded `.bin` files, checkpoints, and JSONL metrics.
- `outputs/...`: generated samples, before/after demos, evaluation JSON, benchmark CSV/MD, and run reports.

If you change tokenizer behavior, regenerate the tokenizer, encoded `.bin` files, checkpoints, and demo outputs so that manifests and artifacts match.

The high-level workflow is:

```text
TinyStories text
 -> train tokenizer
 -> encode train/valid text
 -> pretrain Transformer LM
 -> generate samples
 -> run SFT from your own pretrain checkpoint
 -> compare before/after SFT outputs
 -> run the Training Measurement Mini-lab
 -> write the final report
```

Read `assignment_spec.md` for what must be submitted. Read `Index_tutorial.md` for the recommended order of implementation.
