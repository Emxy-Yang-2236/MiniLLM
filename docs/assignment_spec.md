# Assignment Spec

> This document is a **formal assignment requirement**.

## Week 1: Tokenizer And Basic Layers

Implement a byte-level BPE tokenizer with GPT-2-like pretokenization and the small neural-network layers used by the Transformer.

Files to edit:
- `release/minillm/tokenizer/bpe.py`
- `release/minillm/model/layers.py`

Week 1 validation order:

1. Run the Week 1 tests below.
2. After the tests pass, train a tokenizer once and check that the tokenizer artifacts are written.
3. Keep the tokenizer artifacts; later weeks will use a trained tokenizer to encode text for language-model pretraining.

Tests to pass first:

```bash
cd release
python -m pytest -q ../shared/tests/test_tokenizer*.py
python -m pytest -q ../shared/tests/test_layers.py
python -m pytest -q ../shared/tests/test_cs336_tokenizer_contract.py
python -m pytest -q ../shared/tests/test_cs336_layers_contract.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_tokenizer.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_train_bpe.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_nn_utils.py
```

Then train a tokenizer:

```bash
python scripts/train_tokenizer.py --config configs/tokenizer_smoke.yaml
```

This is a small Week 1 training check, not the final full-data tokenizer run. It should write:
- `runs/student_pipeline/smoke/tokenizer/tokenizer.json`
- `runs/student_pipeline/smoke/tokenizer/tokenizer_manifest.json`

The full release dataset/config will be specified separately. The important Week 1 point is that your tokenizer implementation can both pass correctness tests and produce a saved tokenizer artifact.

## Week 2: Attention, Transformer LM, And Training Utilities

Implement causal self-attention, the full Transformer language model, generation, and training utilities. Generation includes greedy decoding, temperature sampling, top-k sampling, top-p sampling, and `<|endoftext|>` stopping.

Files to edit:
- `release/minillm/model/attention.py`
- `release/minillm/model/transformer.py`
- `release/minillm/model/generation.py`
- `release/minillm/data/pretrain_dataset.py`
- `release/minillm/train/optim.py`
- `release/minillm/train/schedules.py`
- `release/minillm/train/checkpoint.py`

Tests to pass:

```bash
cd release
python -m pytest -q ../shared/tests/test_model.py ../shared/tests/test_generation.py
python -m pytest -q ../shared/tests/test_optimizer.py ../shared/tests/test_checkpoint.py ../shared/tests/test_train_state.py
python -m pytest -q ../shared/tests/test_cs336_transformer_contract.py
python -m pytest -q ../shared/tests/test_cs336_training_utils_contract.py ../shared/tests/test_cs336_checkpoint_contract.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_model.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_optimizer.py ../shared/tests/cs336_a1_exact/test_serialization.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_data.py
```

Do not use ready-made PyTorch layers or optimizers listed in the banned dependency section.

## Week 3: Pretraining And Tiny-SFT

Run the main language-modeling workflow.

Files to edit:
- `release/minillm/train/pretrain.py`
- `release/minillm/data/sft_dataset.py`

Required outcomes:
- train a tokenizer on TinyStories train text;
- encode TinyStories train/valid text to binary token files with manifests;
- pretrain your Transformer LM on TinyStories;
- generate pretraining samples;
- run SFT from your own pretraining checkpoint;
- evaluate SFT on the fixed release SFT/eval data.


SFT is a template-conditioned behavior demo. It should show short story-template following and single-label classification. It is not meant to prove arithmetic reasoning or general assistant behavior.

Tests to pass:

```bash
cd release
python -m pytest -q ../shared/tests/test_memmap_data.py ../shared/tests/test_training.py
python -m pytest -q ../shared/tests/test_sft_data.py ../shared/tests/test_sft_loss.py
python -m pytest -q ../shared/tests/test_sft_release_tasks.py ../shared/tests/test_eval.py
```

Pipeline check:

```bash
python scripts/verify_student_release_data.py --dataset_dir ../data/full_release
python scripts/run_student_pipeline.py --mode smoke --device cpu
```

If you only want to debug the pretraining loop after tokenizer and encoded `.bin` files exist, run:

```bash
python scripts/train_pretrain.py \
  --config configs/pretrain_smoke.yaml \
  --max_steps 4 \
  --device cpu
```

Smoke mode should produce:
- `runs/student_pipeline/smoke/tokenizer/tokenizer.json`
- `runs/student_pipeline/smoke/tokenizer/pretrain_train.bin`
- `runs/student_pipeline/smoke/pretrain/checkpoint_last.pt`
- `runs/student_pipeline/smoke/sft/checkpoint_last.pt`
- `outputs/smoke/pretrain_samples.md`
- `outputs/smoke/sft_before_after.md`
- `outputs/smoke/eval_sft.json`
- `outputs/smoke/metrics_pretrain.jsonl`
- `outputs/smoke/metrics_sft.jsonl`

## Week 4: Final Evaluation And Training Measurement

Finish evaluation, reporting, and the Training Measurement Mini-lab.

Files to edit:
- usually none in provided infrastructure;
- your final report files;
- small fixes in Week 1-3 modules if pipeline evidence exposes bugs.

Required outcomes:
- fixed before/after SFT samples;
- final metrics summary;
- training loss curves generated from JSONL metrics;
- optional next-token top-k inspection from a trained checkpoint;
- held-out SFT eval results;
- benchmark sanity output from the provided training-measurement scripts;
- one short report explaining model behavior, limitations, and benchmark results.

Students run and interpret the Training Measurement Mini-lab. You are not expected to implement systems optimizations.

The final pipeline and report-generation scripts are provided infrastructure. You may inspect them to understand the required artifacts, but they are not a main implementation target.

Tests to pass:

```bash
cd release
python -m pytest -q ../shared/tests/test_systems_smoke.py ../shared/tests/test_systems_analysis.py
python -m pytest -q ../shared/tests
```

Run the following command to get some artifacts for your system report and your training curves:

```bash
python scripts/run_student_pipeline.py --mode student --device cuda
python scripts/plot_training_curves.py
```

To inspect next-token probabilities from a trained checkpoint, run:

```bash
python scripts/inspect_next_token.py \
  --ckpt runs/student_pipeline/student/pretrain/checkpoint_last.pt \
  --prompt "Once upon a time" \
  --top_k 8 \
  --steps 20 \
  --mode greedy
```

Use `--mode choose` if you want to pick one of the displayed top-k tokens at each step. This script is provided infrastructure; it relies on your tokenizer, model forward pass, and checkpoint.

If CUDA is unavailable, use `--device mps` on Apple Silicon or `--device cpu` elsewhere. MPS runs use fp32 for this project; clearly state that CUDA mixed-precision and CUDA memory results were skipped.

The full pipeline writes the main artifacts under `outputs/release_candidate/`, including:
- `run_summary.md`
- `pipeline_run_report.md`
- `metrics_summary.json`
- `training_curves.svg`
- `pretrain_samples.md`
- `sft_before_after.md`
- `eval_sft.json`
- `benchmark_sanity.csv`
- `benchmark_sanity.md`
- `run_artifacts_manifest.json`

For the final Training Measurement Mini-lab table, also run:

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

Use `outputs/final_systems.csv`, `outputs/final_systems.md`, and `../reports/templates/training_measurement_report.md` to answer the six Training Measurement questions. On MPS, run the standalone sweep with `--device mps --precisions fp32`.

## Grading Policy

- CS336 A1-compatible unit tests: 30 pts
- MiniLLM integration tests: 30 pts
- pretraining run + generation samples: 15 pts
- tiny-SFT demo + held-out eval: 15 pts
- training measurement mini-lab report: 5 pts
- Bonus (each): 5 pts

Failed tests will incur a proportional deduction based on their share of the total test suite.
Each bonus category is worth 5 pts. Points do not stack within the same category.

## BONUS Options

You may choose one of the bonus options listed below.
If you have any other ideas for a bonus, feel free to discuss them with the TAs.
- try another SFT task
- try another model architecture (For example, modifying the Transformer block or implementing an RNN to evaluate and compare their performance.)
- try another set of hyperparameters

## Late Policy

Follow the course late-submission policy announced by the teaching staff.

## Allowed Dependencies

Allow:
- Python
- PyTorch tensor/autograd
- `torch.nn.Parameter`
- `torch.nn` container classes such as `Module`, `ModuleList`, `Sequential`
- `torch.optim.Optimizer` base class
- PyTorch utilities outside `torch.nn`, `torch.nn.functional`, and `torch.optim`
- `numpy`
- `regex`
- `tqdm`, `yaml`, basic utilities
- `tiktoken` only inside tests

Banned:
- Layers in `torch.nn`, such as `Linear`, `Embedding`, `LayerNorm`, `Transformer`
- Functions in `torch.nn.functional`, such as `linear`, `embedding`, `scaled_dot_product_attention`
- Optimizers in `torch.optim`, such as `AdamW`

## Submission Artifacts

Your final submission should include:
- your completed code;
- a short report of your pretraining and SFT result, see [model_report.md](../reports/templates/model_report.md) for reference;
- a short report of the training measurement mini-lab, see [training_measurement_report.md](../reports/templates/training_measurement_report.md) for reference.

Compress the three artifacts into a `.zip` file for submission.
