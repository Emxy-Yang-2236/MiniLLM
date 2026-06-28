# Assignment Spec

> This document is the formal assignment requirement.

MiniLLM is a four-week project. You will implement a tokenizer, a Transformer language model, the core training utilities, TinyStories pretraining, a small SFT demo, and a final measurement/report package.

The starter code lives in `release/`. Unless stated otherwise, run commands from `release/`.

## Week 1: Tokenizer And Basic Layers

Week 1 builds the pieces that turn text into token IDs and token IDs into vectors.

### Files To Edit

- `release/minillm/tokenizer/bpe.py`
  - train a byte-level BPE tokenizer;
  - implement GPT-2-like pretokenization;
  - handle `<|endoftext|>` as the official special token;
  - implement encode/decode, iterable encode, batch helpers, save/load compatibility.
- `release/minillm/model/layers.py`
  - implement `Linear`, `Embedding`, `RMSNorm`, `SwiGLU`, `RotaryPositionalEmbedding`, and `softmax`;
  - do not implement `cross_entropy` yet; that is part of Week 2.

### Tests To Pass

Tokenizer tests:

```bash
python -m pytest -q ../shared/tests/test_tokenizer*.py
python -m pytest -q ../shared/tests/test_cs336_tokenizer_contract.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_tokenizer.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_train_bpe.py
```

Layer tests:

```bash
python -m pytest -q ../shared/tests/test_layers.py
python -m pytest -q ../shared/tests/test_cs336_layers_contract.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_nn_utils.py
```

After tests pass, you should train a small tokenizer as described in the tutorial.


## Week 2: Transformer LM And Training Utilities

Week 2 builds the language model and the reusable training utilities.

### Files To Edit

- `release/minillm/model/layers.py`
  - implement `cross_entropy`.
- `release/minillm/model/attention.py`
  - implement scaled dot-product attention and causal multi-head self-attention.
- `release/minillm/model/transformer.py`
  - implement the pre-norm Transformer block and `TransformerLM`.
- `release/minillm/model/generation.py`
  - implement greedy, temperature, top-k, top-p generation and `<|endoftext|>` stopping.
- `release/minillm/data/pretrain_dataset.py`
  - implement `get_batch`.
  - The larger memmap dataset classes are provided infrastructure.
- `release/minillm/train/optim.py`
  - implement AdamW and global gradient clipping.
- `release/minillm/train/schedules.py`
  - implement cosine learning-rate warmup.
- `release/minillm/train/checkpoint.py`
  - implement save/load checkpoint helpers.

### Tests To Pass

Model and generation tests:

```bash
python -m pytest -q ../shared/tests/test_model.py ../shared/tests/test_generation.py
python -m pytest -q ../shared/tests/test_cs336_transformer_contract.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_model.py
```

Training utility tests:

```bash
python -m pytest -q ../shared/tests/test_optimizer.py ../shared/tests/test_train_state.py
python -m pytest -q ../shared/tests/test_cs336_training_utils_contract.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_optimizer.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_data.py
```

Checkpoint tests:

```bash
python -m pytest -q ../shared/tests/test_checkpoint.py
python -m pytest -q ../shared/tests/test_cs336_checkpoint_contract.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_serialization.py
```


## Week 3: Pretraining And Tiny-SFT

Week 3 connects the pieces into the main language-modeling workflow. You will run TinyStories pretraining, generation, Tiny-SFT, and held-out SFT evaluation.

### Files To Edit

- `release/minillm/train/pretrain.py`
  - implement validation-loss evaluation;
  - implement one pretraining optimizer step.
- `release/minillm/data/sft_dataset.py`
  - implement response-only SFT tensors in `make_sft_tensors`.

Release-data manifest code, SFT training orchestration, SFT eval/report glue, and full pipeline orchestration are provided.

### Tests To Pass

Pretraining/data tests:

```bash
python -m pytest -q ../shared/tests/test_memmap_data.py ../shared/tests/test_training.py
```

SFT tests:

```bash
python -m pytest -q ../shared/tests/test_sft_data.py ../shared/tests/test_sft_loss.py
python -m pytest -q ../shared/tests/test_sft_release_tasks.py ../shared/tests/test_eval.py
```

After tests pass, you should do pretrain and SFT training as described in tutorials.


## Week 4: Final Evaluation And Training Measurement

Week 4 is mostly running, inspecting, and reporting. The Training Measurement Mini-lab is provided as a measurement tool, not a systems implementation assignment.

### No Files To Edit


### Final Pipeline And Reports

Run the full student pipeline, inspect the generated artifacts, and complete the Training Measurement Mini-lab. The detailed Week 4 commands and output files are in [Training Measurement Mini-lab.md](./Training%20Measurement%20Mini-lab.md) and [training_measurement_report.md](../reports/templates/training_measurement_report.md).

The pipeline also writes an automatic `PASS` / `WEAK PASS` / `FAIL` sanity verdict in `outputs/release_candidate/metrics_summary.json`. Treat it as a quick check of losses, SFT metrics, and benchmark rows; your report should still explain the actual artifacts and examples.


## Grading Policy

- CS336 A1-compatible unit tests: 30 pts
- MiniLLM integration tests: 30 pts
- pretraining run + generation samples: 15 pts
- tiny-SFT demo + held-out eval: 15 pts
- training measurement mini-lab report: 5 pts
- Bonus: 5 pts each

Failed tests will incur a proportional deduction based on their share of the total test suite. Points do not stack within the same bonus category.

## Bonus Options

You may choose one bonus option, or discuss a different option with the TAs (encouraged!):

- Implement an O(nlogn) algorithm for applying merges in tokenizer decoder;
- try another SFT task;
- try another model architecture;
- try another set of hyperparameters.

## Allowed Dependencies

Allow:

- Python;
- PyTorch tensor/autograd;
- `torch.nn.Parameter`;
- `torch.nn` container classes such as `Module`, `ModuleList`, `Sequential`;
- `torch.optim.Optimizer` base class;
- PyTorch utilities outside `torch.nn`, `torch.nn.functional`, and `torch.optim`;
- `numpy`;
- `regex`;
- `tqdm`, `yaml`, basic utilities;
- `tiktoken` only inside tests.

Banned:

- layers in `torch.nn`, such as `Linear`, `Embedding`, `LayerNorm`, `Transformer`;
- functions in `torch.nn.functional`, such as `linear`, `embedding`, `scaled_dot_product_attention`;
- optimizers in `torch.optim`, such as `AdamW`;
- Hugging Face Trainer, `transformers.AutoModel`, TRL, accelerate, Triton, DDP, FSDP, ZeRO, GRPO/RL.

## Submission Artifacts

Before submission, after all required implementation is complete, run the full public test suite from `release/`:

```bash
python -m pytest -q ../shared/tests
```

Your final submission should include:

- your completed code;
- a short model report;
- a short Training Measurement Mini-lab report.

Compress the three artifacts into a `.zip` file for submission.

## Report Requirements

The report format is flexible. You do not have to follow a fixed template, but every important claim should point to an artifact path, metric, sample, or command output.
Refer to [model_report.md](../reports/templates/model_report.md) and [training_measurement_report.md](../reports/templates/training_measurement_report.md) for required content in your report.
