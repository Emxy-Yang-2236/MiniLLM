# Tiny-SFT

After pretraining, your Transformer LM has learned to continue TinyStories-like text.
That is useful, but it does not yet mean the model knows how to answer a prompt in the format we want.
Tiny-SFT is the small supervised fine-tuning stage that teaches the model a narrow response behavior.

## What Is SFT?

Supervised fine-tuning means training a language model on examples of the form: `prompt -> desired response`.
The model still uses the same next-token prediction objective.
The difference is that the text now has an instruction-like prompt followed by a target response.
In larger instruction-following systems, SFT is often the first stage that teaches the model to imitate demonstrations before more complex alignment methods are used [1].
Modern SFT libraries also treat prompt/response formatting and response-only loss masking as central pieces of the training setup [2].
MiniLLM uses those ideas, but not the high-level libraries.
You will implement the small tensor-building step yourself using PyTorch tensors and the tokenizer you already trained.

---

MiniLLM SFT is a template-conditioned demo, not a general assistant demo.
The released SFT data has two task types: `story_template` writes a short on-topic TinyStories-style story; `label_template` chooses one short feeling label.
The prompt format is the same in train, valid, eval, and fixed before/after prompts:

```text
### Task: story
### Prompt:
Write a short story about a brave mouse.
Goal: help a friend.
Event: found a lost button.
Feeling: proud and happy.
### Response:
```

The target response is short and template-like:

```text
Once there was a brave mouse named Lily.
Lily wanted to help a friend.
One day, Lily found a lost button.
In the end, Lily felt proud and happy.
```

For label examples, the model should output only the label:

```text
### Task: label
### Prompt:
Choose one feeling label: happy, sad, scared, angry, kind, lonely.
Text: Tim lost a toy at the park and cried softly.
Answer:
### Response:
```

```text
sad
```

## Released SFT Data

The release data is fixed; you should not generate new main SFT data at runtime.

```text
data/full_release/sft/train.jsonl
data/full_release/sft/valid.jsonl
data/full_release/eval/sft_eval.jsonl
data/full_release/sft/fixed_prompts.jsonl
```

Each JSONL row contains `task_type`, `prompt`, and `response`.
The fixed prompts drive `sft_before_after.md`; the eval split is held out for automatic metrics.

## SFT Loss

SFT still uses next-token prediction, but the loss should apply only to the response.
The prompt is context, not something the model should be graded on reproducing.
Use `-100` for labels that should be ignored by cross-entropy.

For:

```text
tokens:       [p0 p1 p2 r0 r1 <|endoftext|>]
input_ids:    p0 p1 p2 r0 r1
labels:       p1 p2 r0 r1 <|endoftext|>
masked labels:-100 -100 r0 r1 <|endoftext|>
```

The rule is: labels that predict prompt tokens and padding are `-100`; labels that predict response tokens keep their token IDs.

## What You Implement

The SFT training loop, checkpointing, evaluation, and report generation are provided.
Your task is only the tensor construction step in `release/minillm/data/sft_dataset.py` (def make_sft_tensors). Your implementation should:

- encode the prompt and response;
- append `<|endoftext|>` token to the response;
- truncate to the configured sequence length;
- create `input_ids = seq[:-1]`;
- create `labels = seq[1:]`;
- replace labels that predict prompt tokens with `-100`;
- pad `input_ids`, `labels`, and `attention_mask`.

## SFT Training Flow

SFT must start from your own pretrained checkpoint. The pipeline looks like:

```text
pretrained checkpoint
  -> fixed SFT train/valid data
  -> response-only loss
  -> SFT checkpoint
  -> before/after samples
  -> held-out eval
```

Run the full student pipeline from `release/`:

```bash
python scripts/run_student_pipeline.py --mode student --device cuda
```

Use `--device mps` on Apple Silicon, or `--device cpu` if no accelerator is available.
To debug only SFT after pretraining exists:

```bash
python scripts/run_sft_demo.py \
  --base_ckpt runs/student_pipeline/student/pretrain/checkpoint_last.pt \
  --config configs/sft_student.yaml \
  --device cuda
```

## Outputs To Inspect

```text
runs/student_pipeline/student/sft/checkpoint_last.pt
outputs/release_candidate/metrics_sft.jsonl
outputs/release_candidate/sft_before_after.md
outputs/release_candidate/eval_sft.json
outputs/release_candidate/run_summary.md
```

`outputs/release_candidate/sft_before_after.md` shows that SFT changes the model from plain continuation toward the requested story or label template.
The eval file reports:

```text
story_template/template_followed
story_template/topic_match
story_template/sentence_count_ok
label_template/exact_match
label_template/no_extra_prose
```

Loss should generally decrease, but before/after samples are the easiest way to judge the demo.

## Checks

From `release/`:

```bash
python -m pytest -q ../shared/tests/test_sft_data.py ../shared/tests/test_sft_loss.py
python -m pytest -q ../shared/tests/test_sft_release_tasks.py ../shared/tests/test_eval.py
python scripts/run_student_pipeline.py --mode smoke --device cpu
```

Smoke mode is a correctness check.
Use `--mode student` for the final assignment report.

## References

[1] Long Ouyang et al. "Training language models to follow instructions with human feedback." arXiv:2203.02155, 2022. https://arxiv.org/abs/2203.02155

[2] Hugging Face TRL documentation, "Supervised Fine-tuning Trainer." https://huggingface.co/docs/trl/sft_trainer
