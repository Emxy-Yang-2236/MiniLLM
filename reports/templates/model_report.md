# Model Report

## Artifact Paths

- tokenizer JSON:
- tokenizer manifest:
- encoded TinyStories train manifest:
- encoded TinyStories valid manifest:
- pretraining checkpoint:
- SFT checkpoint:
- pretraining samples:
- SFT before/after samples:
- SFT evaluation JSON:
- metrics summary:
- training curves:
- training measurement report:

## Test Summary

Report the commands you ran and whether they passed.

```bash
uv run pytest
uv run pytest ../shared/tests
uv run pytest ../shared/tests/cs336_a1_exact
uv run python scripts/run_student_pipeline.py --mode smoke --device cpu
```

### SFT Discussion

Answer briefly:

- What changed after SFT?
- Which task improved the most?
- Did SFT hurt any behavior from pretraining?
- Did the model appear to learn the response format, the task content, or both?

## Training Curves

Include `outputs/release_candidate/training_curves.svg`.

Answer briefly:

- Did pretraining train loss and validation loss decrease?
- Did SFT train loss and validation loss decrease?
- Is there evidence of overfitting or underfitting?
- Which curve is more important for your release verdict, and why?

## Failure Analysis

List at least three concrete limitations observed in your run.
