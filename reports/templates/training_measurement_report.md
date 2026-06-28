# Training Measurement Report Outline

This is an optional outline, not a required format.
Use your own benchmark rows from `outputs/final_systems.csv` and `outputs/final_systems.md`.

Suggested contents:

- Command and hardware: exact command, device, GPU or CPU/MPS details, precision rows included, warmup, measured steps.
- Output paths: final CSV, final markdown summary, optional attention-correctness check output.
- Benchmark table: a small subset of rows that supports your claims.
- Naive vs SDPA comparison: compare only matched rows with the same precision, batch size, sequence length, model size, layers, and heads.
- Precision comparison: compare fp32 and bf16/fp16 if CUDA supports them; otherwise state what was skipped.
- Sequence length and batch size trends: explain tokens/sec and memory movement using your rows.
- Q1-Q6 answers from `docs/Training Measurement Mini-lab.md`.
- Limitations: mention synthetic random tokens, no DataLoader/memmap I/O, no tokenizer cost, no checkpoint/eval/generation overhead, and no speed-contest claim.
