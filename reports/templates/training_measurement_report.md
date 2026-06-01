# Training Measurement Report

## Command and Hardware

- command:
- output CSV:
- output markdown:
- device:
- GPU model:
- CUDA available:
- precision rows included:
- warmup:
- measured steps:

## Attention Backend Correctness Check

- command:
- output JSON:
- loss_naive:
- loss_sdpa:
- abs_loss_diff:
- max_abs_diff_logits:
- passed:

## Benchmark Table

Paste or summarize the key rows from `reports/benchmarks/final_systems.csv`.

| attention | precision | batch_size | seq_len | full_step_mean_sec | full_step_median_sec | full_step_p90_sec | tokens_per_sec | peak_memory_bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  |  |  |  |  |  |  |

## Naive vs SDPA Matched Comparison

Compare only rows with the same precision, batch size, sequence length, d_model, layers, and heads.

- matched rows used:
- SDPA faster/slower:
- memory difference:
- likely explanation:

## Precision Comparison

- fp32 result:
- bf16/fp16 result, or skipped reason:
- speed difference:
- memory difference:
- numerical caveat:

## Seq_len / Batch_size Trend

- seq_len trend:
- batch_size trend:
- tokens/sec trend:
- peak memory trend:

## Answers to Q1-Q6

### Q1. CUDA warmup and synchronize


### Q2. Full step time vs forward time


### Q3. Seq_len and batch_size effects


### Q4. Naive attention vs SDPA


### Q5. fp32 vs bf16/fp16


### Q6. What this benchmark does not measure


## Limitations

Mention synthetic random tokens, no DataLoader/memmap IO, no tokenizer cost, no checkpoint/eval/generation overhead, and no leaderboard claim.
