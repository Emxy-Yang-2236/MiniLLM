# MiniLLM Student Release Dataset

This is the fixed dataset used by the main MiniLLM student pipeline.

## Source And Attribution

- Pretraining text: the full TinyStories V2 GPT-4 train/valid text files from Hugging Face.
- Train source: `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt`
- Valid source: `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt`
- TinyStories license: `CDLA-Sharing-1.0`.
- SFT/eval text: fixed assignment-authored prompt-response JSONL examples for template-conditioned short stories and single-label TinyStories-style classification. Arithmetic is not part of the main SFT demo. These examples are committed with the assignment and are not generated at runtime.

## Local Files

- train: `data/raw/tinystories/TinyStoriesV2-GPT4-train.txt`
- valid: `data/raw/tinystories/TinyStoriesV2-GPT4-valid.txt`

The authoritative file counts, token counts, byte counts, SHA256 hashes, source URLs, and split names are in `manifest.json`.
