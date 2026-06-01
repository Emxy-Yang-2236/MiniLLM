from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
IMPL = os.environ.get("MINILLM_IMPL", "reference")
IMPL_PATH = ROOT / IMPL
if not IMPL_PATH.exists():
    raise RuntimeError(f"unknown MINILLM_IMPL={IMPL!r}; expected {IMPL_PATH}")
sys.path.insert(0, str(IMPL_PATH))


def make_config(**overrides):
    from minillm.model.config import TransformerConfig

    data = {
        "vocab_size": 300,
        "context_length": 16,
        "d_model": 32,
        "d_ff": 64,
        "num_layers": 1,
        "num_heads": 4,
        "rope_theta": 10000.0,
        "dropout": 0.0,
        "attention_backend": "naive",
    }
    data.update(overrides)
    return TransformerConfig(**data)


def tokenizer_module():
    from minillm.tokenizer import bpe

    return bpe


def train_bpe(texts, vocab_size=280, min_frequency=2, special_tokens=None, **kwargs):
    return tokenizer_module().train_bpe(
        texts, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens, **kwargs
    )


def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    text = Path(input_path).read_text(encoding="utf-8")
    min_frequency = kwargs.pop("min_frequency", 1)
    tok = tokenizer_module().train_bpe(
        text,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        **kwargs,
    )
    return tok.vocab, tok.merges


def get_tokenizer(vocab, merges, special_tokens=None, **kwargs):
    module = tokenizer_module()
    return module.ByteBPETokenizer(
        vocab={int(k): bytes(v) for k, v in vocab.items()},
        merges=[(bytes(left), bytes(right)) for left, right in merges],
        special_tokens=list(special_tokens or []),
        pretokenizer=kwargs.get("pretokenizer", "gpt2_like"),
        tie_break=kwargs.get("tie_break", "max"),
    )


def load_tokenizer(path):
    return tokenizer_module().ByteBPETokenizer.load(path)


def run_encode(tokenizer, text):
    return tokenizer.encode(text)


def run_decode(tokenizer, ids, **kwargs):
    return tokenizer.decode(ids, **kwargs)


def run_encode_iterable(tokenizer, texts):
    return list(tokenizer.encode_iterable(texts))


def run_save_tokenizer(tokenizer, path):
    return tokenizer.save(path)


def run_load_tokenizer(path):
    return load_tokenizer(path)


def get_layers():
    from minillm.model import layers

    return layers


def run_linear(d_in, d_out, weights, in_features):
    layers = get_layers()
    layer = layers.Linear(d_in, d_out, device=in_features.device, dtype=in_features.dtype)
    with torch.no_grad():
        layer.weight.copy_(weights.to(device=in_features.device, dtype=in_features.dtype))
    return layer(in_features)


def run_embedding(vocab_size, d_model, weights, token_ids):
    layers = get_layers()
    layer = layers.Embedding(vocab_size, d_model, device=token_ids.device, dtype=weights.dtype)
    with torch.no_grad():
        layer.weight.copy_(weights.to(device=token_ids.device, dtype=weights.dtype))
    return layer(token_ids)


def run_rmsnorm(d_model, eps, weights, in_features):
    layers = get_layers()
    layer = layers.RMSNorm(d_model, eps=eps, device=in_features.device, dtype=in_features.dtype)
    with torch.no_grad():
        layer.weight.copy_(weights.to(device=in_features.device, dtype=in_features.dtype))
    return layer(in_features)


def run_silu(in_features):
    return torch.nn.functional.silu(in_features)


def run_softmax(in_features, dim):
    layers = get_layers()
    return layers.softmax(in_features, dim=dim)


def run_cross_entropy(inputs, targets):
    layers = get_layers()
    return layers.cross_entropy(inputs, targets)


def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    layers = get_layers()
    layer = layers.SwiGLU(d_model, d_ff, device=in_features.device, dtype=in_features.dtype)
    with torch.no_grad():
        layer.w1.weight.copy_(w1_weight.to(device=in_features.device, dtype=in_features.dtype))
        layer.w2.weight.copy_(w2_weight.to(device=in_features.device, dtype=in_features.dtype))
        layer.w3.weight.copy_(w3_weight.to(device=in_features.device, dtype=in_features.dtype))
    return layer(in_features)


def run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):
    layers = get_layers()
    rope = layers.RotaryPositionalEmbedding(
        theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device
    )
    return rope(in_query_or_key, token_positions.to(in_query_or_key.device))


def run_scaled_dot_product_attention(Q, K, V, mask=None):
    from minillm.model.attention import scaled_dot_product_attention

    return scaled_dot_product_attention(Q, K, V, mask)


def _copy_attention_weights(attn, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, dtype, device):
    with torch.no_grad():
        attn.q_proj.weight.copy_(q_proj_weight.to(device=device, dtype=dtype))
        attn.k_proj.weight.copy_(k_proj_weight.to(device=device, dtype=dtype))
        attn.v_proj.weight.copy_(v_proj_weight.to(device=device, dtype=dtype))
        attn.output_proj.weight.copy_(o_proj_weight.to(device=device, dtype=dtype))


def run_multihead_self_attention(
    d_model,
    num_heads,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    in_features,
):
    from minillm.model.attention import MultiHeadSelfAttention

    attn = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=in_features.shape[-2],
        use_rope=False,
    ).to(device=in_features.device, dtype=in_features.dtype)
    _copy_attention_weights(
        attn,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        dtype=in_features.dtype,
        device=in_features.device,
    )
    return attn(in_features)


def run_multihead_self_attention_with_rope(
    d_model,
    num_heads,
    max_seq_len,
    theta,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    in_features,
    token_positions=None,
):
    from minillm.model.attention import MultiHeadSelfAttention

    attn = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        use_rope=True,
    ).to(device=in_features.device, dtype=in_features.dtype)
    _copy_attention_weights(
        attn,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        dtype=in_features.dtype,
        device=in_features.device,
    )
    if token_positions is not None:
        token_positions = token_positions.to(in_features.device)
    return attn(in_features, token_positions=token_positions)


def run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features):
    from minillm.model.transformer import TransformerBlock

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
    ).to(device=in_features.device, dtype=in_features.dtype)
    block.load_state_dict({k: v.to(device=in_features.device, dtype=in_features.dtype) for k, v in weights.items()})
    return block(in_features)


def run_transformer_lm(
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    rope_theta,
    weights,
    in_indices,
):
    from minillm.model.transformer import TransformerLM

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        tie_embeddings=False,
    ).to(in_indices.device)
    model.load_state_dict({k: v.to(in_indices.device) for k, v in weights.items()})
    return model(in_indices)["logits"]


def build_model(cfg=None):
    from minillm.model.transformer import TransformerLM

    return TransformerLM(cfg or make_config())


def model_summary(model):
    from minillm.model.transformer import model_summary

    return model_summary(model)


def generate_text(model, tokenizer, prompt, **kwargs):
    from minillm.model.generation import generate

    return generate(model, tokenizer, prompt, **kwargs)


def default_stop_ids(tokenizer):
    from minillm.model.generation import default_stop_ids

    return default_stop_ids(tokenizer)


def generation_config(**kwargs):
    from minillm.model.generation import GenerationConfig

    return GenerationConfig(**kwargs)


def generate_ids(model, input_ids, cfg):
    from minillm.model.generation import generate_ids

    return generate_ids(model, input_ids, cfg)


def pretrain_dataset(path, tokenizer=None, seq_len=16):
    from minillm.data.pretrain_dataset import PretrainDataset

    return PretrainDataset(path, tokenizer, seq_len)


def encode_text_file(input_path, tokenizer, output_path):
    from minillm.data.pretrain_dataset import encode_text_file

    return encode_text_file(input_path, tokenizer, output_path)


def encoded_manifest_path(path):
    from minillm.data.pretrain_dataset import encoded_manifest_path

    return encoded_manifest_path(path)


def validate_encoded_manifest(path, tokenizer_path):
    from minillm.data.pretrain_dataset import validate_encoded_manifest

    return validate_encoded_manifest(path, tokenizer_path)


def sft_dataset(path, tokenizer, seq_len):
    from minillm.data.sft_dataset import SFTDataset

    return SFTDataset(path, tokenizer, seq_len)


def train_pretrain(cfg, max_steps=None):
    from minillm.train.pretrain import train_pretrain as run

    return run(cfg, max_steps=max_steps)


def run_get_batch(dataset, batch_size, context_length, device):
    ids = torch.as_tensor(dataset, dtype=torch.long)
    if ids.ndim != 1:
        raise ValueError("dataset must be a 1D array of token ids")
    if len(ids) <= context_length:
        raise ValueError("dataset must be longer than context_length")
    starts = torch.randint(0, len(ids) - context_length, (batch_size,))
    x = torch.stack([ids[start : start + context_length] for start in starts])
    y = torch.stack([ids[start + 1 : start + context_length + 1] for start in starts])
    return x.to(device), y.to(device)


def build_optimizer(params, **kwargs):
    from minillm.train.optim import AdamW

    return AdamW(params, **kwargs)


def get_adamw_cls():
    from minillm.train.optim import AdamW

    return AdamW


def clip_grad_norm(parameters, max_norm):
    from minillm.train.optim import clip_grad_norm_

    return clip_grad_norm_(parameters, max_norm)


def run_gradient_clipping(parameters, max_l2_norm):
    return clip_grad_norm(parameters, max_l2_norm)


def make_scheduler(**kwargs):
    from minillm.train.schedules import WarmupScheduler

    return WarmupScheduler(**kwargs)


def run_get_lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    if warmup_iters > 0 and it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    denom = max(1, cosine_cycle_iters - warmup_iters)
    progress = (it - warmup_iters) / denom
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def save_checkpoint(
    path,
    model,
    optimizer=None,
    step=0,
    config=None,
    tokenizer_path=None,
    scheduler=None,
    train_state=None,
):
    from minillm.train.checkpoint import save_checkpoint as save

    return save(
        path,
        model,
        optimizer=optimizer,
        step=step,
        config=config,
        tokenizer_path=tokenizer_path,
        scheduler=scheduler,
        train_state=train_state,
    )


def run_save_checkpoint(model, optimizer, iteration, out):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": iteration,
            "step": iteration,
        },
        out,
    )


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, train_state=None, map_location="cpu"):
    from minillm.train.checkpoint import load_checkpoint as load

    return load(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_state=train_state,
        map_location=map_location,
    )


def run_load_checkpoint(src, model, optimizer):
    payload = torch.load(src, map_location="cpu", weights_only=False)
    model_state = payload.get("model_state", payload.get("model"))
    optimizer_state = payload.get("optimizer_state", payload.get("optimizer"))
    if model_state is None or optimizer_state is None:
        raise KeyError("checkpoint must contain model and optimizer state")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return int(payload.get("iteration", payload.get("step", 0)))


def train_state(**kwargs):
    from minillm.train.state import TrainState

    return TrainState(**kwargs)


def jsonl_logger(path):
    from minillm.train.state import JsonlLogger

    return JsonlLogger(path)


def create_optimizer_from_cfg(model, cfg):
    from minillm.train.state import create_optimizer

    return create_optimizer(model, cfg)


def create_scheduler_from_cfg(cfg, max_steps):
    from minillm.train.state import create_scheduler

    return create_scheduler(cfg, max_steps)


def apply_lr(optimizer, lr):
    from minillm.train.state import apply_lr

    return apply_lr(optimizer, lr)


def evaluate_arithmetic(model, tokenizer, path, **kwargs):
    from minillm.eval.arithmetic import evaluate_arithmetic as evaluate

    return evaluate(model, tokenizer, path, **kwargs)


def grade_arithmetic(text, answer):
    from minillm.eval.arithmetic import grade_response

    return grade_response(text, answer)


def story_format_score(text):
    from minillm.eval.story import story_format_score

    return story_format_score(text)


def grade_sft_response(text, row):
    from minillm.eval.sft import grade_sft_response

    return grade_sft_response(text, row)


def summarize_sft_scores(rows):
    from minillm.eval.sft import summarize_scores

    return summarize_scores(rows)


def build_sft_release_rows(seed=0):
    from minillm.data.release import build_sft_release_rows

    return build_sft_release_rows(seed=seed)


def summarize_story_samples(samples):
    from minillm.eval.story import summarize_story_samples

    return summarize_story_samples(samples)


def benchmark_step(cfg, attention="sdpa", device="cpu", warmup=0, steps=1, **kwargs):
    from minillm.systems.benchmark import benchmark_step as run

    return run(cfg, attention=attention, device=device, warmup=warmup, steps=steps, **kwargs)


def benchmark_sweep(cfg, **kwargs):
    from minillm.systems.benchmark import benchmark_sweep as run

    return run(cfg, **kwargs)


def check_attention_backend_correctness(cfg, **kwargs):
    from minillm.systems.benchmark import check_attention_backend_correctness as run

    return run(cfg, **kwargs)


def flatten_benchmark_row(row):
    from minillm.systems.benchmark import flatten_benchmark_row

    return flatten_benchmark_row(row)


def sft_collator(pad_id, label_pad_id=-100):
    from minillm.data.sft_dataset import SFTCollator

    return SFTCollator(pad_id=pad_id, label_pad_id=label_pad_id)


def random_block_dataset(path, seq_len, num_samples, seed=0):
    from minillm.data.pretrain_dataset import RandomBlockDataset

    return RandomBlockDataset(path, seq_len=seq_len, num_samples=num_samples, seed=seed)


def split_text_file(input_path, train_path, valid_path, valid_fraction=0.1):
    from minillm.data.pretrain_dataset import split_text_file

    return split_text_file(input_path, train_path, valid_path, valid_fraction=valid_fraction)


def student_release_dir():
    return ROOT / "data" / "full_release"


def dataset_path_for_split(dataset_dir, manifest, split):
    from minillm.data.release import dataset_path_for_split as resolve

    return resolve(dataset_dir, manifest, split)


def verify_student_release(path=None):
    from minillm.data.release import verify_manifest

    return verify_manifest(path or student_release_dir())


def tiny_overfit_batch(model, batch, **kwargs):
    from minillm.train.overfit import tiny_overfit_batch as run

    return run(model, batch, **kwargs)


def summarize_metrics(rows_or_path):
    from minillm.train.metrics import summarize_metrics

    return summarize_metrics(rows_or_path)


def load_jsonl_metrics(path):
    from minillm.train.metrics import load_jsonl_metrics

    return load_jsonl_metrics(path)


def write_metrics_csv(rows, path):
    from minillm.train.metrics import write_metrics_csv

    return write_metrics_csv(rows, path)


def benchmark_report(rows):
    from minillm.systems.analysis import benchmark_report

    return benchmark_report(rows)


def load_benchmark_csv(path):
    from minillm.systems.analysis import load_benchmark_csv

    return load_benchmark_csv(path)


def compare_attention_backends(rows):
    from minillm.systems.analysis import compare_attention_backends

    return compare_attention_backends(rows)


def write_benchmark_csv(rows, path):
    from minillm.systems.benchmark import write_csv

    return write_csv(rows, path)
