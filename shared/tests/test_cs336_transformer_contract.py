from __future__ import annotations

import math

import torch

from adapters import (
    build_model,
    make_config,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rope,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm,
)


def test_cs336_rope_tiny_numerical_case():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    pos = torch.tensor([[0, 1]])
    out = run_rope(4, theta=10000.0, max_seq_len=4, in_query_or_key=x, token_positions=pos)
    expected = torch.empty_like(x)
    expected[:, 0] = x[:, 0]
    expected[0, 1, 0] = 5.0 * math.cos(1.0) - 6.0 * math.sin(1.0)
    expected[0, 1, 1] = 5.0 * math.sin(1.0) + 6.0 * math.cos(1.0)
    expected[0, 1, 2] = 7.0 * math.cos(0.01) - 8.0 * math.sin(0.01)
    expected[0, 1, 3] = 7.0 * math.sin(0.01) + 8.0 * math.cos(0.01)
    assert torch.allclose(out, expected, atol=1e-6)


def test_cs336_scaled_dot_product_attention_mask_contract():
    q = torch.randn(2, 3, 4, 6)
    k = torch.randn(2, 3, 4, 6)
    v = torch.randn(2, 3, 4, 5)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    out = run_scaled_dot_product_attention(q, k, v, mask)
    scores = q @ k.transpose(-2, -1) / math.sqrt(6)
    expected = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
    assert out.shape == (2, 3, 4, 5)
    assert torch.allclose(out, expected, atol=1e-6)


def test_cs336_causal_mha_prevents_future_token_dependence():
    torch.manual_seed(0)
    d_model = 8
    num_heads = 2
    weights = [torch.randn(d_model, d_model) for _ in range(4)]
    prefix = torch.randn(1, 3, d_model)
    future_a = torch.randn(1, 2, d_model)
    future_b = torch.randn(1, 2, d_model)
    out_a = run_multihead_self_attention(d_model, num_heads, *weights, torch.cat([prefix, future_a], dim=1))
    out_b = run_multihead_self_attention(d_model, num_heads, *weights, torch.cat([prefix, future_b], dim=1))
    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-6)


def test_cs336_mha_with_rope_shape_and_deterministic_numerics():
    torch.manual_seed(1)
    d_model = 8
    num_heads = 2
    x = torch.randn(2, 4, d_model)
    weights = [torch.randn(d_model, d_model) for _ in range(4)]
    pos = torch.arange(4).unsqueeze(0).expand(2, -1)
    out_1 = run_multihead_self_attention_with_rope(d_model, num_heads, 8, 10000.0, *weights, x, token_positions=pos)
    out_2 = run_multihead_self_attention_with_rope(d_model, num_heads, 8, 10000.0, *weights, x, token_positions=pos)
    assert out_1.shape == x.shape
    assert torch.allclose(out_1, out_2, atol=1e-6)
    assert torch.all(torch.isfinite(out_1))


def test_cs336_transformer_block_lm_loss_and_truncated_input():
    from minillm.model.transformer import TransformerBlock, TransformerLM

    torch.manual_seed(2)
    cfg = make_config(
        vocab_size=40,
        context_length=8,
        d_model=16,
        d_ff=32,
        num_layers=2,
        num_heads=4,
        tie_embeddings=False,
    )
    model = build_model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    labels = torch.randint(0, cfg.vocab_size, (2, 5))
    out = model(ids, labels)
    assert out["logits"].shape == (2, 5, cfg.vocab_size)
    assert torch.isfinite(out["loss"])

    adapter_logits = run_transformer_lm(
        cfg.vocab_size,
        cfg.context_length,
        cfg.d_model,
        cfg.num_layers,
        cfg.num_heads,
        int(cfg.d_ff),
        cfg.rope_theta,
        model.state_dict(),
        ids[:, :3],
    )
    with torch.no_grad():
        expected = model(ids[:, :3])["logits"]
    assert torch.allclose(adapter_logits, expected, atol=1e-6)

    block = TransformerBlock(d_model=16, num_heads=4, d_ff=32, max_seq_len=8, theta=10000.0)
    x = torch.randn(2, 5, 16)
    assert run_transformer_block(16, 4, 32, 8, 10000.0, block.state_dict(), x).shape == x.shape
