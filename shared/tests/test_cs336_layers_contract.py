from __future__ import annotations

import torch

from adapters import (
    run_cross_entropy,
    run_embedding,
    run_linear,
    run_rmsnorm,
    run_silu,
    run_softmax,
    run_swiglu,
)


def test_cs336_linear_embedding_and_rmsnorm_contracts():
    x = torch.tensor([[[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]]])
    weights = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.5]])
    assert torch.allclose(run_linear(3, 2, weights, x), x @ weights.T)

    emb_weights = torch.arange(20, dtype=torch.float32).view(5, 4)
    token_ids = torch.tensor([[0, 3, 4]])
    assert torch.equal(run_embedding(5, 4, emb_weights, token_ids), emb_weights[token_ids])

    norm_weights = torch.tensor([1.0, 2.0, -0.5])
    expected = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + 1e-5)
    expected = expected * norm_weights
    assert torch.allclose(run_rmsnorm(3, 1e-5, norm_weights, x), expected, atol=1e-6)


def test_cs336_silu_swiglu_softmax_and_cross_entropy_contracts():
    x = torch.tensor([[[-1.0, 0.0, 2.0]]])
    assert torch.allclose(run_silu(x), torch.nn.functional.silu(x))

    w1 = torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]])
    w2 = torch.tensor([[1.5, -0.5], [0.25, 2.0], [-1.0, 0.75]])
    w3 = torch.tensor([[0.5, 1.0, 1.5], [-1.0, 0.0, 2.0]])
    expected = (torch.nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
    assert torch.allclose(run_swiglu(3, 2, w1, w2, w3, x), expected, atol=1e-6)

    logits = torch.tensor([[1000.0, 1001.0, 999.0], [-5.0, -5.0, -4.0]])
    probs = run_softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
    assert torch.all(torch.isfinite(probs))
    assert torch.allclose(probs, torch.softmax(logits, dim=-1), atol=1e-6)

    ce_logits = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 1.5, 3.0]])
    targets = torch.tensor([0, 2])
    assert torch.allclose(
        run_cross_entropy(ce_logits, targets),
        torch.nn.functional.cross_entropy(ce_logits, targets),
        atol=1e-6,
    )


def test_cs336_rmsnorm_upcasts_float16_for_numerics():
    x = torch.tensor([[1e-3, 2e-3, -3e-3]], dtype=torch.float16)
    weights = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float16)
    out = run_rmsnorm(3, 1e-5, weights, x)
    expected = (x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + 1e-5))
    expected = (expected * weights.float()).to(torch.float16)
    assert out.dtype == torch.float16
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3)
