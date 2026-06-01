from __future__ import annotations

import torch

from adapters import get_layers


def test_linear_embedding_rmsnorm_and_cross_entropy():
    layers = get_layers()
    linear = layers.Linear(3, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(linear(x), torch.tensor([[14.0, 2.0]]))

    emb = layers.Embedding(4, 3)
    ids = torch.tensor([[0, 2]])
    assert emb(ids).shape == (1, 2, 3)

    norm = layers.RMSNorm(3)
    y = norm(torch.tensor([[1.0, 2.0, 2.0]]))
    assert torch.allclose(torch.sqrt(torch.mean(y * y, dim=-1)), torch.ones(1), atol=1e-4)

    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    targets = torch.tensor([0, 1])
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(layers.cross_entropy(logits, targets), expected)


def test_rope_preserves_vector_norms():
    layers = get_layers()
    rope = layers.RotaryPositionalEmbedding(theta=10000.0, d_k=8, max_seq_len=5)
    positions = torch.arange(5)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 5, 8)
    q2 = rope(q, positions)
    k2 = rope(k, positions)
    assert torch.allclose(torch.linalg.norm(q, dim=-1), torch.linalg.norm(q2, dim=-1), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(k, dim=-1), torch.linalg.norm(k2, dim=-1), atol=1e-5)
