from __future__ import annotations

import pytest
import torch

from adapters import build_model, generate_text, make_config, model_summary, train_bpe


def test_transformer_forward_loss_and_generation_smoke():
    tok = train_bpe(["Once upon a time", "tiny model"], vocab_size=270)
    cfg = make_config(vocab_size=tok.vocab_size, context_length=12)
    model = build_model(cfg)
    x = torch.randint(0, tok.vocab_size, (2, 12))
    y = torch.randint(0, tok.vocab_size, (2, 12))
    out = model(x, y)
    assert out["logits"].shape == (2, 12, tok.vocab_size)
    assert torch.isfinite(out["loss"])
    text = generate_text(model, tok, "Once", max_new_tokens=2, temperature=0.0, device="cpu", return_full_text=True)
    assert isinstance(text, str)
    assert text.startswith("Once")


def test_attention_is_causal_for_prefix_positions():
    torch.manual_seed(0)
    cfg = make_config(vocab_size=64, context_length=8, attention_backend="naive")
    model = build_model(cfg).eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    a = torch.cat([prefix, torch.tensor([[5, 6, 7, 8]])], dim=1)
    b = torch.cat([prefix, torch.tensor([[8, 7, 6, 5]])], dim=1)
    with torch.no_grad():
        logits_a = model(a)["logits"][:, :4]
        logits_b = model(b)["logits"][:, :4]
    assert torch.allclose(logits_a, logits_b, atol=1e-5)


def test_config_validation_and_model_summary():
    with pytest.raises(ValueError):
        make_config(d_model=30, num_heads=8)
    with pytest.raises(ValueError):
        make_config(attention_backend="flash")
    cfg = make_config(vocab_size=80, context_length=8, tie_embeddings=False)
    model = build_model(cfg)
    summary = model_summary(model)
    assert summary["parameters"] > 0
    assert "mlp_type" not in summary
    assert summary["tie_embeddings"] is False


def test_loss_backpropagates_gradients():
    torch.manual_seed(0)
    cfg = make_config(vocab_size=64, context_length=8, attention_backend="sdpa")
    model = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    loss = model(x, y)["loss"]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(grad is not None for grad in grads)
    assert sum(float(torch.sum(grad.detach().abs())) for grad in grads) > 0.0
