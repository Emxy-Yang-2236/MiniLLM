from __future__ import annotations

from types import SimpleNamespace

import torch

from adapters import default_stop_ids, generate_ids, generate_text, generation_config, get_tokenizer


class ToyNextTokenModel:
    def __init__(self):
        self.cfg = SimpleNamespace(context_length=8)
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, input_ids):
        vocab = 8
        logits = torch.full((*input_ids.shape, vocab), -1000.0, device=input_ids.device)
        next_token = 3 if input_ids.size(1) < 3 else 4
        logits[:, -1, next_token] = 1000.0
        return {"logits": logits}


class FixedTokenModel(torch.nn.Module):
    def __init__(self, token_ids: list[int], vocab_size: int):
        super().__init__()
        self.cfg = SimpleNamespace(context_length=8)
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.token_ids = token_ids
        self.vocab_size = vocab_size
        self.calls = 0
        self.inputs: list[list[list[int]]] = []

    def forward(self, input_ids):
        self.inputs.append(input_ids.detach().cpu().tolist())
        token = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        logits = torch.full((*input_ids.shape, self.vocab_size), -1000.0, device=input_ids.device)
        logits[:, -1, token] = 1000.0
        return {"logits": logits}


def _cs336_style_tokenizer():
    vocab = {0: b"<|endoftext|>", 1: b"a", 2: b"!"}
    return get_tokenizer(vocab, merges=[], special_tokens=["<|endoftext|>"])


def test_generate_ids_greedy_stops_on_stop_id_and_restores_train_mode():
    model = ToyNextTokenModel()
    cfg = generation_config(max_new_tokens=5, temperature=0.0, stop_ids=[4])
    out = generate_ids(model, torch.tensor([[1, 2]]), cfg)
    assert out.tolist() == [[1, 2, 3, 4]]
    assert model.training is True


def test_generate_ids_respects_max_new_tokens_without_stop():
    model = ToyNextTokenModel()
    cfg = generation_config(max_new_tokens=3, temperature=0.0, stop_ids=[])
    out = generate_ids(model, torch.tensor([[1, 2]]), cfg)
    assert out.shape == (1, 5)


def test_generate_cs336_style_tokenizer_uses_endoftext_stop_and_no_fake_bos():
    tok = _cs336_style_tokenizer()
    assert default_stop_ids(tok) == [0]
    model = FixedTokenModel([2, 0], vocab_size=3)
    completion = generate_text(model, tok, "a", max_new_tokens=4, temperature=0.0)
    assert completion == "!"
    assert model.inputs[0] == [[1]]


def test_generate_return_full_text_option():
    tok = _cs336_style_tokenizer()
    completion = generate_text(FixedTokenModel([2, 0], vocab_size=3), tok, "a", max_new_tokens=4, temperature=0.0)
    full = generate_text(
        FixedTokenModel([2, 0], vocab_size=3),
        tok,
        "a",
        max_new_tokens=4,
        temperature=0.0,
        return_full_text=True,
    )
    assert completion == "!"
    assert full == "a!"
