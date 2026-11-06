"""Tests for lfslab.generate — greedy / temperature / top-k sampling."""

from __future__ import annotations

import torch

from lfslab.generate import generate
from lfslab.model import TransformerLM


class _ByteTok:
    """Minimal tokenizer stub: one id per character (code point mod vocab)."""

    def __init__(self, vocab: int = 32) -> None:
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab for c in text]

    def decode(self, ids: list[int]) -> list[int]:
        return list(ids)  # return ids so tests can inspect them


def _model(vocab: int = 32, ctx: int = 8) -> TransformerLM:
    torch.manual_seed(0)
    m = TransformerLM(
        vocab_size=vocab, context_length=ctx, d_model=16,
        num_layers=1, num_heads=2, d_ff=32,
    )
    m.eval()
    return m


def test_greedy_is_deterministic_and_extends_prompt():
    m, tok = _model(), _ByteTok()
    a = generate(m, tok, "abc", max_new_tokens=5, temperature=0)
    b = generate(m, tok, "abc", max_new_tokens=5, temperature=0)
    assert a == b
    assert len(a) == 3 + 5
    assert a[:3] == tok.encode("abc")


def test_greedy_matches_argmax_of_model():
    m, tok = _model(), _ByteTok()
    out = generate(m, tok, "abc", max_new_tokens=1, temperature=0)
    x = torch.tensor([tok.encode("abc")])
    assert out[-1] == int(m(x)[0, -1].argmax())


def test_top_k_one_equals_greedy():
    m, tok = _model(), _ByteTok()
    greedy = generate(m, tok, "abc", max_new_tokens=6, temperature=0)
    torch.manual_seed(123)
    topk1 = generate(m, tok, "abc", max_new_tokens=6, temperature=1.0, top_k=1)
    assert topk1 == greedy


def test_eos_stops_generation_without_emitting_eos():
    m, tok = _model(), _ByteTok()
    first = generate(m, tok, "abc", max_new_tokens=1, temperature=0)[-1]
    out = generate(m, tok, "abc", max_new_tokens=10, temperature=0, eos_id=first)
    assert out == tok.encode("abc")


def test_prompt_longer_than_context_is_truncated_not_crash():
    m, tok = _model(ctx=8), _ByteTok()
    out = generate(m, tok, "a" * 20, max_new_tokens=3, temperature=0)
    assert len(out) == 23


def test_top_k_larger_than_vocab_is_clamped():
    m, tok = _model(vocab=32), _ByteTok(32)
    out = generate(m, tok, "ab", max_new_tokens=2, temperature=1.0, top_k=1000)
    assert len(out) == 4 and all(0 <= i < 32 for i in out)
