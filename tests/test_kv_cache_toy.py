"""Phase 7 KV-cache toy — TDD.

Cached decoding must produce the same greedy token sequence as full recompute,
because both apply the same model to the same prefixes (modulo numerical noise).
"""

from __future__ import annotations

import time

import torch

from lfslab.kv_cache_toy import decode_with_cache, full_recompute_decode
from lfslab.model import TransformerLM


def _tiny_lm(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    return TransformerLM(
        vocab_size=64,
        context_length=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=64,
    )


def test_cached_decode_matches_full_recompute_greedy():
    model = _tiny_lm()
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    n_new = 12

    full = full_recompute_decode(model, prompt, n_new)
    cached = decode_with_cache(model, prompt, n_new)

    assert full.tolist() == cached.tolist()


def test_cached_decode_produces_expected_length():
    model = _tiny_lm()
    prompt = torch.tensor([[7, 8]], dtype=torch.long)
    out = decode_with_cache(model, prompt, max_new_tokens=10)
    assert out.shape == (1, 2 + 10)


def test_cached_decode_returns_int_tensor():
    model = _tiny_lm()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = decode_with_cache(model, prompt, max_new_tokens=5)
    assert out.dtype == torch.long


def test_cached_decode_is_faster_than_full_recompute_for_long_outputs():
    """Cached path computes O(L) per step vs O(L*T_total) for full recompute.
    On a small model this gap is small but should still be > 0."""
    model = _tiny_lm()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    n_new = 32

    # Warm up.
    decode_with_cache(model, prompt, 2)
    full_recompute_decode(model, prompt, 2)

    t0 = time.perf_counter()
    decode_with_cache(model, prompt, n_new)
    cached_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_recompute_decode(model, prompt, n_new)
    full_s = time.perf_counter() - t0

    # We expect cached <= full * 1.2 (allow noise).  Stronger claim — cached
    # should usually be faster — is left to a benchmark, not a unit test.
    assert cached_s <= full_s * 1.2, f"cached {cached_s:.4f}s vs full {full_s:.4f}s"
