"""Anchor tests against JAX Scaling Book worked examples (Phase 5 target).

These currently pass off the closed-form formulas in lfslab.estimator;
keeping them green is a hard constraint when refactoring the estimator.
"""

from __future__ import annotations

import math

from lfslab.estimator import (
    ModelSpec,
    decode_step_time_lower_bound_s,
    kv_cache_bytes_per_sequence,
    param_count,
    train_flops_rule_of_thumb,
)


def test_param_count_jax_book_16b_example():
    m = ModelSpec(
        d_model=4096,
        d_ff=16384,
        n_layers=64,
        n_heads=32,
        n_kv_heads=32,
        head_dim=128,
        vocab_size=32000,
        tied_embeddings=False,
    )
    p = param_count(m)
    # JAX book quotes ~16B; the exact closed form (incl. untied embeddings and
    # the 2D(N+K)H attention parameterization) lands near 17.4B. Anchor at 10%.
    assert abs(p - 16_000_000_000) / 16_000_000_000 < 0.10


def test_attention_fraction_in_F_eq_4D_case():
    D = 4096
    F = 4 * D
    attention = 4 * D * D
    dense_core = attention + 3 * D * F
    assert math.isclose(attention / dense_core, 0.25, rel_tol=1e-6)


def test_kv_cache_per_token_64L_4096D_int8():
    m = ModelSpec(
        d_model=4096,
        d_ff=16384,
        n_layers=64,
        n_heads=32,
        n_kv_heads=32,
        head_dim=128,
        vocab_size=32000,
    )
    bytes_per_token = kv_cache_bytes_per_sequence(m, seq_len=1, bytes_kv=1)
    assert bytes_per_token == 524_288  # 512 KiB


def test_training_flops_70B_15T_tokens():
    # Use rule-of-thumb directly with a stand-in param count.
    total = 6 * 70e9 * 15e12
    assert math.isclose(total, 6.3e24, rel_tol=1e-9)
    # And via the function (different path, same number).
    m = ModelSpec(
        d_model=8192,
        d_ff=28672,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        vocab_size=128000,
        tied_embeddings=False,
    )
    rough = train_flops_rule_of_thumb(m, int(15e12))
    # We don't care that param_count for Llama-3-70B matches exactly here;
    # just that the 6·P·T law is wired correctly.
    assert rough == 6 * param_count(m) * int(15e12)


def test_decode_latency_lower_bound_example():
    # JAX inference worked example: batch_size=4, 16 GPUs each w/ 0.82 TB/s.
    t = decode_step_time_lower_bound_s(
        batch_size=4,
        kv_cache_bytes_per_seq=int(819e6),
        param_bytes_total=int(30e9),
        total_memory_bandwidth_bytes_per_s=16 * 8.2e11,
    )
    # Expected ~2.5 ms
    assert math.isclose(t, 0.00254, rel_tol=0.02)
