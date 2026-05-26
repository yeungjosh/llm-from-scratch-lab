"""Phase 9.5 estimator v2: roofline switch.

The v1 bandwidth-bound formula is wildly wrong in regimes the technical note
identifies. v2 adds a roofline gate: compute arithmetic intensity per step,
compare to hardware's `peak_flops / peak_bandwidth` threshold, return
whichever bound is tighter (and report which one was used).
"""

from __future__ import annotations

import math

from lfslab.estimator import (
    HardwareSpec,
    ModelSpec,
    RuntimeSpec,
    predict_tokens_per_second_v2,
)


def _tiny_spec() -> ModelSpec:
    return ModelSpec(
        d_model=128, d_ff=512, n_layers=4, n_heads=4, n_kv_heads=4,
        head_dim=32, vocab_size=10000, tied_embeddings=True,
    )


def _a100_like() -> HardwareSpec:
    return HardwareSpec(
        name="a100-like",
        peak_flops_per_s=312e12,
        memory_bandwidth_bytes_per_s=2e12,
        device_memory_bytes=80 * 1024**3,
    )


def test_v2_returns_dict_with_mode_and_tokens_per_sec():
    out = predict_tokens_per_second_v2(_tiny_spec(), RuntimeSpec(batch_size=1, seq_len=64), _a100_like())
    assert "tokens_per_sec" in out
    assert "bound" in out  # "memory" or "compute"
    assert out["bound"] in {"memory", "compute"}
    assert out["tokens_per_sec"] > 0


def test_v2_picks_compute_bound_for_small_model_high_seq():
    """A tiny model with long sequence has high arithmetic intensity and is compute-bound."""
    out = predict_tokens_per_second_v2(
        _tiny_spec(),
        RuntimeSpec(batch_size=4, seq_len=512),
        _a100_like(),
    )
    assert out["bound"] == "compute"


def test_v2_picks_memory_bound_for_large_model_short_seq():
    """A multi-billion-param model with batch=1 short seq is memory-bound on any
    GPU because per-token FLOPs are tiny relative to parameter bytes that must
    be loaded for that one step."""
    big = ModelSpec(
        d_model=8192, d_ff=32768, n_layers=80, n_heads=64, n_kv_heads=8,
        head_dim=128, vocab_size=128000, tied_embeddings=False,
    )
    out = predict_tokens_per_second_v2(big, RuntimeSpec(batch_size=1, seq_len=1), _a100_like())
    assert out["bound"] == "memory"


def test_v2_reports_arithmetic_intensity_and_threshold():
    out = predict_tokens_per_second_v2(
        _tiny_spec(), RuntimeSpec(batch_size=4, seq_len=256), _a100_like()
    )
    assert "arithmetic_intensity" in out
    assert "compute_bandwidth_threshold" in out
    # A100 threshold is peak_flops / peak_bw = 312e12 / 2e12 = 156 FLOPs/byte.
    assert math.isclose(out["compute_bandwidth_threshold"], 156.0, rel_tol=1e-6)
    assert out["arithmetic_intensity"] > 0


def test_v2_compute_bound_value_matches_peak_flops_over_2P():
    """When compute-bound, tokens/s upper bound is peak_flops / (2 * params)."""
    spec = _tiny_spec()
    hw = _a100_like()
    out = predict_tokens_per_second_v2(
        spec, RuntimeSpec(batch_size=4, seq_len=512), hw
    )
    if out["bound"] == "compute":
        from lfslab.estimator import param_count
        expected = hw.peak_flops_per_s * hw.mfu / (2 * param_count(spec))
        assert math.isclose(out["tokens_per_sec"], expected, rel_tol=1e-6)


def test_v2_overhead_floor_caps_tokens_per_sec():
    """With a 2.5 ms overhead floor at seq=64 batch=1, tokens/s is capped at 64/0.0025 = 25,600."""
    spec = _tiny_spec()
    hw = _a100_like()
    out = predict_tokens_per_second_v2(
        spec,
        RuntimeSpec(batch_size=1, seq_len=64),
        hw,
        overhead_floor_s=0.0025,
    )
    assert out["overhead_capped"] is True
    assert math.isclose(out["tokens_per_sec"], 64 / 0.0025, rel_tol=1e-6)


def test_v2_overhead_floor_inactive_for_long_step():
    """At a large config where step time naturally exceeds the floor, no capping."""
    big = ModelSpec(
        d_model=8192, d_ff=32768, n_layers=80, n_heads=64, n_kv_heads=8,
        head_dim=128, vocab_size=128000,
    )
    out = predict_tokens_per_second_v2(
        big,
        RuntimeSpec(batch_size=64, seq_len=4096),
        _a100_like(),
        overhead_floor_s=0.0025,
    )
    assert out["overhead_capped"] is False


def test_v2_includes_mfu_in_compute_bound():
    """Compute-bound prediction should scale linearly with MFU."""
    spec = _tiny_spec()
    runtime = RuntimeSpec(batch_size=4, seq_len=512)
    hw_full = HardwareSpec(name="x", peak_flops_per_s=1e15, memory_bandwidth_bytes_per_s=2e12, device_memory_bytes=1, mfu=1.0)
    hw_half = HardwareSpec(name="x", peak_flops_per_s=1e15, memory_bandwidth_bytes_per_s=2e12, device_memory_bytes=1, mfu=0.5)
    out_full = predict_tokens_per_second_v2(spec, runtime, hw_full)
    out_half = predict_tokens_per_second_v2(spec, runtime, hw_half)
    assert out_full["bound"] == "compute"
    assert out_half["bound"] == "compute"
    assert math.isclose(out_full["tokens_per_sec"] / out_half["tokens_per_sec"], 2.0, rel_tol=1e-6)
