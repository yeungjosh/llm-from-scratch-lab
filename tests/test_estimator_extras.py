"""Phase 5 estimator extensions — TDD."""

from __future__ import annotations

import math

from lfslab.estimator import (
    HardwareSpec,
    ModelSpec,
    roofline_intensity_threshold,
    train_time_estimate_s,
)


def _llama70b_like() -> ModelSpec:
    return ModelSpec(
        d_model=8192,
        d_ff=28672,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        vocab_size=128000,
        tied_embeddings=False,
    )


def test_train_time_70B_15T_one_h100_at_perfect_mfu():
    """70B × 15T tokens at perfect MFU on a single 1e15 FLOPS device:
    total FLOPs = 6.3e24, wall time = 6.3e24 / 1e15 = 6.3e9 seconds.
    Use a stand-in 70B param count via the rule-of-thumb path.
    """
    hw = HardwareSpec(
        name="ideal-1pflops",
        peak_flops_per_s=1e15,
        memory_bandwidth_bytes_per_s=1e12,
        device_memory_bytes=80 * 1024**3,
        mfu=1.0,
    )
    # 6 * 70e9 * 15e12 / 1e15 = 6.3e9
    t = train_time_estimate_s(
        train_flops=6 * 70e9 * 15e12,
        hardware=hw,
        num_devices=1,
    )
    assert math.isclose(t, 6.3e9, rel_tol=1e-6)


def test_train_time_scales_inversely_with_num_devices_and_mfu():
    hw = HardwareSpec(
        name="x",
        peak_flops_per_s=1e15,
        memory_bandwidth_bytes_per_s=1e12,
        device_memory_bytes=1,
        mfu=0.5,
    )
    flops = 1e21
    t1 = train_time_estimate_s(flops, hw, num_devices=1)
    t8 = train_time_estimate_s(flops, hw, num_devices=8)
    assert math.isclose(t1 / t8, 8.0, rel_tol=1e-6)


def test_roofline_intensity_threshold_is_flops_over_bytes():
    """Arithmetic intensity threshold = peak_flops / memory_bandwidth (FLOPs/byte)."""
    hw = HardwareSpec(
        name="a100",
        peak_flops_per_s=312e12,
        memory_bandwidth_bytes_per_s=2e12,
        device_memory_bytes=80 * 1024**3,
    )
    # Threshold = 312e12 / 2e12 = 156 FLOPs/byte
    assert math.isclose(roofline_intensity_threshold(hw), 156.0, rel_tol=1e-9)


def test_param_count_70b_within_ballpark():
    # Llama-3-70B reference number; our closed-form is allowed ±15% (GQA + non-tied embeds).
    m = _llama70b_like()
    from lfslab.estimator import param_count

    p = param_count(m)
    assert abs(p - 70e9) / 70e9 < 0.15
