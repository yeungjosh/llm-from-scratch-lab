"""Phase 6 plot-data helpers — TDD."""

from __future__ import annotations

import json
from pathlib import Path

from lfslab.estimator import HardwareSpec, ModelSpec
from lfslab.plotting import (
    activation_memory_curve,
    kv_cache_curve,
    load_measurements,
    measured_vs_estimated,
    params_vs_d_model_curve,
)


def _spec() -> ModelSpec:
    return ModelSpec(
        d_model=512,
        d_ff=2048,
        n_layers=8,
        n_heads=8,
        n_kv_heads=8,
        head_dim=64,
        vocab_size=32000,
    )


def test_params_vs_d_model_curve():
    base = _spec()
    df = params_vs_d_model_curve(base, d_models=[128, 256, 512, 1024])
    assert list(df.columns) == ["d_model", "params"]
    assert len(df) == 4
    # Monotonic increasing in d_model.
    assert list(df["params"]) == sorted(df["params"])


def test_kv_cache_curve_scales_linearly_with_context():
    df = kv_cache_curve(_spec(), context_lengths=[128, 256, 512], bytes_kv=1)
    assert {"context_length", "bytes_kv", "kv_bytes"}.issubset(df.columns)
    # Doubling context length doubles KV bytes.
    row_128 = df[df["context_length"] == 128]["kv_bytes"].iloc[0]
    row_256 = df[df["context_length"] == 256]["kv_bytes"].iloc[0]
    assert row_256 == 2 * row_128


def test_activation_memory_curve_grows_with_remat_coeff():
    df = activation_memory_curve(_spec(), seq_lens=[128, 256], act_coeffs=[1, 7, 20])
    assert len(df) == 2 * 3
    # For a fixed seq_len, higher act_coeff means more memory.
    for s in [128, 256]:
        sub = df[df["seq_len"] == s].sort_values("act_coeff")
        vals = sub["activation_bytes"].tolist()
        assert vals == sorted(vals)


def test_load_measurements_reads_sweep_json(tmp_path: Path):
    rows = [
        {
            "seq_len": 128, "batch_size": 1, "num_steps": 5,
            "device": "cpu", "step_time_ms": 12.3, "tokens_per_sec": 1000.0,
            "peak_memory_bytes": 0,
        },
        {
            "seq_len": 256, "batch_size": 1, "num_steps": 5,
            "device": "cpu", "step_time_ms": 25.0, "tokens_per_sec": 1024.0,
            "peak_memory_bytes": 0,
        },
    ]
    p = tmp_path / "sweep.json"
    p.write_text(json.dumps(rows))
    df = load_measurements(p)
    assert len(df) == 2
    assert "tokens_per_sec" in df.columns


def test_measured_vs_estimated_adds_estimated_columns(tmp_path: Path):
    rows = [
        {
            "seq_len": 128, "batch_size": 4, "num_steps": 5,
            "device": "cpu", "step_time_ms": 50.0, "tokens_per_sec": 10240.0,
            "peak_memory_bytes": 0,
        },
    ]
    p = tmp_path / "sweep.json"
    p.write_text(json.dumps(rows))
    hw = HardwareSpec(
        name="x",
        peak_flops_per_s=1e12,
        memory_bandwidth_bytes_per_s=1e11,
        device_memory_bytes=1,
    )
    df = measured_vs_estimated(p, _spec(), hw)
    assert "tokens_per_sec_measured" in df.columns
    assert "tokens_per_sec_estimated" in df.columns
    assert "tokens_per_sec_pct_error" in df.columns
