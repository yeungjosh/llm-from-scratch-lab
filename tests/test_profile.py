"""Phase 4 tests — profiler harness, sweep, trace emission. TDD."""

from __future__ import annotations

import json
from pathlib import Path

from lfslab.model import TransformerLM
from lfslab.profile_train import profile_one, sweep


def _tiny_model() -> TransformerLM:
    return TransformerLM(
        vocab_size=50,
        context_length=16,
        d_model=16,
        num_layers=1,
        num_heads=4,
        d_ff=32,
    )


def test_profile_one_returns_required_keys():
    result = profile_one(
        _tiny_model(), seq_len=8, batch_size=2, num_steps=2, device="cpu"
    )
    expected_keys = {
        "seq_len",
        "batch_size",
        "num_steps",
        "device",
        "step_time_ms",
        "tokens_per_sec",
        "peak_memory_bytes",
    }
    assert expected_keys.issubset(set(result.keys()))
    assert result["seq_len"] == 8
    assert result["batch_size"] == 2
    assert result["device"] == "cpu"
    assert result["tokens_per_sec"] > 0
    assert result["step_time_ms"] > 0


def test_sweep_returns_cross_product():
    seq_lens = [4, 8]
    batch_sizes = [1, 2]
    results = sweep(
        _tiny_model(),
        seq_lens=seq_lens,
        batch_sizes=batch_sizes,
        num_steps=1,
        device="cpu",
    )
    assert len(results) == len(seq_lens) * len(batch_sizes)
    pairs = {(r["seq_len"], r["batch_size"]) for r in results}
    assert pairs == {(s, b) for s in seq_lens for b in batch_sizes}


def test_profile_one_writes_trace_file(tmp_path: Path):
    result = profile_one(
        _tiny_model(),
        seq_len=4,
        batch_size=1,
        num_steps=1,
        device="cpu",
        trace_dir=tmp_path,
    )
    assert "trace_path" in result
    trace = Path(result["trace_path"])
    assert trace.exists()
    assert trace.stat().st_size > 0


def test_sweep_writes_json(tmp_path: Path):
    out = tmp_path / "sweep.json"
    sweep(
        _tiny_model(),
        seq_lens=[4],
        batch_sizes=[1, 2],
        num_steps=1,
        device="cpu",
        out_path=out,
    )
    assert out.exists()
    rows = json.loads(out.read_text())
    assert len(rows) == 2
    assert {r["batch_size"] for r in rows} == {1, 2}
