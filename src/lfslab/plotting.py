"""Plot-data helpers for the Streamlit explorer.

All chart-data functions return pandas DataFrames. Titles must declare provenance
(`Simulated`, `Measured`, `Hybrid`).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

import pandas as pd

from .estimator import (
    HardwareSpec,
    ModelSpec,
    RuntimeSpec,
    activation_bytes,
    kv_cache_bytes_per_sequence,
    param_count,
    tokens_per_second_upper_bound,
)

Provenance = Literal["Simulated", "Measured", "Hybrid"]


def title_with_provenance(base: str, provenance: Provenance) -> str:
    return f"{base}  ·  {provenance}"


def params_vs_d_model_curve(base: ModelSpec, d_models: list[int]) -> pd.DataFrame:
    rows = []
    for d in d_models:
        spec = dataclasses.replace(base, d_model=d)
        rows.append({"d_model": d, "params": param_count(spec)})
    return pd.DataFrame(rows)


def kv_cache_curve(
    spec: ModelSpec, context_lengths: list[int], bytes_kv: int = 1
) -> pd.DataFrame:
    rows = []
    for ctx in context_lengths:
        rows.append(
            {
                "context_length": ctx,
                "bytes_kv": bytes_kv,
                "kv_bytes": kv_cache_bytes_per_sequence(spec, ctx, bytes_kv),
            }
        )
    return pd.DataFrame(rows)


def activation_memory_curve(
    spec: ModelSpec, seq_lens: list[int], act_coeffs: list[int]
) -> pd.DataFrame:
    rows = []
    for s in seq_lens:
        for c in act_coeffs:
            r = RuntimeSpec(batch_size=1, seq_len=s, act_coeff=c)
            rows.append(
                {"seq_len": s, "act_coeff": c, "activation_bytes": activation_bytes(spec, r)}
            )
    return pd.DataFrame(rows)


def load_measurements(sweep_json_path: str | Path) -> pd.DataFrame:
    raw = json.loads(Path(sweep_json_path).read_text())
    return pd.DataFrame(raw)


def measured_vs_estimated(
    sweep_json_path: str | Path,
    spec: ModelSpec,
    hardware: HardwareSpec,
    bytes_param: int = 2,
) -> pd.DataFrame:
    """Read a sweep.json and join estimator-predicted tokens/s per row."""
    df = load_measurements(sweep_json_path).rename(
        columns={"tokens_per_sec": "tokens_per_sec_measured"}
    )
    p_bytes = param_count(spec) * bytes_param
    estimated = []
    for _, row in df.iterrows():
        kv_bytes = kv_cache_bytes_per_sequence(spec, int(row["seq_len"]))
        est = tokens_per_second_upper_bound(
            batch_size=int(row["batch_size"]),
            kv_cache_bytes_per_seq=kv_bytes,
            param_bytes_total=p_bytes,
            total_memory_bandwidth_bytes_per_s=hardware.memory_bandwidth_bytes_per_s,
        )
        estimated.append(est)
    df["tokens_per_sec_estimated"] = estimated
    df["tokens_per_sec_pct_error"] = (
        (df["tokens_per_sec_measured"] - df["tokens_per_sec_estimated"])
        / df["tokens_per_sec_estimated"]
        * 100
    )
    return df
