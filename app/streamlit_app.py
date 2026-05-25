"""Streamlit explorer for the scaling estimator + measured benchmarks.

Plotly charts. Each title declares provenance (Simulated / Measured / Hybrid).
"""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import streamlit as st

from lfslab import __version__
from lfslab.estimator import (
    HardwareSpec,
    ModelSpec,
    RuntimeSpec,
    activation_bytes,
    decode_step_time_lower_bound_s,
    kv_cache_bytes_per_sequence,
    param_count,
    tokens_per_second_upper_bound,
)
from lfslab.hardware import CATALOGUE
from lfslab.plotting import (
    activation_memory_curve,
    kv_cache_curve,
    measured_vs_estimated,
    params_vs_d_model_curve,
    title_with_provenance,
)

st.set_page_config(page_title="llm_from_scratch_lab", layout="wide")
st.title("llm_from_scratch_lab")
st.caption(f"v{__version__} · scaling-estimator explorer")

# --- sidebar -----------------------------------------------------------------

st.sidebar.header("ModelSpec")
d_model = st.sidebar.number_input("d_model", 64, 8192, 512, step=64)
d_ff = st.sidebar.number_input("d_ff", 64, 32768, 2048, step=64)
n_layers = st.sidebar.number_input("n_layers", 1, 128, 8)
n_heads = st.sidebar.number_input("n_heads", 1, 64, 8)
head_dim = st.sidebar.number_input("head_dim", 16, 256, 64)
vocab_size = st.sidebar.number_input("vocab_size", 256, 200000, 32000, step=1000)
tied = st.sidebar.checkbox("tied embeddings", value=False)

st.sidebar.header("RuntimeSpec")
batch_size = st.sidebar.number_input("batch_size", 1, 1024, 8)
seq_len = st.sidebar.number_input("seq_len", 16, 32768, 1024)
act_coeff = st.sidebar.select_slider(
    "activation remat policy (lower = more remat)", options=[1, 7, 20], value=7
)

st.sidebar.header("HardwareSpec")
hw_key = st.sidebar.selectbox("device", list(CATALOGUE.keys()), index=1)
hw: HardwareSpec = CATALOGUE[hw_key]

m = ModelSpec(
    d_model=int(d_model),
    d_ff=int(d_ff),
    n_layers=int(n_layers),
    n_heads=int(n_heads),
    n_kv_heads=int(n_heads),
    head_dim=int(head_dim),
    vocab_size=int(vocab_size),
    tied_embeddings=tied,
)
r = RuntimeSpec(batch_size=int(batch_size), seq_len=int(seq_len), act_coeff=int(act_coeff))

# --- headline metrics --------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Params", f"{param_count(m):,}")
col2.metric("Activation bytes (est.)", f"{activation_bytes(m, r) / 1e9:.2f} GB")
col3.metric(
    "KV cache / seq",
    f"{kv_cache_bytes_per_sequence(m, r.seq_len) / 1e6:.2f} MB",
)
p_bytes = param_count(m) * r.bytes_param
kv_bytes = kv_cache_bytes_per_sequence(m, r.seq_len)
t_decode = decode_step_time_lower_bound_s(
    r.batch_size, kv_bytes, p_bytes, hw.memory_bandwidth_bytes_per_s
)
col4.metric("Decode step (lower bound)", f"{t_decode * 1000:.2f} ms")

# --- charts ------------------------------------------------------------------

tab_param, tab_kv, tab_act, tab_decode, tab_meas = st.tabs(
    ["Params vs d_model", "KV cache vs context", "Activation memory", "Decode frontier", "Measured vs estimated"]
)

with tab_param:
    d_models = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    df = params_vs_d_model_curve(m, d_models)
    fig = px.line(
        df,
        x="d_model",
        y="params",
        markers=True,
        log_y=True,
        title=title_with_provenance("Parameter count vs d_model", "Simulated"),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_kv:
    ctxs = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    df = kv_cache_curve(m, ctxs, bytes_kv=int(r.bytes_kv))
    df["kv_mb"] = df["kv_bytes"] / 1e6
    fig = px.line(
        df,
        x="context_length",
        y="kv_mb",
        markers=True,
        log_x=True,
        log_y=True,
        labels={"kv_mb": "KV cache (MB / sequence)"},
        title=title_with_provenance("KV cache vs context length", "Simulated"),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_act:
    seqs = [128, 256, 512, 1024, 2048, 4096]
    df = activation_memory_curve(m, seqs, act_coeffs=[1, 7, 20])
    df["act_gb"] = df["activation_bytes"] / 1e9
    fig = px.line(
        df,
        x="seq_len",
        y="act_gb",
        color="act_coeff",
        markers=True,
        log_x=True,
        log_y=True,
        labels={"act_gb": "Activation memory (GB)", "act_coeff": "remat coeff"},
        title=title_with_provenance(
            "Activation memory vs seq_len under remat policies", "Simulated"
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_decode:
    batches = [1, 2, 4, 8, 16, 32, 64]
    rows = []
    for b in batches:
        rows.append(
            {
                "batch_size": b,
                "tokens_per_sec": tokens_per_second_upper_bound(
                    b, kv_bytes, p_bytes, hw.memory_bandwidth_bytes_per_s
                ),
                "ms_per_step": decode_step_time_lower_bound_s(
                    b, kv_bytes, p_bytes, hw.memory_bandwidth_bytes_per_s
                )
                * 1000,
            }
        )
    import pandas as pd

    df = pd.DataFrame(rows)
    fig = px.line(
        df,
        x="ms_per_step",
        y="tokens_per_sec",
        markers=True,
        log_x=True,
        log_y=True,
        hover_data=["batch_size"],
        title=title_with_provenance(
            f"Decode latency↔throughput frontier on {hw.name}", "Simulated"
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_meas:
    sweep_path = Path("results/local/sweep.json")
    if sweep_path.exists():
        df = measured_vs_estimated(sweep_path, m, hw)
        fig = px.scatter(
            df,
            x="tokens_per_sec_estimated",
            y="tokens_per_sec_measured",
            hover_data=["seq_len", "batch_size", "device"],
            title=title_with_provenance(
                "Measured vs estimated tokens/s", "Hybrid"
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)
    else:
        st.info(
            "Run `uv run python -m lfslab.profile_train --config configs/tiny_mps.yaml` "
            "to generate `results/local/sweep.json`, then this overlay populates."
        )
