"""Streamlit explorer. Phase 6.

Renders a placeholder until phases 1–5 are wired up.
"""

from __future__ import annotations

import streamlit as st

from lfslab import __version__
from lfslab.estimator import (
    ModelSpec,
    RuntimeSpec,
    activation_bytes,
    kv_cache_bytes_per_sequence,
    param_count,
)

st.set_page_config(page_title="llm_from_scratch_lab", layout="wide")
st.title("llm_from_scratch_lab")
st.caption(f"v{__version__} · estimator explorer (Phase 6 WIP)")

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
r = RuntimeSpec(batch_size=int(batch_size), seq_len=int(seq_len))

col1, col2, col3 = st.columns(3)
col1.metric("Params", f"{param_count(m):,}")
col2.metric("Activation bytes (est.)", f"{activation_bytes(m, r) / 1e9:.2f} GB")
col3.metric(
    "KV cache / seq (bytes)",
    f"{kv_cache_bytes_per_sequence(m, r.seq_len) / 1e6:.2f} MB",
)

st.info("Plotly charts will land in Phase 6 — sidebar wired to the estimator already.")
