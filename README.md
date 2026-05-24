# llm_from_scratch_lab

A from-scratch implementation of a small decoder-only language model, plus a calibrated scaling/memory/latency estimator and an interactive Streamlit explorer that overlays **measured** local benchmarks against **simulated** estimator curves.

Inspired by Stanford's CS336 (Language Modeling from Scratch) and the JAX Scaling Book. Independent reimplementation — not coursework.

## Why

Two things I wanted in one repo:
1. A correct, tested, end-to-end transformer trained on TinyStories that I can poke at on a MacBook.
2. A transparent assumptions model — closed-form formulas for parameters, training FLOPs, activation memory, KV-cache size, decode latency — calibrated against real measurements on Apple silicon (and, optionally, against rented GPUs).

Most "GPT from scratch" repos stop at (1). The interesting questions live in (2): where does the back-of-envelope estimator agree with reality, and where does it break?

## Quickstart

```bash
uv sync
uv run pytest -q
uv run streamlit run app/streamlit_app.py
```

## Layout

```
src/lfslab/
  tokenizer.py        # byte-level BPE
  model.py            # RMSNorm + SwiGLU + RoPE + causal MHA, decoder-only
  train.py            # AdamW + cosine schedule + grad clip + checkpoints
  generate.py         # greedy / temperature / top-k sampling
  profile_train.py    # torch.profiler harness + seq_len/batch_size sweep
  estimator.py        # closed-form params/FLOPs/memory/latency (no torch import)
  hardware.py         # device catalogue (MPS, T4, A10G, A100, H100)
  kv_cache_toy.py     # cached autoregressive decode (optional)
app/streamlit_app.py  # Plotly explorer with measured-vs-estimated overlays
tests/                # pytest suite (adapter-anchored to CS336 A1 signatures)
configs/              # tiny_cpu.yaml, tiny_mps.yaml, estimator_defaults.yaml
```

## Status

- [ ] Phase 0 — Bootstrap
- [ ] Phase 1 — Tokenizer + data path
- [ ] Phase 2 — Transformer core
- [ ] Phase 3 — Training + eval loop
- [ ] Phase 4 — Profiling + measurement
- [ ] Phase 5 — Scaling estimator
- [ ] Phase 6 — Streamlit explorer
- [ ] Phase 7 — KV-cache toy (optional)
- [ ] Phase 8 — Cloud GPU validation (optional)
- [ ] Phase 9 — Documentation + polish

## License

MIT
