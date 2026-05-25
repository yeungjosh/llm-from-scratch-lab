# llm_from_scratch_lab

A from-scratch decoder-only language model in PyTorch, plus a closed-form scaling /
memory / latency estimator and an interactive Streamlit explorer that overlays
**measured** local benchmarks against **simulated** estimator curves.

Inspired by Stanford CS336 (Spring 2025) and the JAX Scaling Book.
Independent reimplementation — not coursework.

[![CI](https://github.com/yeungjosh/llm-from-scratch-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/yeungjosh/llm-from-scratch-lab/actions/workflows/ci.yml)

## Why

Two things I wanted in one repo:
1. A correct, tested, end-to-end transformer trained on TinyStories that runs on a MacBook.
2. A transparent assumptions model — closed-form formulas for parameters, training FLOPs, activation memory, KV-cache size, decode latency — calibrated against real measurements.

Most "GPT from scratch" repos stop at (1). The interesting questions live in (2):
where does the back-of-envelope estimator agree with reality, and where does it break?
See [reports/technical_note.md](reports/technical_note.md) for a calibration study
that finds the bandwidth-bound upper bound for `tokens/s` is wildly loose at
`batch ≥ 2` on Apple silicon for a small model, and discusses why.

## Quickstart (target: under 15 minutes from clone to charts)

```bash
git clone https://github.com/yeungjosh/llm-from-scratch-lab
cd llm-from-scratch-lab
uv sync                              # installs torch, streamlit, plotly, ...
uv run pytest -q                     # 64/64 green
uv run streamlit run app/streamlit_app.py   # interactive explorer

# end-to-end demo: train a tiny model + run a benchmark sweep
uv run python scripts/download_data.py      # ~2 GB TinyStories (skips if present)
uv run python scripts/tokenize_corpus.py \
    --train data/TinyStoriesV2-GPT4-train.txt \
    --valid data/TinyStoriesV2-GPT4-valid.txt \
    --vocab-size 10000 \
    --train-sample-bytes 5000000        # cap BPE training corpus for speed
uv run python -m lfslab.train --config configs/tiny_cpu.yaml --max-steps 200
uv run python -m lfslab.profile_train --config configs/tiny_cpu.yaml \
    --seq-lens 64 128 256 --batch-sizes 1 2 4 --num-steps 5
```

After the sweep, refresh the Streamlit app and the "Measured vs estimated" tab
populates with measured-vs-predicted scatter + percent-error table.

## Architecture

```
                ┌──────────────────────────────────────────────┐
                │              app/streamlit_app.py            │
                │  ModelSpec / RuntimeSpec / HardwareSpec UI   │
                │  Plotly tabs (provenance-labeled):           │
                │   · Params vs d_model           Simulated    │
                │   · KV cache vs context         Simulated    │
                │   · Activation memory           Simulated    │
                │   · Decode latency frontier     Simulated    │
                │   · Measured vs estimated       Hybrid       │
                └────────────────┬─────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
  ┌────────────────┐  ┌───────────────────┐  ┌──────────────────┐
  │  lfslab.model  │  │ lfslab.estimator  │  │ lfslab.plotting  │
  │ TransformerLM  │  │  closed-form      │  │ DataFrames for   │
  │ RMSNorm SwiGLU │  │  params, FLOPs,   │  │ each chart;      │
  │ RoPE causal    │  │  activation mem,  │  │ measured_vs_     │
  │ MHA (GQA-able) │  │  KV cache, decode │  │ estimated()      │
  │ pre-norm       │  │  latency, train   │  │ joins sweep.json │
  └───────┬────────┘  │  time, roofline   │  └──────────────────┘
          │           │  (pure-Python,    │
          ▼           │   no torch)       │
  ┌────────────────┐  └───────────────────┘
  │  lfslab.train  │
  │ AdamW + cosine │            ┌─────────────────────────────┐
  │ schedule + grad│ ─run sweep▶│   lfslab.profile_train      │
  │ clip + ckpts   │            │ profile_one / sweep         │
  └───────┬────────┘            │ torch.profiler traces       │
          │                     │ → results/local/sweep.json  │
          ▼                     └─────────────────────────────┘
  ┌────────────────┐
  │ tests/         │
  │ adapters.py    │  ◀── CS336 A1 official adapter signatures
  │ test_*.py      │      (drop CS336 pytest in to verify against course)
  └────────────────┘
```

## Status (8 phases shipped)

- [x] Phase 0 — Bootstrap (`uv`, src/ layout, CI, anchor tests)
- [x] Phase 1 — Byte-level BPE tokenizer + TinyStories data path
- [x] Phase 2 — Decoder-only Transformer core (RMSNorm + SwiGLU + RoPE + causal MHA)
- [x] Phase 3 — Training loop (AdamW, cosine schedule, grad clip, checkpoints, smoke train)
- [x] Phase 4 — `torch.profiler` harness + seq/batch sweep
- [x] Phase 5 — Scaling estimator (params, FLOPs, activation, KV, decode latency, train time, roofline)
- [x] Phase 6 — Streamlit explorer with measured-vs-estimated overlay
- [x] Phase 9 — Technical note (calibration study, [reports/technical_note.md](reports/technical_note.md))
- [ ] Phase 7 — KV-cache toy decode (optional)
- [ ] Phase 8 — Cloud GPU validation appendix (optional)

## Anchor tests (JAX Scaling Book worked examples)

8 closed-form anchor tests guard the estimator against drift:

- 16 B-param check (D=4096, F=4D, L=64, V=32k) within 10 %
- Attention parameter fraction in F=4D case is exactly ¼
- 512 KiB / token KV cache (64-layer, D=4096, int8)
- 6.3 × 10²⁴ FLOPs total for a 70 B-param × 15 T-token run
- 2.5 ms decode-step lower bound on the JAX inference example
- Llama-3-70B-shape param count within 15 %
- `train_time_estimate_s` exact for `flops / (devices × peak_flops × MFU)`
- `roofline_intensity_threshold = peak_flops / bandwidth`

## Layout

```
src/lfslab/
  tokenizer.py        # byte-level BPE
  model.py            # RMSNorm + SwiGLU + RoPE + causal MHA, decoder-only
  train.py            # AdamW + cosine + grad clip + checkpoints
  generate.py         # greedy / temperature / top-k sampling
  profile_train.py    # torch.profiler harness + seq/batch sweep
  estimator.py        # closed-form params / FLOPs / memory / latency (pure-Python)
  hardware.py         # device catalogue (Apple CPU, MPS, T4, A10G, A100, H100)
  plotting.py         # Plotly chart DataFrame helpers, provenance labels
  kv_cache_toy.py     # cached autoregressive decode (Phase 7, optional)
app/streamlit_app.py  # Plotly explorer with measured-vs-estimated overlay
tests/                # 64 tests (model, training, estimator, profile, app)
configs/              # tiny_cpu.yaml, tiny_mps.yaml, estimator_defaults.yaml
scripts/              # download_data.py, tokenize_corpus.py
reports/              # technical_note.md
```

## License

MIT
