# Where does the simple memory-bandwidth estimator break on Apple silicon?

> Independent reimplementation of a small decoder-only transformer + a closed-form
> scaling / memory / latency estimator, calibrated against measured benchmarks on
> a MacBook. Inspired by Stanford CS336 and the JAX Scaling Book.

## Question

The JAX Scaling Book and several systems-engineering references use a bandwidth-bound
upper bound for decode-style inference:

```
tokens/s_max  =  (batch · total_memory_bandwidth) / (batch · kv_cache_per_seq + parameter_bytes)
```

This is an upper bound. It assumes the kernel is **purely memory-bound** — every
parameter and KV-cache byte streams through HBM (or unified memory on Apple
silicon) at peak bandwidth, with compute essentially free in comparison. It
collapses several real effects into one number.

Concrete question for this note: **how far off is this upper bound on a real
MacBook CPU run with a small model?** Specifically:

- Which dimension drives error: batch size, sequence length, or model size?
- Where is the estimator usefully tight (≤ 15% error)?
- Where does it overestimate by 2–3×, and what mechanism explains it?

## Setup

| Item | Value |
|---|---|
| Hardware | Apple silicon CPU (M-series), unified memory |
| Peak memory bandwidth (assumed) | 200 GB/s |
| Model | decoder-only, RMSNorm + SwiGLU + RoPE, 4 layers |
| `d_model` | 128 |
| `d_ff` | 512 |
| `n_heads` (= `n_kv_heads`) | 4 |
| `head_dim` | 32 |
| `vocab_size` | 10,000 |
| Tied embeddings | yes |
| `param_count` | 2,329,088 (≈ 4.66 MB at bf16) |
| Sweep | `seq_len ∈ {64, 128, 256}` × `batch_size ∈ {1, 2, 4}` |
| Steps per cell | 5 forward passes after one warmup |

Reproduce:

```bash
uv sync
uv run python -m lfslab.profile_train --config configs/tiny_cpu.yaml \
    --seq-lens 64 128 256 --batch-sizes 1 2 4 --num-steps 5
```

## Result: measured vs estimated tokens/s

| seq_len | batch | KV bytes / seq | Estimated tok/s (upper bound) | Measured tok/s | Error |
|--------:|------:|---------------:|------------------------------:|---------------:|------:|
|      64 |     1 |         65,536 |                        42,340 |         28,818 | **−31.9 %** |
|      64 |     2 |         65,536 |                        83,520 |         36,801 | **−55.9 %** |
|      64 |     4 |         65,536 |                       162,591 |         55,678 | **−65.8 %** |
|     128 |     1 |        131,072 |                        41,760 |         38,050 |    **−8.9 %** |
|     128 |     2 |        131,072 |                        81,296 |         56,159 | **−30.9 %** |
|     128 |     4 |        131,072 |                       154,367 |         53,351 | **−65.4 %** |
|     256 |     1 |        262,144 |                        40,648 |         45,742 |    **+12.5 %** |
|     256 |     2 |        262,144 |                        77,183 |         47,065 | **−39.0 %** |
|     256 |     4 |        262,144 |                       140,185 |         49,212 | **−64.9 %** |

The "Error" column is `(measured − estimated) / estimated · 100 %`. Negative means
the upper bound was too optimistic; positive means measured exceeded the predicted
upper bound (which can happen if the assumed bandwidth is understated).

## What the data says

**The upper bound is usefully tight (≤ 15 %) in exactly one regime here:**
`batch = 1` at moderate `seq_len`. At `seq=128, batch=1` it overshoots measured
by 8.9 %; at `seq=256, batch=1` it actually **under**-predicts by 12.5 %.

**The error grows monotonically with batch size**, blowing out to roughly −65 %
at `batch = 4` across every sequence length. The formula predicts linear scaling
of throughput in `batch`, because more parallel sequences amortize the same
parameter-load cost. Measured throughput is sublinear: between `batch = 1` and
`batch = 4`, measured tokens/s only grows by **~1.9×** instead of the predicted
~4×.

**The model is tiny (~2.3 M params, ~4.7 MB at bf16) and the CPU is not memory-
bandwidth-bound on it.** The bandwidth-only upper bound assumes streaming reads
of the parameter matrix dominate step time. At this model size, parameters
comfortably fit in L2/L3 cache after warmup, so the dominant cost shifts to
compute (matmul throughput, softmax, RoPE, RMSNorm reductions) and to
arithmetic intensity, neither of which the formula sees. Batch parallelism then
doesn't help proportionally because the compute side is the new bottleneck.

**Why does `seq=256, batch=1` slightly under-predict?** The denominator in the
upper bound grows linearly with `seq_len` via the KV cache term, but the model
also amortizes more compute work over a longer sequence (the cost of the parameter
load is the same, but more tokens are produced per load). At small models
that fit in cache, this amortization wins over the KV-cache cost, and the
"upper" bound flips into a slight under-estimate of what the hardware can do
when re-using cached parameters.

## What I'd change in a v2 of the estimator

1. **Add a roofline-style switch.** Compute arithmetic intensity per step
   (FLOPs / bytes touched). If it exceeds `peak_flops / bandwidth ≈ 156 FLOPs/byte`
   on an A100, the kernel is compute-bound and the bandwidth formula is the
   wrong bound. The repo's `roofline_intensity_threshold(hardware)` already
   exposes this number; the estimator should consult it.
2. **Cache-aware parameter cost.** When `param_bytes < L2_size_per_core`,
   parameters are reused from cache after the first step. Replace the
   per-step parameter-load term with a one-time cost amortized over steps.
3. **Sub-linear batch scaling on small models.** Add a soft cap at the
   measured compute ceiling rather than assuming linear scaling forever.

These changes turn the formula from a single number into a piecewise estimate.
The current single-number form is most useful when the model is large enough
that parameters demonstrably do not fit in cache — which, for a MacBook study
of model implementations, is rarely the case at the smoke-training scale.

## Caveats

- 5 forward passes per cell is small and noisy; numbers within ±10 % are not
  meaningful as deltas, only as ballpark.
- "Peak memory bandwidth" is an Apple-published number for the full memory
  controller; Python-PyTorch-on-CPU never approaches that limit. A more honest
  parameter is *measured sustained bandwidth* on a known kernel.
- I tested CPU only here. On MPS (Apple's GPU backend) the picture shifts
  because MPS launches Metal kernels — the estimator may behave very differently
  there, and a follow-up note is the natural next step.

## Reproducibility

- Code: <https://github.com/yeungjosh/llm-from-scratch-lab>
- Phase 0–6 commit: `git log --oneline` shows one commit per phase
- 64-test suite, including 8 anchor tests against JAX Scaling Book worked
  examples (16B param check, 512 KiB/token KV check, 6.3e24 FLOPs check,
  2.5 ms decode lower bound, 70B Llama-shape param check, etc.)
- Sweep script: `lfslab.profile_train.sweep(...)` writes a `results/local/sweep_cpu.json`
  consumed by the Streamlit explorer's "Measured vs estimated" tab.
