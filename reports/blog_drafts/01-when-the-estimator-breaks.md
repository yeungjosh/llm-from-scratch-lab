# When the simple transformer cost estimator breaks (MacBook edition)

*Draft. Target: a short lab note, 800 to 1200 words.*

A common back-of-envelope formula for decoding throughput from the JAX
Scaling Book reads:

```
tokens/s_max  =  (batch · memory_bandwidth) / (batch · kv_cache_bytes + parameter_bytes)
```

This is a bandwidth-bound upper bound. It is the formula you sketch on a
napkin when someone asks "how fast can I run inference on this?", and
it is the formula a fresh repo's `estimator.py` tends to ship with.

I implemented a small decoder-only transformer (RMSNorm + SwiGLU + RoPE,
2.3 M parameters), wrote that formula as the centerpiece of an explorer,
and then asked the obvious question: **does the formula actually predict
what my MacBook does?**

Two answers, both negative results.

## CPU answer: the bound is too optimistic, by a lot

Ran a 3 × 3 sweep (`seq_len ∈ {64, 128, 256}` × `batch ∈ {1, 2, 4}`, five
forward passes per cell on Apple silicon CPU).

The formula was tight (≤ 15 % error) at exactly **one cell**: `batch=1, seq=128`
came in at −8.9 %. Everywhere else, the upper bound overshot measured
throughput by 30 to 66 %. The error grew monotonically with batch size:
between `batch=1` and `batch=4`, the formula predicted ~4× more
throughput, but measured throughput grew by only ~1.9×.

Why? The model is 4.66 MB at bf16. It fits in L2/L3 cache. After a
single warmup pass, you stop streaming parameters from main memory.
You start streaming them from cache. So **bandwidth is not the
bottleneck**, the compute side is. Once you are compute-bound, more
batches do not help you proportionally; you are spinning the same CPU
cores faster, not loading more parameter bytes per unit time.

## MPS answer: the bound is too *pessimistic* at moderate seq_len

Then I ran the same model on MPS (Apple's PyTorch GPU backend) with a
slightly wider sweep, using the catalogued M3-Max peak bandwidth of
400 GB/s.

The picture flipped:

- `seq=64`: formula too high by 70 to 94 %. (Launch overhead dominates;
  the formula has no fixed-overhead term.)
- `seq=128`: formula too high by ~40 to 50 %.
- `seq=256`: formula too **low** by 16 to 43 %.
- `seq=512, batch ∈ {1, 2, 4}`: formula too low by **95 to 154 %**.
  Measured throughput at `seq=512, batch=2` was 355 k tokens/sec,
  more than 2× what the upper bound says is possible.
- `seq=512, batch=8`: step time jumped from ~3 ms at batch=4 to 21 ms.
  Measured throughput collapsed from 462 k to 194 k. The formula did
  not see this cliff coming.

Two of those are particularly interesting.

The +135 % at `seq=512, batch=1` means one of two things must be true.
Either (a) the "peak memory bandwidth" number I plugged in (400 GB/s)
understates what Metal kernels actually achieve on this model, or
(b) the working set is small enough that the kernel never goes to
main memory at all (it streams from on-chip caches) and the
"bandwidth" term in the denominator is the wrong constant entirely.
Either way, the formula's notion of "upper bound" no longer holds:
the hardware is moving faster than the formula said it could.

The cliff at `batch=8` is the other failure mode. The model is
deterministic; the only thing that changed between batch=4 and
batch=8 is the size of the activation tensor. Somewhere between
those two points, the working set crossed a threshold (probably
spilling out of some level of on-chip cache, given the unified-memory
architecture), and step time multiplied by ~5. Anyone trusting the
linear-in-batch model and picking a batch size to maximize tokens/s
would size their batch wrong here.

## What this means

I do not think the formula is wrong. I think it is *narrowly applicable*,
and it is dangerous to use it as a single number without checking the
regime you are in.

What I would change in v2:

1. **Roofline switch.** Compute the kernel's arithmetic intensity
   (FLOPs / bytes touched) per step. If it exceeds
   `peak_flops / peak_bandwidth`, the kernel is compute-bound and
   the bandwidth formula is the wrong upper bound; use a different
   one. The threshold is ~156 FLOPs/byte on an A100.
2. **Fixed-overhead floor.** Add a per-step launch cost that the
   formula cannot dip below. On MPS for this model it was about
   2.5 ms; the formula's prediction of 0.2 ms at `seq=64` is
   physically impossible there.
3. **Cache-aware parameter cost.** When `param_bytes` < the device's
   cache hierarchy, parameters do not stream from HBM after warmup.
   Replace the per-step parameter-load term with a one-time cost.
4. **Memory-pressure cliff detection.** Flag configurations whose
   working set crosses a known threshold. Do not pretend throughput
   scales linearly in batch past that point.

Each of these turns the formula from a single number into a piecewise
estimate. That is more work to maintain, but it is much more honest
about what the hardware does.

## Why I bothered

The whole point of doing this on a MacBook was to test the formulas
against something. The estimator is a tool for fast-feedback decision
making (should I rent an A100 or an H100? is this batch size going to
work? how long will this training run actually take?). You only trust
a tool like that if you have calibrated it against real measurements,
and you have written down where you stopped trusting it.

Calibration without disagreement is suspicious. Two real
disagreements per device-class (CPU: optimistic at high batch;
MPS: pessimistic at moderate seq, cliff at seq=512/batch=8) is
exactly the kind of signal that makes the next version of the
formula better. The plan is now to repeat this on a cloud A100 or
A10G, at a model size where parameters genuinely do not fit in
cache, and see whether the formula recovers its accuracy in the
regime it was designed for.

Code, full sweep data, and the estimator anchor tests against
published worked examples are in the repo:
**https://github.com/yeungjosh/llm-from-scratch-lab**

## What's next

- Re-run the same model on a rented A100 / A10G; expect the formula
  to be tight in that regime.
- Implement the four estimator-v2 fixes above and verify each one
  improves measured-vs-estimated error.
- Move from synthetic / micro-benchmarks to a real training run on
  TinyStories with a recorded loss curve and val-PPL endpoint.
