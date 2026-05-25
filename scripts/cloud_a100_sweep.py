"""Phase 8 cloud validation sweep — 1.36B-param model on an NVIDIA GPU.

Why this model size:
At d_model=2048, d_ff=8192, L=24, the parameter tensor is ~2.7 GB at bf16.
That is comfortably larger than the L2 cache on any current GPU (A100: ~40 MB
per SM partition, H100: ~50 MB), so parameters genuinely stream from HBM each
step — i.e., we are in the regime the bandwidth-bound tokens/s upper bound
was designed for. If the estimator is going to be tight anywhere, it should
be tight here.

Usage on the pod:
    uv run python scripts/cloud_a100_sweep.py

Output: results/cloud/sweep_a100.json with one row per (seq_len, batch_size).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from lfslab.estimator import (
    ModelSpec,
    kv_cache_bytes_per_sequence,
    param_count,
    tokens_per_second_upper_bound,
)
from lfslab.hardware import CATALOGUE
from lfslab.model import TransformerLM
from lfslab.profile_train import sweep


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. This script is meant to run on a cloud GPU pod."
        )
    device_name = torch.cuda.get_device_name(0)
    print(f"[cloud-sweep] device: {device_name}")
    print(f"[cloud-sweep] cuda: {torch.version.cuda}, torch: {torch.__version__}")

    model = TransformerLM(
        vocab_size=32000,
        context_length=4096,
        d_model=2048,
        num_layers=24,
        num_heads=16,
        d_ff=8192,
        tied_embeddings=False,
    )
    spec = ModelSpec(
        d_model=2048, d_ff=8192, n_layers=24, n_heads=16, n_kv_heads=16,
        head_dim=128, vocab_size=32000, tied_embeddings=False,
    )
    n_params = param_count(spec)
    param_bytes = n_params * 2  # bf16
    print(f"[cloud-sweep] model: {n_params:,} params  (~{param_bytes / 1e9:.2f} GB at bf16)")

    Path("results/cloud").mkdir(parents=True, exist_ok=True)

    # Move model in bf16 to save HBM during the sweep itself.
    model = model.to(torch.bfloat16).to("cuda")

    seq_lens = [256, 1024, 2048, 4096]
    batch_sizes = [1, 2, 4, 8]
    t0 = time.time()
    rows = sweep(
        model,
        seq_lens=seq_lens,
        batch_sizes=batch_sizes,
        num_steps=10,
        device="cuda",
        out_path="results/cloud/sweep_a100.json",
    )
    print(f"[cloud-sweep] sweep done in {time.time() - t0:.1f}s — {len(rows)} cells")

    # Pick hardware spec for predicted tokens/s.
    if "A100" in device_name:
        hw = CATALOGUE["a100_80g"]
    elif "H100" in device_name:
        hw = CATALOGUE["h100_80g"]
    elif "A10G" in device_name or "A10" in device_name:
        hw = CATALOGUE["a10g"]
    elif "T4" in device_name:
        hw = CATALOGUE["t4"]
    else:
        hw = CATALOGUE["a100_80g"]  # default
        print(f"[cloud-sweep] WARN: unknown device {device_name}; using a100 hw spec")

    print(f"\n[cloud-sweep] estimator vs measured ({hw.name}, "
          f"peak_bw={hw.memory_bandwidth_bytes_per_s:.2e} B/s):\n")
    print(f"  {'seq':>4} {'bs':>3} {'step ms':>9} {'meas tok/s':>12} "
          f"{'est tok/s':>12} {'err %':>8}")
    summary_rows = []
    for r in rows:
        kv = kv_cache_bytes_per_sequence(spec, r["seq_len"])
        est = tokens_per_second_upper_bound(
            r["batch_size"], kv, param_bytes, hw.memory_bandwidth_bytes_per_s
        )
        err = (r["tokens_per_sec"] - est) / est * 100
        print(f"  {r['seq_len']:>4} {r['batch_size']:>3} "
              f"{r['step_time_ms']:>9.2f} {r['tokens_per_sec']:>12.0f} "
              f"{est:>12.0f} {err:>+8.1f}")
        summary_rows.append({**r, "tokens_per_sec_estimated": est, "pct_error": err})

    # Save augmented JSON.
    Path("results/cloud/sweep_a100_with_estimator.json").write_text(
        json.dumps(
            {
                "device_name": device_name,
                "hardware_used_in_estimator": hw.name,
                "n_params": n_params,
                "param_bytes_bf16": param_bytes,
                "rows": summary_rows,
            },
            indent=2,
        )
    )
    print("\n[cloud-sweep] wrote results/cloud/sweep_a100.json + sweep_a100_with_estimator.json")
    print("[cloud-sweep] download both files before terminating the pod.")


if __name__ == "__main__":
    main()
