"""Profiler harness + sweep. Phase 4. MEASURED side."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from .model import TransformerLM


def profile_one(
    model: TransformerLM,
    seq_len: int,
    batch_size: int,
    num_steps: int = 20,
    device: str = "cpu",
    trace_dir: str | Path | None = None,
) -> dict:
    """One forward-only timing run. Returns metrics dict.

    When `trace_dir` is given, also writes a Chrome-trace JSON via torch.profiler.
    """
    model = model.to(device)
    vocab = model.vocab_size
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    # Warmup.
    with torch.no_grad():
        model(x)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_steps):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    step_time_ms = (elapsed / num_steps) * 1000
    tokens_per_sec = (batch_size * seq_len * num_steps) / elapsed

    peak = 0
    if device == "cuda":
        peak = int(torch.cuda.max_memory_allocated())

    result: dict = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "device": device,
        "step_time_ms": step_time_ms,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_bytes": peak,
    }

    if trace_dir is not None:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"trace_seq{seq_len}_b{batch_size}.json"
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, record_shapes=False) as prof, torch.no_grad():
            model(x)
        prof.export_chrome_trace(str(trace_path))
        result["trace_path"] = str(trace_path)

    return result


def sweep(
    model: TransformerLM,
    seq_lens: list[int],
    batch_sizes: list[int],
    num_steps: int = 10,
    device: str = "cpu",
    trace_dir: str | Path | None = None,
    out_path: str | Path | None = None,
) -> list[dict]:
    results: list[dict] = []
    for s in seq_lens:
        for b in batch_sizes:
            results.append(
                profile_one(
                    model,
                    seq_len=s,
                    batch_size=b,
                    num_steps=num_steps,
                    device=device,
                    trace_dir=trace_dir,
                )
            )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--out", default="results/local/sweep.json")
    p.add_argument("--trace-dir", default="results/local/traces")
    args = p.parse_args()

    import yaml

    from .hardware import pick_device
    from .train import build_model_from_config

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = cfg["training"].get("device") or pick_device()
    model = build_model_from_config(cfg)

    rows = sweep(
        model,
        seq_lens=args.seq_lens,
        batch_sizes=args.batch_sizes,
        num_steps=args.num_steps,
        device=device,
        trace_dir=args.trace_dir,
        out_path=args.out,
    )
    print(f"wrote {len(rows)} rows to {args.out}; traces under {args.trace_dir}/")


if __name__ == "__main__":
    main()
