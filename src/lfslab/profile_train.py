"""Profiler harness + sweep. Phase 4. MEASURED side."""

from __future__ import annotations

import argparse


def profile_one(config_path: str, seq_len: int, batch_size: int, num_steps: int = 20) -> dict:
    """One profiled run; returns dict of metrics + path to trace file."""
    raise NotImplementedError("Phase 4")


def sweep(config_path: str, seq_lens: list[int], batch_sizes: list[int]) -> list[dict]:
    raise NotImplementedError("Phase 4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--out", default="results/local/sweep.json")
    args = parser.parse_args()
    raise NotImplementedError(f"Phase 4 — wire {args}")


if __name__ == "__main__":
    main()
