"""Training loop. Phase 3."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import torch
import torch.nn as nn


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average CE across the batch. Numerically stable."""
    raise NotImplementedError("Phase 3")


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    raise NotImplementedError("Phase 3")


class AdamW(torch.optim.Optimizer):
    """AdamW with decoupled weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):  # noqa: ARG002
        raise NotImplementedError("Phase 3")


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * (it + 1) / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / max(1, cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    raise NotImplementedError("Phase 3")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    raise NotImplementedError("Phase 3")


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    raise NotImplementedError("Phase 3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    raise NotImplementedError(f"Phase 3 — wire {args}")


if __name__ == "__main__":
    main()
