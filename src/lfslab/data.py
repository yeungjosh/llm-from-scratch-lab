"""Dataset / dataloader helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
):
    """Sample (inputs, labels) where labels are inputs shifted by one. Phase 3."""
    raise NotImplementedError("Phase 3")


def load_token_stream(path: str | Path) -> np.ndarray:
    """Load a 1-D token array (mmapped)."""
    return np.load(path, mmap_mode="r")


def save_token_stream(path: str | Path, ids: list[int] | np.ndarray) -> None:
    arr = np.asarray(ids, dtype=np.uint16 if max(ids) < 2**16 else np.uint32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
