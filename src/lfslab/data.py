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
    """Save token ids as a 1-D .npy array (uint16 if all ids fit, else uint32).

    An empty ``ids`` is saved as an empty uint16 array rather than raising.
    """
    ids = np.asarray(ids)
    dtype = np.uint16 if ids.size == 0 or ids.max() < 2**16 else np.uint32
    arr = ids.astype(dtype)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
