"""Dataset / dataloader helpers. Phase 1 (tokenization) + Phase 3 (batching)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
):
    """Sample (inputs, labels) where labels are inputs shifted by one."""
    raise NotImplementedError("Phase 3")


def load_token_stream(path: str) -> np.ndarray:
    """Load a 1-D uint16/uint32 token array (mmapped where possible)."""
    raise NotImplementedError("Phase 1")
