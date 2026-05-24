"""Device catalogue + device selection. Phase 4-5."""

from __future__ import annotations

from .estimator import HardwareSpec

# Peak FLOPs are bf16/fp16 dense compute; memory bandwidth is per-GPU HBM.
# Sources: NVIDIA datasheets; Apple silicon figures are approximate published numbers.
CATALOGUE: dict[str, HardwareSpec] = {
    "cpu_apple_m_series": HardwareSpec(
        name="Apple silicon CPU",
        peak_flops_per_s=2e11,
        memory_bandwidth_bytes_per_s=2e11,
        device_memory_bytes=32 * 1024**3,
    ),
    "mps_m3_max": HardwareSpec(
        name="Apple M3 Max GPU (MPS)",
        peak_flops_per_s=14e12,
        memory_bandwidth_bytes_per_s=400e9,
        device_memory_bytes=64 * 1024**3,
    ),
    "t4": HardwareSpec(
        name="NVIDIA T4",
        peak_flops_per_s=65e12,
        memory_bandwidth_bytes_per_s=320e9,
        device_memory_bytes=16 * 1024**3,
    ),
    "a10g": HardwareSpec(
        name="NVIDIA A10G",
        peak_flops_per_s=125e12,
        memory_bandwidth_bytes_per_s=600e9,
        device_memory_bytes=24 * 1024**3,
    ),
    "a100_80g": HardwareSpec(
        name="NVIDIA A100 80GB",
        peak_flops_per_s=312e12,
        memory_bandwidth_bytes_per_s=2039e9,
        device_memory_bytes=80 * 1024**3,
    ),
    "h100_80g": HardwareSpec(
        name="NVIDIA H100 80GB",
        peak_flops_per_s=989e12,
        memory_bandwidth_bytes_per_s=3350e9,
        device_memory_bytes=80 * 1024**3,
    ),
}


def pick_device() -> str:
    """Return the best torch device available."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"
