"""Closed-form scaling/memory/latency estimator. Phase 5. SIMULATED side.

Pure-Python — no torch import. Anchored to JAX Scaling Book worked examples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    tied_embeddings: bool = False


@dataclass(frozen=True)
class RuntimeSpec:
    batch_size: int
    seq_len: int
    bytes_param: int = 2
    bytes_grad: int = 2
    bytes_opt_state: int = 4
    num_opt_states: int = 2
    bytes_kv: int = 1
    act_coeff: int = 7  # 20 = save all, 7 = save big matmuls, 1 = block rematerialization


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    peak_flops_per_s: float
    memory_bandwidth_bytes_per_s: float
    device_memory_bytes: int
    mfu: float = 0.4


def param_count(m: ModelSpec) -> int:
    per_layer = (
        3 * m.d_model * m.d_ff
        + 2 * m.d_model * (m.n_heads + m.n_kv_heads) * m.head_dim
        + m.d_model
    )
    embed = m.d_model * m.vocab_size if m.tied_embeddings else 2 * m.d_model * m.vocab_size
    return m.n_layers * per_layer + embed


def train_flops_per_step(m: ModelSpec, r: RuntimeSpec) -> int:
    B, T = r.batch_size, r.seq_len
    return m.n_layers * (
        18 * B * T * m.d_model * m.d_ff
        + 12 * B * T * m.d_model * (m.n_heads + m.n_kv_heads) * m.head_dim
        + 12 * B * T * T * m.n_heads * m.head_dim
    )


def train_flops_rule_of_thumb(m: ModelSpec, tokens: int) -> int:
    return 6 * param_count(m) * tokens


def activation_bytes(m: ModelSpec, r: RuntimeSpec) -> int:
    return 2 * r.act_coeff * r.batch_size * r.seq_len * m.d_model * m.n_layers


def kv_cache_bytes_per_sequence(m: ModelSpec, seq_len: int, bytes_kv: int = 1) -> int:
    return 2 * seq_len * m.n_layers * m.n_kv_heads * m.head_dim * bytes_kv


def total_train_memory_bytes(m: ModelSpec, r: RuntimeSpec) -> int:
    P = param_count(m)
    return (
        P * r.bytes_param
        + P * r.bytes_grad
        + P * r.bytes_opt_state * r.num_opt_states
        + activation_bytes(m, r)
    )


def decode_step_time_lower_bound_s(
    batch_size: int,
    kv_cache_bytes_per_seq: int,
    param_bytes_total: int,
    total_memory_bandwidth_bytes_per_s: float,
) -> float:
    return (
        batch_size * kv_cache_bytes_per_seq + param_bytes_total
    ) / total_memory_bandwidth_bytes_per_s


def train_time_estimate_s(
    train_flops: float,
    hardware: HardwareSpec,
    num_devices: int = 1,
) -> float:
    """Wall-clock training time given total FLOPs, hardware peak, MFU, parallelism.

    t = total_flops / (num_devices * peak_flops_per_s * mfu)
    """
    return train_flops / (num_devices * hardware.peak_flops_per_s * hardware.mfu)


def roofline_intensity_threshold(hardware: HardwareSpec) -> float:
    """Arithmetic-intensity (FLOPs / byte) above which a kernel is compute-bound."""
    return hardware.peak_flops_per_s / hardware.memory_bandwidth_bytes_per_s


def predict_tokens_per_second_v2(
    model: ModelSpec,
    runtime: RuntimeSpec,
    hardware: HardwareSpec,
    overhead_floor_s: float = 0.0,
) -> dict:
    """Roofline-gated tokens/s prediction.

    Computes arithmetic intensity (forward FLOPs / bytes touched) and picks
    the tighter of (memory-bound bandwidth bound, compute-bound peak-FLOPs bound).

    Returns a dict with:
      tokens_per_sec: the chosen prediction
      bound: "memory" or "compute"
      arithmetic_intensity: FLOPs / byte for the step
      compute_bandwidth_threshold: hardware threshold to cross
      tokens_per_sec_memory: memory-bound prediction (for reference)
      tokens_per_sec_compute: compute-bound prediction (for reference)
    """
    P = param_count(model)
    param_bytes = P * runtime.bytes_param
    kv_bytes = kv_cache_bytes_per_sequence(model, runtime.seq_len, runtime.bytes_kv)

    # Forward-pass FLOPs per step (rule of thumb: 2 * P * tokens_per_step).
    tokens_per_step = runtime.batch_size * runtime.seq_len
    forward_flops_per_step = 2 * P * tokens_per_step

    # Bytes touched per step (the formula's existing denominator).
    bytes_per_step = runtime.batch_size * kv_bytes + param_bytes

    intensity = forward_flops_per_step / bytes_per_step
    threshold = roofline_intensity_threshold(hardware)

    tps_memory = tokens_per_second_upper_bound(
        runtime.batch_size, kv_bytes, param_bytes, hardware.memory_bandwidth_bytes_per_s
    )
    # Compute-bound rate: peak_flops * mfu / FLOPs_per_token.
    flops_per_token = 2 * P
    tps_compute = hardware.peak_flops_per_s * hardware.mfu / flops_per_token

    if intensity > threshold:
        chosen_bound = "compute"
        chosen_tps = tps_compute
    else:
        chosen_bound = "memory"
        chosen_tps = tps_memory

    # Optional fixed per-step launch-overhead floor: step time can't dip below it.
    overhead_capped = False
    if overhead_floor_s > 0 and chosen_tps > 0:
        step_time_unbounded = tokens_per_step / chosen_tps
        if step_time_unbounded < overhead_floor_s:
            chosen_tps = tokens_per_step / overhead_floor_s
            overhead_capped = True

    return {
        "tokens_per_sec": chosen_tps,
        "bound": chosen_bound,
        "arithmetic_intensity": intensity,
        "compute_bandwidth_threshold": threshold,
        "tokens_per_sec_memory": tps_memory,
        "tokens_per_sec_compute": tps_compute,
        "overhead_capped": overhead_capped,
    }


def tokens_per_second_upper_bound(
    batch_size: int,
    kv_cache_bytes_per_seq: int,
    param_bytes_total: int,
    total_memory_bandwidth_bytes_per_s: float,
) -> float:
    num = batch_size * total_memory_bandwidth_bytes_per_s
    den = batch_size * kv_cache_bytes_per_seq + param_bytes_total
    return num / den
