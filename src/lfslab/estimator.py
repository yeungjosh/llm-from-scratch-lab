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


def tokens_per_second_upper_bound(
    batch_size: int,
    kv_cache_bytes_per_seq: int,
    param_bytes_total: int,
    total_memory_bandwidth_bytes_per_s: float,
) -> float:
    num = batch_size * total_memory_bandwidth_bytes_per_s
    den = batch_size * kv_cache_bytes_per_seq + param_bytes_total
    return num / den
