"""CS336 Assignment 1 adapter signatures — mirrored verbatim, wired to lfslab."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    return F.linear(in_features, weights)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    return F.embedding(token_ids, weights)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from lfslab.model import swiglu_functional

    return swiglu_functional(in_features, w1_weight, w2_weight, w3_weight)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    from lfslab.model import scaled_dot_product_attention

    return scaled_dot_product_attention(Q, K, V, mask=mask)


def _mha_no_rope(
    d_model: int,
    num_heads: int,
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    o_w: Tensor,
    x: Tensor,
) -> Tensor:
    from lfslab.model import scaled_dot_product_attention

    H = num_heads
    D = d_model // num_heads
    *lead, T, _ = x.shape
    q = F.linear(x, q_w).view(*lead, T, H, D).transpose(-2, -3)
    k = F.linear(x, k_w).view(*lead, T, H, D).transpose(-2, -3)
    v = F.linear(x, v_w).view(*lead, T, H, D).transpose(-2, -3)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    out = scaled_dot_product_attention(q, k, v, mask=mask)
    out = out.transpose(-2, -3).contiguous().view(*lead, T, d_model)
    return F.linear(out, o_w)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    return _mha_no_rope(
        d_model, num_heads,
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        in_features,
    )


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    from lfslab.model import _rope_cache, apply_rope, scaled_dot_product_attention

    H = num_heads
    D = d_model // num_heads
    *lead, T, _ = in_features.shape
    q = F.linear(in_features, q_proj_weight).view(*lead, T, H, D).transpose(-2, -3)
    k = F.linear(in_features, k_proj_weight).view(*lead, T, H, D).transpose(-2, -3)
    v = F.linear(in_features, v_proj_weight).view(*lead, T, H, D).transpose(-2, -3)
    cos, sin = _rope_cache(D, theta, max_seq_len, device=in_features.device)
    if token_positions is None:
        token_positions = torch.arange(T, device=in_features.device)
    q = apply_rope(q, token_positions, cos, sin)
    k = apply_rope(k, token_positions, cos, sin)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=in_features.device))
    out = scaled_dot_product_attention(q, k, v, mask=mask)
    out = out.transpose(-2, -3).contiguous().view(*lead, T, d_model)
    return F.linear(out, o_proj_weight)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    from lfslab.model import _rope_cache, apply_rope

    cos, sin = _rope_cache(d_k, theta, max_seq_len, device=in_query_or_key.device)
    return apply_rope(in_query_or_key, token_positions, cos, sin)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    from lfslab.model import TransformerBlock

    block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
    missing, unexpected = block.load_state_dict(weights, strict=False)
    bad_missing = [k for k in missing if "cos_cache" not in k and "sin_cache" not in k]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch: missing={bad_missing} unexpected={unexpected}"
        )
    block = block.to(in_features.device)
    with torch.no_grad():
        return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    from lfslab.model import TransformerLM

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    missing, unexpected = model.load_state_dict(weights, strict=False)
    bad_missing = [k for k in missing if "cos_cache" not in k and "sin_cache" not in k]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch: missing={bad_missing} unexpected={unexpected}"
        )
    model = model.to(in_indices.device)
    with torch.no_grad():
        return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from lfslab.model import rmsnorm_functional

    return rmsnorm_functional(in_features, weights, eps)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    from lfslab.model import silu

    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError  # Phase 3


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    from lfslab.model import softmax

    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    raise NotImplementedError  # Phase 3


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    raise NotImplementedError  # Phase 3


def get_adamw_cls() -> Any:
    raise NotImplementedError  # Phase 3


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    from lfslab.train import get_lr_cosine_schedule

    return get_lr_cosine_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    raise NotImplementedError  # Phase 3


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    raise NotImplementedError  # Phase 3


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    from lfslab.tokenizer import Tokenizer

    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    from lfslab.tokenizer import train_bpe

    return train_bpe(input_path, vocab_size, special_tokens)
