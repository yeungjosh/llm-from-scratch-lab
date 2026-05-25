"""Cached autoregressive decode — educational, not perf-optimized.

Demonstrates the principle of KV-caching: instead of re-running the full
sequence through every block on each generated token (O(T) per token,
O(T²) total), we keep K/V tensors per layer and grow them by one row per
step (O(1) per token in attention beyond the current position).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import TransformerBlock, TransformerLM, apply_rope, scaled_dot_product_attention


@torch.no_grad()
def full_recompute_decode(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedy decode by re-running the full prefix on every step."""
    ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(ids)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


def _block_with_cache(
    block: TransformerBlock,
    x: torch.Tensor,
    positions: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One transformer block forward with optional past KV cache.

    cache: (K_past, V_past) of shape (B, H, T_past, D) each, or None.
    Returns (block_output, (K_full, V_full)).
    """
    attn = block.attn
    H = attn.num_heads
    D = attn.head_dim
    d_model = attn.d_model

    h = block.ln1(x)
    B, T_new, _ = h.shape
    q = F.linear(h, attn.q_proj.weight).view(B, T_new, H, D).transpose(1, 2)
    k_new = F.linear(h, attn.k_proj.weight).view(B, T_new, H, D).transpose(1, 2)
    v_new = F.linear(h, attn.v_proj.weight).view(B, T_new, H, D).transpose(1, 2)

    if attn.rope is not None:
        q = apply_rope(q, positions, attn.rope.cos_cache, attn.rope.sin_cache)
        k_new = apply_rope(k_new, positions, attn.rope.cos_cache, attn.rope.sin_cache)

    if cache is None:
        k_full, v_full = k_new, v_new
    else:
        k_past, v_past = cache
        k_full = torch.cat([k_past, k_new], dim=2)
        v_full = torch.cat([v_past, v_new], dim=2)

    T_full = k_full.shape[2]
    if T_new == T_full:
        # Prefill — causal mask within the prompt.
        mask = torch.tril(torch.ones(T_new, T_full, dtype=torch.bool, device=x.device))
    else:
        # Single-step decode — every new query can attend to all cached positions.
        mask = None

    attn_out = scaled_dot_product_attention(q, k_full, v_full, mask=mask)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_new, d_model)
    attn_out = F.linear(attn_out, attn.output_proj.weight)

    x = x + attn_out
    x = x + block.ffn(block.ln2(x))
    return x, (k_full, v_full)


@torch.no_grad()
def decode_with_cache(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedy decode with per-layer KV cache."""
    ids = prompt_ids.clone()
    _B, T_prompt = ids.shape

    caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(model.layers)

    # Prefill on the whole prompt.
    x = model.token_embeddings(ids)
    positions = torch.arange(T_prompt, device=ids.device)
    for i, layer in enumerate(model.layers):
        x, caches[i] = _block_with_cache(layer, x, positions, cache=None)
    h_final = model.ln_final(x)
    logits = F.linear(h_final[:, -1, :], model.lm_head.weight)
    next_id = logits.argmax(dim=-1, keepdim=True)
    ids = torch.cat([ids, next_id], dim=1)

    # Decode loop: one token per step using the cache.
    for _ in range(1, max_new_tokens):
        new_id = ids[:, -1:]
        pos = torch.tensor([ids.shape[1] - 1], device=ids.device)
        x = model.token_embeddings(new_id)
        for i, layer in enumerate(model.layers):
            x, caches[i] = _block_with_cache(layer, x, pos, cache=caches[i])
        h_final = model.ln_final(x)
        logits = F.linear(h_final[:, -1, :], model.lm_head.weight)
        next_id = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)

    return ids
