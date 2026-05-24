"""Decoder-only Transformer (RMSNorm + SwiGLU + RoPE + causal MHA, pre-norm).

CS336 A1-compatible state_dict keys:
  token_embeddings.weight
  layers.{i}.attn.q_proj.weight
  layers.{i}.attn.k_proj.weight
  layers.{i}.attn.v_proj.weight
  layers.{i}.attn.output_proj.weight
  layers.{i}.ln1.weight
  layers.{i}.ffn.w1.weight
  layers.{i}.ffn.w2.weight
  layers.{i}.ffn.w3.weight
  layers.{i}.ln2.weight
  ln_final.weight
  lm_head.weight
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- functional primitives ---------------------------------------------------


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def rmsnorm_functional(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x32 / rms * weight.to(torch.float32)).to(in_dtype)


def swiglu_functional(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU: w2(silu(x @ w1.T) * (x @ w3.T))."""
    return F.linear(silu(F.linear(x, w1)) * F.linear(x, w3), w2)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.amax(dim=dim, keepdim=True)
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # mask is True where attention is allowed.
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def _rope_cache(d_k: int, theta: float, max_seq_len: int, device, dtype=torch.float32):
    assert d_k % 2 == 0, "RoPE requires even head_dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, inv_freq)  # (max_seq_len, d_k/2)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
) -> torch.Tensor:
    """x: (..., seq_len, d_k). positions: (..., seq_len) integer."""
    cos = cos_cache[positions]  # (..., seq_len, d_k/2)
    sin = sin_cache[positions]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# --- nn.Module wrappers ------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_functional(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu_functional(x, self.w1.weight, self.w2.weight, self.w3.weight)


class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int) -> None:
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        cos, sin = _rope_cache(d_k, theta, max_seq_len, device=torch.device("cpu"))
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return apply_rope(x, positions, self.cos_cache, self.sin_cache)


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        if self.rope is not None:
            if positions is None:
                positions = torch.arange(T, device=x.device)
            # Broadcast positions across batch + heads.
            q = apply_rope(q, positions, self.rope.cos_cache, self.rope.sin_cache)
            k = apply_rope(k, positions, self.rope.cos_cache, self.rope.sin_cache)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask=mask)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.rope = RoPE(d_model // num_heads, rope_theta, max_seq_len)
        self.attn = CausalMultiHeadAttention(d_model, num_heads, rope=self.rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), positions=positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        tied_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tied_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        positions = torch.arange(in_indices.shape[-1], device=in_indices.device)
        for layer in self.layers:
            x = layer(x, positions=positions)
        x = self.ln_final(x)
        return self.lm_head(x)


def param_count(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total
