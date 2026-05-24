"""Decoder-only Transformer. Phase 2.

RMSNorm + SwiGLU + RoPE + causal multi-head attention, pre-norm residual.
Anchors against CS336 A1 adapters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 2")


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 2")


class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int) -> None:
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 2")


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

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("Phase 2")


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 2")


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

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 2")


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
