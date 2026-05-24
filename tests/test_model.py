"""Model tests — shapes, causal mask, determinism, param count, adapter wiring."""

from __future__ import annotations

import math

import torch

from lfslab.model import (
    RMSNorm,
    RoPE,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    apply_rope,
    param_count,
    rmsnorm_functional,
    scaled_dot_product_attention,
    silu,
    softmax,
    swiglu_functional,
)


def test_silu_matches_torch():
    x = torch.randn(8, 16)
    expected = torch.nn.functional.silu(x)
    assert torch.allclose(silu(x), expected, atol=1e-6)


def test_rmsnorm_unit_variance_input():
    d = 32
    x = torch.randn(4, d) * 2.0
    w = torch.ones(d)
    out = rmsnorm_functional(x, w)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_softmax_sums_to_one():
    x = torch.randn(3, 5)
    s = softmax(x, dim=-1)
    assert torch.allclose(s.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert (s >= 0).all()


def test_swiglu_shapes():
    B, T, D, FF = 2, 4, 16, 64
    x = torch.randn(B, T, D)
    w1 = torch.randn(FF, D) * 0.1
    w2 = torch.randn(D, FF) * 0.1
    w3 = torch.randn(FF, D) * 0.1
    out = swiglu_functional(x, w1, w2, w3)
    assert out.shape == (B, T, D)


def test_rope_preserves_norm():
    d_k, T = 32, 8
    from lfslab.model import _rope_cache

    cos, sin = _rope_cache(d_k, 10000.0, T, device=torch.device("cpu"))
    x = torch.randn(2, T, d_k)
    positions = torch.arange(T)
    rotated = apply_rope(x, positions, cos, sin)
    assert torch.allclose(rotated.pow(2).sum(-1), x.pow(2).sum(-1), atol=1e-5)


def test_rope_position_zero_is_identity():
    d_k = 16
    from lfslab.model import _rope_cache

    cos, sin = _rope_cache(d_k, 10000.0, 4, device=torch.device("cpu"))
    x = torch.randn(2, 4, d_k)
    positions = torch.zeros(4, dtype=torch.long)
    out = apply_rope(x, positions, cos, sin)
    assert torch.allclose(out, x, atol=1e-6)


def test_scaled_dot_product_attention_causal_mask_blocks_future():
    B, H, T, D = 1, 2, 5, 8
    torch.manual_seed(0)
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    out_causal = scaled_dot_product_attention(Q, K, V, mask=mask)
    V_perturbed = V.clone()
    V_perturbed[..., 4, :] += 100.0
    out_perturbed = scaled_dot_product_attention(Q, K, V_perturbed, mask=mask)
    assert torch.allclose(out_causal[..., :4, :], out_perturbed[..., :4, :], atol=1e-5)
    assert not torch.allclose(out_causal[..., 4, :], out_perturbed[..., 4, :])


def test_transformer_block_forward_shape_and_determinism():
    torch.manual_seed(42)
    block = TransformerBlock(
        d_model=32, num_heads=4, d_ff=64, max_seq_len=16, rope_theta=10000.0
    )
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        out1 = block(x)
        out2 = block(x)
    assert out1.shape == x.shape
    assert torch.allclose(out1, out2, atol=1e-6)


def test_transformer_lm_forward_shape():
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=100, context_length=16, d_model=32,
        num_layers=2, num_heads=4, d_ff=64,
    )
    ids = torch.randint(0, 100, (3, 8))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (3, 8, 100)


def test_param_count_manual_match():
    d_model, n_layers, n_heads, d_ff, vocab = 32, 2, 4, 64, 100
    model = TransformerLM(
        vocab, 16, d_model, n_layers, n_heads, d_ff, tied_embeddings=False
    )
    per_layer = 4 * d_model * d_model + 3 * d_ff * d_model + 2 * d_model
    expected = n_layers * per_layer + d_model + 2 * vocab * d_model
    assert param_count(model) == expected


def test_tied_embeddings_shares_weight():
    model = TransformerLM(100, 16, 32, 1, 4, 64, tied_embeddings=True)
    assert model.token_embeddings.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_rmsnorm_module_matches_functional():
    d = 16
    x = torch.randn(2, 4, d)
    ln = RMSNorm(d)
    ln.weight.data = torch.linspace(0.5, 1.5, d)
    assert torch.allclose(ln(x), rmsnorm_functional(x, ln.weight, ln.eps), atol=1e-6)


def test_swiglu_module_matches_functional():
    d, FF = 8, 32
    x = torch.randn(2, 4, d)
    mod = SwiGLU(d, FF)
    expected = swiglu_functional(x, mod.w1.weight, mod.w2.weight, mod.w3.weight)
    assert torch.allclose(mod(x), expected, atol=1e-6)


def test_rope_module_matches_functional():
    d_k, T = 16, 8
    x = torch.randn(2, T, d_k)
    positions = torch.arange(T)
    mod = RoPE(d_k, 10000.0, max_seq_len=T)
    from lfslab.model import _rope_cache

    cos, sin = _rope_cache(d_k, 10000.0, T, device=torch.device("cpu"))
    assert torch.allclose(mod(x, positions), apply_rope(x, positions, cos, sin), atol=1e-6)


def test_adapter_transformer_lm_matches_module():
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=50, context_length=16, d_model=16,
        num_layers=2, num_heads=4, d_ff=32,
    )
    ids = torch.randint(0, 50, (1, 6))
    with torch.no_grad():
        direct = model(ids)
    weights = dict(model.state_dict())

    from tests.adapters import run_transformer_lm

    adapter_out = run_transformer_lm(
        vocab_size=50,
        context_length=16,
        d_model=16,
        num_layers=2,
        num_heads=4,
        d_ff=32,
        rope_theta=10000.0,
        weights=weights,
        in_indices=ids,
    )
    assert torch.allclose(direct, adapter_out, atol=1e-5)


def test_adapter_rmsnorm():
    from tests.adapters import run_rmsnorm

    d = 16
    x = torch.randn(2, 4, d)
    w = torch.linspace(0.5, 1.5, d)
    out = run_rmsnorm(d, 1e-5, w, x)
    assert out.shape == x.shape


def test_adapter_swiglu():
    from tests.adapters import run_swiglu

    d, FF = 8, 32
    x = torch.randn(2, 4, d)
    w1 = torch.randn(FF, d) * 0.1
    w2 = torch.randn(d, FF) * 0.1
    w3 = torch.randn(FF, d) * 0.1
    out = run_swiglu(d, FF, w1, w2, w3, x)
    assert out.shape == x.shape


def test_adapter_rope():
    from tests.adapters import run_rope

    d_k, T = 16, 8
    x = torch.randn(2, T, d_k)
    positions = torch.arange(T)
    out = run_rope(d_k, 10000.0, T, x, positions)
    assert torch.allclose(out.pow(2).sum(-1), x.pow(2).sum(-1), atol=1e-4)


def test_attention_logit_scaling_uses_sqrt_dk():
    B, H, T, D = 1, 1, 3, 16
    Q = torch.ones(B, H, T, D)
    K = torch.ones(B, H, T, D)
    V = torch.eye(T).unsqueeze(0).unsqueeze(0)
    out = scaled_dot_product_attention(Q, K, V)
    expected = V.mean(dim=-2, keepdim=True).expand_as(V)
    assert torch.allclose(out, expected, atol=1e-5)
    _ = math.sqrt(D)
