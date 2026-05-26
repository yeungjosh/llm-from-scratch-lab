"""Phase 3 tests — loss, optimizer, schedule, clip, checkpoint, smoke train."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lfslab.model import TransformerLM
from lfslab.train import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def test_cross_entropy_matches_torch():
    logits = torch.randn(8, 100)
    targets = torch.randint(0, 100, (8,))
    ours = cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(ours, expected, atol=1e-5)


def test_cross_entropy_handles_3d_inputs():
    logits = torch.randn(4, 16, 50)
    targets = torch.randint(0, 50, (4, 16))
    out = cross_entropy(logits, targets)
    assert out.dim() == 0


def test_get_batch_shapes_and_shift():
    ds = np.arange(1000, dtype=np.int64)
    x, y = get_batch(ds, batch_size=4, context_length=16, device="cpu")
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    # Labels are inputs shifted by one within the same window.
    assert torch.all(y[:, :-1] == x[:, 1:])


def test_lr_schedule_endpoints():
    # Warmup convention: lr(it=0) = 0, lr(it=warmup_iters) = max_lr.
    assert get_lr_cosine_schedule(0, 1.0, 0.1, 10, 100) == 0.0
    assert get_lr_cosine_schedule(10, 1.0, 0.1, 10, 100) == 1.0
    assert get_lr_cosine_schedule(99, 1.0, 0.1, 10, 100) >= 0.1
    assert get_lr_cosine_schedule(300, 1.0, 0.1, 10, 100) == 0.1


def test_lr_schedule_warmup_is_linear():
    vals = [get_lr_cosine_schedule(i, 1.0, 0.0, 10, 100) for i in range(11)]
    assert vals[0] == 0.0
    assert vals[10] == 1.0
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    assert max(diffs) - min(diffs) < 1e-9


def test_gradient_clipping_caps_global_norm():
    p1 = torch.nn.Parameter(torch.zeros(3))
    p2 = torch.nn.Parameter(torch.zeros(5))
    p1.grad = torch.tensor([3.0, 4.0, 0.0])  # norm 5
    p2.grad = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    gradient_clipping([p1, p2], max_l2_norm=1.0)
    total = (p1.grad.pow(2).sum() + p2.grad.pow(2).sum()).sqrt()
    assert total <= 1.0 + 1e-4


def test_gradient_clipping_noop_under_threshold():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([0.3, 0.4])  # norm 0.5
    before = p.grad.clone()
    gradient_clipping([p], max_l2_norm=1.0)
    assert torch.allclose(p.grad, before)


def test_adamw_reduces_quadratic_loss():
    torch.manual_seed(0)
    x = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
    opt = AdamW([x], lr=0.1, weight_decay=0.0)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = (x**2).sum()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.05


def test_adamw_weight_decay_shrinks_param():
    torch.manual_seed(0)
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = AdamW([x], lr=0.1, weight_decay=0.5)
    for _ in range(20):
        opt.zero_grad()
        loss = (x**2).sum()
        loss.backward()
        opt.step()
    assert abs(float(x)) < 0.5


def test_checkpoint_save_load_round_trip(tmp_path: Path):
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=50, context_length=8, d_model=16,
        num_layers=1, num_heads=4, d_ff=32,
    )
    opt = AdamW(model.parameters(), lr=1e-3)
    # Take one optimizer step so AdamW state is populated.
    x = torch.randint(0, 50, (2, 8))
    logits = model(x)
    loss = cross_entropy(logits, x)
    loss.backward()
    opt.step()

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, iteration=42, out=ckpt)

    model2 = TransformerLM(
        vocab_size=50, context_length=8, d_model=16,
        num_layers=1, num_heads=4, d_ff=32,
    )
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    it = load_checkpoint(ckpt, model2, opt2)
    assert it == 42
    # Forward should match exactly.
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_smoke_training_loss_decreases():
    """50-step train on synthetic copying data; final loss < 0.5 * step-0."""
    torch.manual_seed(0)
    np.random.seed(0)
    # Synthetic corpus with a strong bigram pattern: token i is followed by (i+1) % V.
    V = 64
    ctx = 16
    base = np.tile(np.arange(V, dtype=np.int64), 200)  # length 12800

    model = TransformerLM(
        vocab_size=V, context_length=ctx, d_model=32,
        num_layers=2, num_heads=4, d_ff=64,
    )
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    losses = []
    for _step in range(50):
        x, y = get_batch(base, batch_size=8, context_length=ctx, device="cpu")
        logits = model(x)
        loss = cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))

    assert losses[-1] < losses[0] * 0.5, f"loss did not drop enough: {losses[0]} -> {losses[-1]}"


def test_adapter_wiring():
    from tests.adapters import (
        get_adamw_cls,
        run_cross_entropy,
        run_get_batch,
        run_get_lr_cosine_schedule,
        run_gradient_clipping,
    )

    assert get_adamw_cls() is AdamW
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    assert run_cross_entropy(logits, targets).dim() == 0

    ds = np.arange(200, dtype=np.int64)
    x, y = run_get_batch(ds, 2, 8, "cpu")
    assert x.shape == (2, 8) and y.shape == (2, 8)

    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([3.0, 4.0, 0.0])
    run_gradient_clipping([p], 1.0)
    assert p.grad.norm().item() <= 1.0 + 1e-4

    assert run_get_lr_cosine_schedule(5, 1.0, 0.1, 10, 100) > 0
