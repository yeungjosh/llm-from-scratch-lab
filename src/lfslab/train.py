"""Training loop, optimizer, schedule, gradient clip, checkpointing."""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import yaml

from .data import load_token_stream
from .hardware import pick_device
from .model import TransformerLM, softmax

# --- losses ------------------------------------------------------------------


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy. Numerically stable: log-sum-exp on logits.

    logits: (..., vocab); targets: (...,) int.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1).to(torch.long)
    log_z = torch.logsumexp(flat_logits, dim=-1)
    gathered = flat_logits.gather(-1, flat_targets.unsqueeze(-1)).squeeze(-1)
    return (log_z - gathered).mean()


# --- data --------------------------------------------------------------------


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (inputs, labels). Labels are inputs shifted by one."""
    n = len(dataset)
    assert n > context_length, "dataset shorter than context_length"
    starts = np.random.randint(0, n - context_length, size=batch_size)
    x = np.stack([np.asarray(dataset[s : s + context_length]) for s in starts])
    y = np.stack([np.asarray(dataset[s + 1 : s + 1 + context_length]) for s in starts])
    return (
        torch.as_tensor(x, dtype=torch.long, device=device),
        torch.as_tensor(y, dtype=torch.long, device=device),
    )


# --- optimizer ---------------------------------------------------------------


class AdamW(torch.optim.Optimizer):
    """AdamW with decoupled weight decay (Loshchilov & Hutter)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid lr {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ARG002
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]
                g = p.grad
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)


# --- schedule + clip ---------------------------------------------------------


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / max(1, cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total = torch.sqrt(sum((g.detach().pow(2).sum() for g in grads), torch.tensor(0.0)))
    eps = 1e-6
    if total > max_l2_norm:
        scale = max_l2_norm / (total + eps)
        for g in grads:
            g.mul_(scale)


# --- checkpoint --------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    state = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["iteration"])


# --- training driver ---------------------------------------------------------


def build_model_from_config(cfg: dict) -> TransformerLM:
    mcfg = cfg["model"]
    return TransformerLM(
        vocab_size=mcfg["vocab_size"],
        context_length=mcfg["context_length"],
        d_model=mcfg["d_model"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        d_ff=mcfg["d_ff"],
        rope_theta=mcfg.get("rope_theta", 10000.0),
        tied_embeddings=mcfg.get("tied_embeddings", False),
    )


def train(config_path: str, max_steps: int | None = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = cfg["training"].get("device") or pick_device()
    model = build_model_from_config(cfg).to(device)
    opt = AdamW(
        model.parameters(),
        lr=cfg["training"]["max_lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )

    train_ds = load_token_stream(cfg["data"]["train_tokens"])
    valid_ds = load_token_stream(cfg["data"]["valid_tokens"])

    ctx = cfg["model"]["context_length"]
    bs = cfg["training"]["batch_size"]
    steps = max_steps if max_steps is not None else cfg["training"]["max_steps"]
    max_lr = cfg["training"]["max_lr"]
    min_lr = cfg["training"]["min_lr"]
    warmup = cfg["training"].get("warmup_iters", 0)
    cycle = cfg["training"].get("cosine_cycle_iters", steps)
    clip = cfg["training"].get("grad_clip", 1.0)
    log_every = cfg["training"].get("log_every", 25)

    losses: list[float] = []
    t0 = time.time()
    for step in range(steps):
        lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup, cycle)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_ds, bs, ctx, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), clip)
        opt.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0:
            print(
                f"[step {step:>5}]  loss={losses[-1]:.4f}  lr={lr:.2e}  "
                f"elapsed={time.time() - t0:.1f}s"
            )

    val_ppl = validation_perplexity(model, valid_ds, ctx, bs, device, num_batches=10)
    return {"losses": losses, "val_ppl": val_ppl}


@torch.no_grad()
def validation_perplexity(
    model: nn.Module,
    dataset,
    context_length: int,
    batch_size: int,
    device: str,
    num_batches: int = 20,
) -> float:
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        total_loss += float(cross_entropy(logits, y))
    return math.exp(total_loss / num_batches)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--checkpoint-out", default=None)
    args = p.parse_args()
    result = train(args.config, max_steps=args.max_steps)
    print(f"final val PPL: {result['val_ppl']:.2f}")
    if args.checkpoint_out:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model = build_model_from_config(cfg)
        opt = AdamW(model.parameters(), lr=cfg["training"]["max_lr"])
        Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, opt, args.max_steps or 0, args.checkpoint_out)
        print(f"checkpoint saved -> {args.checkpoint_out}")


# Keep a tiny shim so other modules can import softmax via train.
__all__ = [
    "AdamW",
    "cross_entropy",
    "get_batch",
    "get_lr_cosine_schedule",
    "gradient_clipping",
    "load_checkpoint",
    "save_checkpoint",
    "softmax",
    "train",
    "validation_perplexity",
]


if __name__ == "__main__":
    main()
