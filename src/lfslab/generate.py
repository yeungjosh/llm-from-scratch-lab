"""Greedy / temperature / top-k sampling."""

from __future__ import annotations

import argparse

import torch

from .model import softmax


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str | None = None,
    eos_id: int | None = None,
) -> str:
    device = device or next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_in = x[:, -model.context_length :] if x.shape[1] > model.context_length else x
        logits = model(x_in)[:, -1, :]
        if temperature <= 0:
            next_id = int(logits.argmax(dim=-1).item())
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                v, _ = torch.topk(logits, k)
                logits = torch.where(
                    logits < v[..., [-1]],
                    torch.full_like(logits, float("-inf")),
                    logits,
                )
            probs = softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        if eos_id is not None and next_id == eos_id:
            break
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(x[0].tolist())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    args = p.parse_args()
    raise NotImplementedError(
        f"CLI: load checkpoint {args.checkpoint} and tokenizer {args.tokenizer}, "
        f"then call generate() — wire in Phase 9 if needed."
    )


if __name__ == "__main__":
    main()
