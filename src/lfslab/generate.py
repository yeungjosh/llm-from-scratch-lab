"""Text generation. Phase 3."""

from __future__ import annotations

import argparse


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    raise NotImplementedError("Phase 3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    raise NotImplementedError(f"Phase 3 — wire {args}")


if __name__ == "__main__":
    main()
