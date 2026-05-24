"""Byte-level BPE tokenizer. Phase 1."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        raise NotImplementedError("Phase 1")

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("Phase 1")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError("Phase 1")

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("Phase 1")

    def save(self, path: str | os.PathLike) -> None:
        raise NotImplementedError("Phase 1")

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Tokenizer":
        raise NotImplementedError("Phase 1")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer."""
    raise NotImplementedError("Phase 1")
