"""Byte-level BPE tokenizer (GPT-2 pretokenization, CS336-faithful).

Vocab is dict[int, bytes]; merges is list[tuple[bytes, bytes]] in creation order.
Training tie-break: pair with highest count, lexicographically-greatest pair wins ties.
"""

from __future__ import annotations

import base64
import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

# GPT-2 pretokenization pattern (handles contractions, words, numbers, punctuation, whitespace).
PAT = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
_PAT_RE = re.compile(PAT)


def _pretokenize_to_bytes(text: str) -> list[bytes]:
    return [m.group(0).encode("utf-8") for m in _PAT_RE.finditer(text)]


def _split_on_special(text: str, special_tokens: list[str]) -> Iterator[tuple[str, bool]]:
    """Yield (chunk, is_special). Longest specials matched first to handle overlap."""
    if not special_tokens:
        yield text, False
        return
    specials = sorted(special_tokens, key=len, reverse=True)
    pat = "|".join(re.escape(s) for s in specials)
    last = 0
    for m in re.finditer(pat, text):
        if m.start() > last:
            yield text[last : m.start()], False
        yield m.group(0), True
        last = m.end()
    if last < len(text):
        yield text[last:], False


def _apply_merge(
    tokens: tuple[bytes, ...], pair: tuple[bytes, bytes], merged: bytes
) -> tuple[bytes, ...]:
    out: list[bytes] = []
    i = 0
    n = len(tokens)
    while i < n:
        if i < n - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return tuple(out)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns (vocab, merges).
    vocab: token_id -> bytes. Initial 256 ids are the byte values; then special tokens;
    then merged pairs in the order they were created.
    """
    text = Path(input_path).read_text(encoding="utf-8")

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    # Pretoken word counts (split on special tokens first so we never merge across them).
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk, is_special in _split_on_special(text, special_tokens):
        if is_special:
            continue
        for pt in _pretokenize_to_bytes(chunk):
            word_counts[tuple(bytes([b]) for b in pt)] += 1

    merges: list[tuple[bytes, bytes]] = []
    target_merges = vocab_size - len(vocab)

    while len(merges) < target_merges:
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for word, c in word_counts.items():
            for a, b in zip(word, word[1:], strict=False):
                pair_counts[(a, b)] += c
        if not pair_counts:
            break
        # Tie-break: highest count; if equal, lexicographically greater pair.
        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merged = best_pair[0] + best_pair[1]
        vocab[next_id] = merged
        next_id += 1
        merges.append(best_pair)

        new_counts: Counter[tuple[bytes, ...]] = Counter()
        for word, c in word_counts.items():
            new_counts[_apply_merge(word, best_pair, merged)] += c
        word_counts = new_counts

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens or [])
        # Reverse maps for fast lookup.
        self._bytes_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}
        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        # Special-token bytes → id (must already exist in vocab).
        for st in self.special_tokens:
            stb = st.encode("utf-8")
            if stb not in self._bytes_to_id:
                # Add it to the vocab on the fly (defensive — adapter-tested path).
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = stb
                self._bytes_to_id[stb] = new_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab = _read_vocab(vocab_filepath)
        merges = _read_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def _bpe_encode_pretoken(self, pretoken: bytes) -> list[int]:
        if not pretoken:
            return []
        tokens: list[bytes] = [bytes([b]) for b in pretoken]
        while len(tokens) > 1:
            best_rank = None
            best_i = -1
            for i in range(len(tokens) - 1):
                r = self._merge_rank.get((tokens[i], tokens[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i < 0:
                break
            merged = tokens[best_i] + tokens[best_i + 1]
            tokens = tokens[:best_i] + [merged] + tokens[best_i + 2 :]
        return [self._bytes_to_id[t] for t in tokens]

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk, is_special in _split_on_special(text, self.special_tokens):
            if is_special:
                out.append(self._bytes_to_id[chunk.encode("utf-8")])
                continue
            for pt in _pretokenize_to_bytes(chunk):
                out.extend(self._bpe_encode_pretoken(pt))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)

    def decode(self, ids: list[int]) -> str:
        buf = b"".join(self.vocab[i] for i in ids)
        return buf.decode("utf-8", errors="replace")

    def save(self, path: str | os.PathLike) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        vocab_out = {
            str(i): base64.b64encode(b).decode("ascii") for i, b in self.vocab.items()
        }
        (p / "vocab.json").write_text(json.dumps(vocab_out, indent=0))
        merges_lines = [
            f"{base64.b64encode(a).decode('ascii')} {base64.b64encode(b).decode('ascii')}"
            for a, b in self.merges
        ]
        (p / "merges.txt").write_text("\n".join(merges_lines))
        (p / "special_tokens.json").write_text(json.dumps(self.special_tokens))

    @classmethod
    def load(cls, path: str | os.PathLike) -> Tokenizer:
        p = Path(path)
        vocab = _read_vocab(p / "vocab.json")
        merges = _read_merges(p / "merges.txt")
        special = json.loads((p / "special_tokens.json").read_text())
        return cls(vocab, merges, special)


def _read_vocab(path: str | os.PathLike) -> dict[int, bytes]:
    data = json.loads(Path(path).read_text())
    return {int(k): base64.b64decode(v) for k, v in data.items()}


def _read_merges(path: str | os.PathLike) -> list[tuple[bytes, bytes]]:
    out: list[tuple[bytes, bytes]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        a, b = line.split(" ")
        out.append((base64.b64decode(a), base64.b64decode(b)))
    return out
