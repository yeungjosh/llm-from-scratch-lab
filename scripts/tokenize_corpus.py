"""Train a BPE tokenizer on a corpus, then tokenize train/valid splits to .npy.

Usage:
    uv run python scripts/tokenize_corpus.py \\
        --train data/TinyStoriesV2-GPT4-train.txt \\
        --valid data/TinyStoriesV2-GPT4-valid.txt \\
        --vocab-size 10000 \\
        --tokenizer-out data/tokenizer/ \\
        --train-out data/processed/tinystories_train.npy \\
        --valid-out data/processed/tinystories_valid.npy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lfslab.tokenizer import Tokenizer, train_bpe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--special-tokens", nargs="+", default=["<|endoftext|>"])
    p.add_argument("--tokenizer-out", default="data/tokenizer")
    p.add_argument("--train-out", default="data/processed/tinystories_train.npy")
    p.add_argument("--valid-out", default="data/processed/tinystories_valid.npy")
    p.add_argument("--train-sample-bytes", type=int, default=None,
                   help="Optional cap for BPE training corpus size (bytes).")
    args = p.parse_args()

    train_path = Path(args.train)
    if args.train_sample_bytes:
        sample_path = train_path.with_suffix(".sample.txt")
        if not sample_path.exists():
            print(f"[sample] capping training corpus at {args.train_sample_bytes:,} bytes")
            with open(train_path, "rb") as fin, open(sample_path, "wb") as fout:
                fout.write(fin.read(args.train_sample_bytes))
        bpe_input = sample_path
    else:
        bpe_input = train_path

    t0 = time.time()
    print(f"[bpe] training on {bpe_input} -> vocab_size={args.vocab_size}")
    vocab, merges = train_bpe(bpe_input, args.vocab_size, args.special_tokens)
    print(f"[bpe] done in {time.time() - t0:.1f}s  ({len(vocab)} tokens, {len(merges)} merges)")

    tok = Tokenizer(vocab, merges, args.special_tokens)
    tok.save(args.tokenizer_out)
    print(f"[bpe] tokenizer saved -> {args.tokenizer_out}")

    for in_path, out_path in [(args.train, args.train_out), (args.valid, args.valid_out)]:
        t0 = time.time()
        text = Path(in_path).read_text(encoding="utf-8")
        ids = tok.encode(text)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(ids, dtype=np.uint16 if max(ids) < 2**16 else np.uint32)
        np.save(out_path, arr)
        print(
            f"[tok] {in_path}: {len(text):,} chars -> {len(ids):,} tokens "
            f"({time.time() - t0:.1f}s) -> {out_path}"
        )


if __name__ == "__main__":
    main()
