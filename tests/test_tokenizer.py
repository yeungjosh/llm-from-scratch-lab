"""Tokenizer tests — round-trip, training, special tokens, save/load."""

from __future__ import annotations

from pathlib import Path

import pytest

from lfslab.tokenizer import Tokenizer, train_bpe

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"
SPECIAL = ["<|endoftext|>"]


@pytest.fixture(scope="module")
def trained() -> Tokenizer:
    vocab, merges = train_bpe(FIXTURE, vocab_size=500, special_tokens=SPECIAL)
    return Tokenizer(vocab, merges, SPECIAL)


def test_train_bpe_returns_expected_shapes():
    vocab, merges = train_bpe(FIXTURE, vocab_size=400, special_tokens=SPECIAL)
    assert len(vocab) <= 400
    assert len(vocab) >= 256 + len(SPECIAL)
    # First 256 are byte values.
    assert vocab[0] == b"\x00"
    assert vocab[255] == b"\xff"
    # Special token is in the vocab.
    assert b"<|endoftext|>" in set(vocab.values())
    # Merges are bytes pairs.
    for a, b in merges:
        assert isinstance(a, bytes) and isinstance(b, bytes)
        assert len(a) >= 1 and len(b) >= 1


def test_no_empty_merges():
    _, merges = train_bpe(FIXTURE, vocab_size=400, special_tokens=SPECIAL)
    for a, b in merges:
        assert a and b  # non-empty


def test_round_trip_ascii(trained: Tokenizer):
    s = "The little cat sat on the mat."
    assert trained.decode(trained.encode(s)) == s


def test_round_trip_with_special_token(trained: Tokenizer):
    s = "Hello world<|endoftext|>Goodbye world"
    ids = trained.encode(s)
    assert trained.decode(ids) == s
    # Special token compresses to a single id.
    eot_id = next(i for i, b in trained.vocab.items() if b == b"<|endoftext|>")
    assert eot_id in ids


def test_round_trip_unicode(trained: Tokenizer):
    s = "café — naïve résumé 🐱"
    out = trained.decode(trained.encode(s))
    assert out == s


def test_encode_iterable_decodes_back_to_joined_text(trained: Tokenizer):
    # encode_iterable processes chunks independently — pretoken boundaries can
    # differ from full-text encode. The invariant we require: decode(stream) ==
    # concat(parts), so the tokenizer is still lossless at the document level.
    parts = ["Hello ", "world", "!"]
    streamed = list(trained.encode_iterable(parts))
    assert trained.decode(streamed) == "".join(parts)


def test_save_load_round_trip(trained: Tokenizer, tmp_path: Path):
    out = tmp_path / "tok"
    trained.save(out)
    reloaded = Tokenizer.load(out)
    s = "Test save/load round trip."
    assert reloaded.encode(s) == trained.encode(s)
    assert reloaded.decode(trained.encode(s)) == s


def test_from_files_classmethod(trained: Tokenizer, tmp_path: Path):
    out = tmp_path / "tok2"
    trained.save(out)
    reloaded = Tokenizer.from_files(
        out / "vocab.json", out / "merges.txt", special_tokens=SPECIAL
    )
    s = "Friends play together."
    assert reloaded.encode(s) == trained.encode(s)


def test_adapter_wiring():
    from tests.adapters import get_tokenizer, run_train_bpe

    vocab, merges = run_train_bpe(FIXTURE, vocab_size=400, special_tokens=SPECIAL)
    tok = get_tokenizer(vocab, merges, SPECIAL)
    s = "Adapter round trip."
    assert tok.decode(tok.encode(s)) == s
