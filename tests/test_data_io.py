"""Round-trip tests for token-stream save/load helpers."""

import numpy as np

from lfslab.data import load_token_stream, save_token_stream


def test_roundtrip_small_ids_use_uint16(tmp_path):
    path = tmp_path / "sub" / "ids.npy"  # parent dir is created on demand
    save_token_stream(path, [0, 1, 65535])
    out = load_token_stream(path)
    assert out.dtype == np.uint16
    assert out.tolist() == [0, 1, 65535]


def test_large_ids_use_uint32(tmp_path):
    path = tmp_path / "ids.npy"
    save_token_stream(path, np.array([3, 70000]))
    out = load_token_stream(path)
    assert out.dtype == np.uint32
    assert out.tolist() == [3, 70000]


def test_empty_stream_does_not_raise(tmp_path):
    path = tmp_path / "empty.npy"
    save_token_stream(path, [])
    assert load_token_stream(path).shape == (0,)
