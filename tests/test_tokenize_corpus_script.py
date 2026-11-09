"""End-to-end test for scripts/tokenize_corpus.py, including an empty split."""

import subprocess
import sys
from pathlib import Path

from lfslab.data import load_token_stream

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tokenize_corpus.py"


def test_script_tokenizes_and_tolerates_empty_valid_split(tmp_path):
    train = tmp_path / "train.txt"
    valid = tmp_path / "valid.txt"
    train.write_text("hello world hello<|endoftext|>foo bar baz\n", encoding="utf-8")
    valid.write_text("", encoding="utf-8")
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--train", str(train), "--valid", str(valid),
            "--vocab-size", "270",
            "--tokenizer-out", str(tmp_path / "tok"),
            "--train-out", str(out / "train.npy"),
            "--valid-out", str(out / "valid.npy"),
        ],
        check=True, capture_output=True,
    )
    assert load_token_stream(out / "train.npy").size > 0
    assert load_token_stream(out / "valid.npy").size == 0
