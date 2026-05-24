"""Download TinyStories v2 + OWT sample via HuggingFace. Phase 1."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Phase 1: pull TinyStoriesV2-GPT4-train.txt + owt_train.txt.gz into "
        f"{args.out}/"
    )


if __name__ == "__main__":
    main()
