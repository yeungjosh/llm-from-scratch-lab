"""Download TinyStories v2 GPT4 train/valid via HuggingFace.

Usage:
    uv run python scripts/download_data.py --out data/

Skips files that already exist.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

FILES = {
    "TinyStoriesV2-GPT4-train.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "TinyStoriesV2-GPT4-valid.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already present ({dest.stat().st_size:,} bytes)")
        return
    print(f"[fetch] {url}  ->  {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"[done]  {dest.name}  ({dest.stat().st_size:,} bytes)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    args = p.parse_args()
    out = Path(args.out)
    for name, url in FILES.items():
        _download(url, out / name)


if __name__ == "__main__":
    main()
