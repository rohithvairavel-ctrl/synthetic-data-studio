#!/usr/bin/env python3
"""Download Adult/Census Income and write a tiny committed sample."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import download_adult, write_sample


def main() -> None:
    path = download_adult(force=False)
    sample = write_sample(n=200, seed=42)
    print(f"Full dataset: {path} ({path.stat().st_size} bytes)")
    print(f"Sample:       {sample} ({sample.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
