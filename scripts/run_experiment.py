#!/usr/bin/env python3
"""Run the full utility/privacy tradeoff experiment and write reports + figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import run_tradeoff_sweep
from src.plotting import make_all_figures


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic Data Studio experiment")
    p.add_argument("--max-rows", type=int, default=2000, help="Subsample size for speed")
    p.add_argument("--ctgan", action="store_true", help="Include CTGAN (slower)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    result = run_tradeoff_sweep(
        max_rows=args.max_rows,
        seed=args.seed,
        include_ctgan=args.ctgan,
    )
    paths = make_all_figures(
        result["metrics_df"],
        result["real_train"],
        result["synth"],
        result["dcr_distances"],
    )
    print("\n=== Summary ===")
    print(result["metrics_df"].to_string(index=False))
    print("\nFigures:")
    for path in paths:
        print(f"  {path}")
    print(f"\nReports: {ROOT / 'reports'}")


if __name__ == "__main__":
    main()
