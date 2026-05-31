#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Restore bundled published aggregate results into the "
            "results/comparisons/<dataset>/results_mean_std.csv layout expected "
            "by scripts/build_paper_results.py."
        )
    )
    parser.add_argument(
        "--published-dir",
        type=Path,
        default=Path("results/published"),
        help="Directory containing paper_results_long.csv from the submitted package.",
    )
    parser.add_argument(
        "--comparisons-dir",
        type=Path,
        default=Path("results/comparisons"),
        help="Output comparisons directory to populate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-dataset results_mean_std.csv files.",
    )
    args = parser.parse_args()

    source = args.published_dir / "paper_results_long.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing published result file: {source}")

    df = pd.read_csv(source)
    if "dataset" not in df.columns:
        raise ValueError(f"{source} must contain a 'dataset' column.")
    if df.empty:
        raise ValueError(f"{source} is empty.")

    args.comparisons_dir.mkdir(parents=True, exist_ok=True)

    restored = 0
    skipped = 0
    for dataset, group in df.groupby("dataset", sort=True):
        dataset_name = str(dataset).strip()
        if not dataset_name:
            continue

        out_dir = args.comparisons_dir / dataset_name
        out_file = out_dir / "results_mean_std.csv"
        if out_file.exists() and not args.overwrite:
            print(f"[SKIP] {out_file} exists; pass --overwrite to replace it.")
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        out = group.drop(columns=["dataset"]).copy()
        out.to_csv(out_file, index=False)
        print(f"[OK] restored {out_file} ({len(out)} rows)")
        restored += 1

    print(f"Restored {restored} dataset file(s); skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
