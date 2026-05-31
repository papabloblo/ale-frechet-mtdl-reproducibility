#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FILES = [
    "paper_results_long.csv",
    "best_configs.csv",
    "method_ranking.csv",
    "rank_statistical_tests.csv",
    "test_rmse_wide.csv",
    "test_mae_wide.csv",
    "total_time_wide.csv",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    return pd.read_csv(path)


def _sort_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        col
        for col in ["dataset", "method", "metric", "combo_index", "comparison", "rank"]
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")
    else:
        df = df.sort_values(list(df.columns), kind="mergesort")
    return df.reset_index(drop=True)


def _compare_csv(actual_path: Path, expected_path: Path, rtol: float, atol: float) -> list[str]:
    actual = _sort_for_comparison(_read_csv(actual_path))
    expected = _sort_for_comparison(_read_csv(expected_path))
    issues: list[str] = []

    if set(actual.columns) != set(expected.columns):
        return [
            "column mismatch: "
            f"actual={list(actual.columns)} expected={list(expected.columns)}"
        ]

    actual = actual[list(expected.columns)]

    if actual.shape != expected.shape:
        return [f"shape mismatch: actual={actual.shape} expected={expected.shape}"]

    for col in actual.columns:
        left = actual[col]
        right = expected[col]
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        numeric = left_num.notna() | right_num.notna()

        if numeric.any():
            equal_num = np.isclose(
                left_num[numeric].astype(float),
                right_num[numeric].astype(float),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
            if not bool(np.all(equal_num)):
                mismatch_count = int((~equal_num).sum())
                issues.append(f"{col}: {mismatch_count} numeric mismatch(es)")

        text_mask = ~numeric
        if text_mask.any():
            left_text = left[text_mask].fillna("").astype(str)
            right_text = right[text_mask].fillna("").astype(str)
            unequal = left_text.to_numpy() != right_text.to_numpy()
            if bool(np.any(unequal)):
                issues.append(f"{col}: {int(unequal.sum())} text mismatch(es)")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare regenerated paper CSV outputs against bundled published outputs."
    )
    parser.add_argument("--actual-dir", type=Path, default=Path("paper/results"))
    parser.add_argument("--published-dir", type=Path, default=Path("results/published"))
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()

    failures = 0
    for rel in args.files:
        actual_path = args.actual_dir / rel
        expected_path = args.published_dir / rel
        try:
            issues = _compare_csv(actual_path, expected_path, args.rtol, args.atol)
        except Exception as exc:
            print(f"[FAIL] {rel}: {exc}")
            failures += 1
            continue

        if issues:
            print(f"[FAIL] {rel}")
            for issue in issues:
                print(f"  - {issue}")
            failures += 1
        else:
            print(f"[OK] {rel}")

    if failures:
        print(f"\nValidation failed for {failures} file(s).")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
