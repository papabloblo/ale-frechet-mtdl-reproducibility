#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VARIANT_DISPLAY = {
    "full": r"Full ALE--Fr\'{e}chet",
    "no_similarity": r"No similarity regularization ($\lambda=0$)",
    "random_graph": "Random task-neighbor graph",
}

VARIANT_ORDER = {
    "full": 0,
    "no_similarity": 1,
    "random_graph": 2,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build the component-ablation table for ALE--Frechet variants."
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/component_ablation"),
        help="Root containing <dataset>/results_mean_std.csv.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/results/component_ablation"),
        help="Directory for ablation summary CSV outputs.",
    )
    ap.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("paper/tables"),
        help="Directory for LaTeX table output.",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["multisine", "polynomial"],
        help="Datasets to include.",
    )
    ap.add_argument(
        "--selection-metric",
        default="test_rmse_mean",
        help="Metric used if multiple rows exist for the same dataset and variant.",
    )
    ap.add_argument("--decimals", type=int, default=4)
    return ap.parse_args()


def variant_from_row(row: pd.Series) -> str:
    for col in [
        "sweep__component_ablation__variant",
        "component_ablation_variant",
        "component_ablation.variant",
    ]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    combo_tag = str(row.get("combo_tag", ""))
    for variant in VARIANT_ORDER:
        if f"component_ablation-variant_{variant}" in combo_tag:
            return variant
    return "full"


def read_dataset_summary(dataset: str, results_root: Path, selection_metric: str) -> pd.DataFrame:
    path = results_root / dataset / "results_mean_std.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing component ablation results: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Component ablation results are empty: {path}")

    df = df[df["method"].astype(str).str.lower().isin(["ale_frechet", "ale", "ours"])].copy()
    if df.empty:
        raise ValueError(f"No ALE--Frechet rows found in {path}")

    df["dataset"] = dataset
    df["variant"] = df.apply(variant_from_row, axis=1)
    df["variant_order"] = df["variant"].map(VARIANT_ORDER).fillna(99).astype(int)

    rows = []
    for variant, sub in df.groupby("variant", sort=False):
        sub = sub.copy()
        if selection_metric in sub.columns:
            sub = sub.sort_values(selection_metric, ascending=True, kind="mergesort")
        elif "combo_index" in sub.columns:
            sub = sub.sort_values("combo_index", ascending=True, kind="mergesort")
        rows.append(sub.iloc[0])

    return pd.DataFrame(rows)


def format_mean_std(row: pd.Series, metric: str, decimals: int) -> str:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in row or pd.isna(row[mean_col]):
        return "--"
    if std_col not in row or pd.isna(row[std_col]):
        return f"{float(row[mean_col]):.{decimals}f}"
    return f"{float(row[mean_col]):.{decimals}f} $\\pm$ {float(row[std_col]):.{decimals}f}"


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def build_summary(results_root: Path, datasets: list[str], selection_metric: str) -> pd.DataFrame:
    blocks = [
        read_dataset_summary(dataset, results_root, selection_metric)
        for dataset in datasets
    ]
    out = pd.concat(blocks, ignore_index=True)

    full_rmse = {
        row["dataset"]: safe_float(row.get("test_rmse_mean"))
        for _, row in out[out["variant"].eq("full")].iterrows()
    }

    deltas = []
    for _, row in out.iterrows():
        base = full_rmse.get(row["dataset"], np.nan)
        value = safe_float(row.get("test_rmse_mean"))
        if np.isfinite(base) and base != 0 and np.isfinite(value):
            deltas.append(100.0 * (value - base) / abs(base))
        else:
            deltas.append(np.nan)
    out["delta_test_rmse_pct_vs_full"] = deltas
    out = out.sort_values(["dataset", "variant_order"], kind="mergesort").reset_index(drop=True)
    return out


def build_latex(summary: pd.DataFrame, decimals: int) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Component ablation of ALE--Fr\'{e}chet. The variants test "
            r"whether performance depends on learned ALE-based task similarity rather "
            r"than only on generic regularized soft sharing.}"
        ),
        r"\label{tab:component_ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Variant & Test RMSE & Test MAE & $\Delta$RMSE vs. full & Total time \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        delta = row.get("delta_test_rmse_pct_vs_full")
        delta_cell = "--" if pd.isna(delta) else f"{float(delta):+.2f}\\%"
        cells = [
            str(row["dataset"]),
            VARIANT_DISPLAY.get(str(row["variant"]), str(row["variant"]).replace("_", " ")),
            format_mean_std(row, "test_rmse", decimals),
            format_mean_std(row, "test_mae", decimals),
            delta_cell,
            format_mean_std(row, "total_time", decimals),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        results_root=args.results_root,
        datasets=args.datasets,
        selection_metric=args.selection_metric,
    )
    summary_path = args.out_dir / "component_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    table_path = args.tables_dir / "component_ablation_table.tex"
    table_path.write_text(build_latex(summary, args.decimals), encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
