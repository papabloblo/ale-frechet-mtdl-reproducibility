#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DEFAULT_DATASETS = ["polynomial", "multisine"]
UPDATE_ORDER = ["1", "3", "5", "10"]
KEEP_ORDER = ["1", "3", "5", "10"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a relative ALE-Frechet similarity-update table. Rows are "
            "similarity update frequencies, columns are keep-epoch values, and "
            "cells are relative to update_every=1 and keep_epochs=1."
        )
    )
    ap.add_argument(
        "--comparisons-root",
        type=Path,
        default=Path("results/comparisons"),
        help="Root directory containing per-dataset comparison outputs.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/results/ablation"),
        help="Directory where summary CSV files will be written.",
    )
    ap.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("paper/tables"),
        help="Directory where LaTeX tables will be written.",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to include in the comparison.",
    )
    ap.add_argument(
        "--selection-metric",
        default="test_rmse_mean",
        help="Metric used to select the best config within each update/keep pair.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Accepted for compatibility with the Makefile target. The relative "
            "table reports RMSE and total-time speedup."
        ),
    )
    ap.add_argument(
        "--decimals",
        type=int,
        default=1,
        help="Number of decimals for relative percentage values.",
    )
    return ap.parse_args()


def read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path)


def latex_escape(text: object) -> str:
    out = str(text)
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def normalize_factor(value: object) -> str:
    if pd.isna(value):
        return "none"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def format_pct(value: object, decimals: int) -> str:
    if pd.isna(value):
        return "--"
    numeric = float(value)
    sign = "+" if numeric > 0 else ""
    return f"{sign}{numeric:.{decimals}f}\\%"


def format_speedup(value: object) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.2f}$\\times$"


def select_best_per_update_keep(
    df: pd.DataFrame,
    dataset: str,
    selection_metric: str,
) -> pd.DataFrame:
    work = df.copy()
    work["method"] = work["method"].astype(str).str.lower()
    work = work[work["method"] == "ale_frechet"].copy()
    if work.empty:
        raise ValueError(f"No ale_frechet rows found for dataset '{dataset}'.")
    if selection_metric not in work.columns:
        raise KeyError(
            f"Selection metric '{selection_metric}' is missing for dataset '{dataset}'."
        )

    work["dataset"] = dataset
    work["update"] = work["sweep__similarity__update_every"].map(normalize_factor)
    work["keep"] = work["sweep__similarity__keep_epochs"].map(normalize_factor)
    work = work[
        work["update"].isin(UPDATE_ORDER)
        & work["keep"].isin(KEEP_ORDER)
    ].copy()
    if work.empty:
        raise ValueError(
            f"No ale_frechet rows remain for dataset '{dataset}' after excluding "
            "update_every=none and keep_epochs=none."
        )

    std_col = selection_metric.replace("_mean", "_std")
    sort_cols = ["dataset", "update", "keep", selection_metric]
    ascending = [True, True, True, True]
    if std_col in work.columns:
        sort_cols.append(std_col)
        ascending.append(True)
    if "combo_index" in work.columns:
        sort_cols.append("combo_index")
        ascending.append(True)

    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    best = work.groupby(["dataset", "update", "keep"], as_index=False, sort=False).head(1)
    return best.reset_index(drop=True)


def add_relative_metrics(best_df: pd.DataFrame) -> pd.DataFrame:
    out = best_df.copy()
    base = out[(out["update"] == "1") & (out["keep"] == "1")].copy()
    base = base[
        [
            "dataset",
            "test_rmse_mean",
            "test_mae_mean",
            "total_time_mean",
        ]
    ].rename(
        columns={
            "test_rmse_mean": "baseline_test_rmse_mean",
            "test_mae_mean": "baseline_test_mae_mean",
            "total_time_mean": "baseline_total_time_mean",
        }
    )

    out = out.merge(base, on=["dataset"], how="left")
    out["rel_rmse_pct"] = (
        (out["test_rmse_mean"] / out["baseline_test_rmse_mean"]) - 1.0
    ) * 100.0
    out["rel_mae_pct"] = (
        (out["test_mae_mean"] / out["baseline_test_mae_mean"]) - 1.0
    ) * 100.0
    out["relative_speedup"] = out["baseline_total_time_mean"] / out["total_time_mean"]
    return out


def rmse_cell(summary_df: pd.DataFrame, dataset: str, update: str, keep: str, decimals: int) -> str:
    row = summary_df[
        (summary_df["dataset"] == dataset)
        & (summary_df["update"] == update)
        & (summary_df["keep"] == keep)
    ]
    if row.empty:
        return "--"

    item = row.iloc[0]
    if update == "1" and keep == "1":
        return r"\textit{Baseline}"

    cell = format_pct(item["rel_rmse_pct"], decimals)
    if item["rel_rmse_pct"] <= 0:
        return r"\textbf{" + cell + "}"
    return cell


def speedup_cell(summary_df: pd.DataFrame, dataset: str, update: str, keep: str) -> str:
    row = summary_df[
        (summary_df["dataset"] == dataset)
        & (summary_df["update"] == update)
        & (summary_df["keep"] == keep)
    ]
    if row.empty:
        return "--"

    item = row.iloc[0]
    if update == "1" and keep == "1":
        return r"\textit{Baseline}"

    cell = format_speedup(item["relative_speedup"])
    if item["relative_speedup"] >= 1:
        return r"\textbf{" + cell + "}"
    return cell


def build_grid_latex_table(
    summary_df: pd.DataFrame,
    datasets: Iterable[str],
    decimals: int,
    *,
    value_kind: str,
) -> str:
    datasets = list(datasets)
    if value_kind == "rmse":
        caption = (
            r"Relative test RMSE for ALE--Fr\'echet similarity recomputation frequency. "
            r"Rows report \texttt{similarity.update\_every}; columns report "
            r"\texttt{similarity.keep\_epochs}. Values are relative to recomputing "
            r"similarity every step (\texttt{update\_every=1}, \texttt{keep\_epochs=1}). "
            r"Negative values indicate lower RMSE; the reference configuration is marked as baseline."
        )
        label = "tab:similarity_update_relative_rmse"
    elif value_kind == "speedup":
        caption = (
            r"Runtime speedup for ALE--Fr\'echet similarity recomputation frequency. "
            r"Rows report \texttt{similarity.update\_every}; columns report "
            r"\texttt{similarity.keep\_epochs}. Values are speedups relative to recomputing "
            r"similarity every step (\texttt{update\_every=1}, \texttt{keep\_epochs=1}). "
            r"Values above $1\times$ are faster; the reference configuration is marked as baseline."
        )
        label = "tab:similarity_update_speedup"
    else:
        raise ValueError(f"Unsupported table kind: {value_kind}")

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll" + "c" * len(KEEP_ORDER) + r"}")
    lines.append(r"\toprule")
    lines.append(
        "Dataset & Update & "
        + " & ".join([rf"Keep {latex_escape(k)}" for k in KEEP_ORDER])
        + r" \\"
    )
    lines.append(r"\midrule")

    for dataset_index, dataset in enumerate(datasets):
        if dataset_index > 0:
            lines.append(r"\addlinespace")
        for update_index, update in enumerate(UPDATE_ORDER):
            dataset_cell = latex_escape(dataset) if update_index == 0 else ""
            row_cells = [dataset_cell, latex_escape(update)]
            for keep in KEEP_ORDER:
                if value_kind == "rmse":
                    row_cells.append(rmse_cell(summary_df, dataset, update, keep, decimals))
                else:
                    row_cells.append(speedup_cell(summary_df, dataset, update, keep))
            lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_verdict(summary_df: pd.DataFrame) -> pd.DataFrame:
    candidates = summary_df[summary_df["update"] != "1"].copy()
    rows: List[Dict[str, object]] = []
    for dataset in candidates["dataset"].drop_duplicates():
        dataset_rows = candidates[candidates["dataset"] == dataset].copy()
        dataset_rows = dataset_rows.sort_values(
            ["test_rmse_mean", "total_time_mean", "update", "keep"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
        best = dataset_rows.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "best_update": best["update"],
                "best_keep": best["keep"],
                "best_combo_index": int(best["combo_index"]),
                "test_rmse_mean": best["test_rmse_mean"],
                "rel_rmse_pct_vs_update_1_keep_1": best["rel_rmse_pct"],
                "relative_speedup_vs_update_1_keep_1": best["relative_speedup"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    blocks = []
    for dataset in args.datasets:
        results_path = args.comparisons_root / dataset / "results_mean_std.csv"
        dataset_df = read_results(results_path)
        blocks.append(
            select_best_per_update_keep(
                dataset_df,
                dataset=dataset,
                selection_metric=args.selection_metric,
            )
        )

    best_df = pd.concat(blocks, ignore_index=True)
    summary_df = add_relative_metrics(best_df)
    summary_df["update"] = pd.Categorical(summary_df["update"], categories=UPDATE_ORDER, ordered=True)
    summary_df["keep"] = pd.Categorical(summary_df["keep"], categories=KEEP_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["dataset", "update", "keep"], kind="mergesort")
    verdict_df = build_verdict(summary_df)

    summary_df.to_csv(args.out_dir / "similarity_update_summary.csv", index=False)
    verdict_df.to_csv(args.out_dir / "similarity_update_verdict.csv", index=False)
    rmse_table = build_grid_latex_table(
        summary_df=summary_df,
        datasets=args.datasets,
        decimals=args.decimals,
        value_kind="rmse",
    )
    speedup_table = build_grid_latex_table(
        summary_df=summary_df,
        datasets=args.datasets,
        decimals=args.decimals,
        value_kind="speedup",
    )
    (args.tables_dir / "similarity_update_rmse_table.tex").write_text(
        rmse_table + "\n",
        encoding="utf-8",
    )
    (args.tables_dir / "similarity_update_speedup_table.tex").write_text(
        speedup_table + "\n",
        encoding="utf-8",
    )
    (args.tables_dir / "similarity_update_table.tex").write_text(
        rmse_table + "\n\n" + speedup_table + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
