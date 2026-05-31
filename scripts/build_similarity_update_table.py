#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATASETS = ["polynomial", "multisine"]
DEFAULT_METRICS = ["test_rmse", "test_mae", "total_time"]
UPDATE_ORDER = ["1", "3", "5", "10"]
LOWER_IS_BETTER = {
    "test_rmse": True,
    "test_mae": True,
    "total_time": True,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a paper-ready table to evaluate whether ALE-Frechet similarity "
            "should be recomputed every step."
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
        help="Metric used to select the best config within each update frequency.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to show in the final table.",
    )
    ap.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimals for formatted metric values.",
    )
    return ap.parse_args()


def read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path)


def format_float(value: object, decimals: int) -> str:
    if pd.isna(value):
        return "--"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.{decimals}f}"


def format_mean_std(mean: object, std: object, decimals: int) -> str:
    if pd.isna(mean):
        return "--"
    mean_text = format_float(mean, decimals)
    if pd.isna(std):
        return mean_text
    return f"{mean_text} $\\pm$ {format_float(std, decimals)}"


def format_factor(value: object) -> str:
    if pd.isna(value):
        return "none"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


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


def normalize_update(value: object) -> str:
    if pd.isna(value):
        return "none"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def select_best_per_update(
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

    work["update_label"] = work["sweep__similarity__update_every"].map(normalize_update)
    work["keep_label"] = work["sweep__similarity__keep_epochs"].map(normalize_update)
    work = work[
        (work["update_label"] != "none")
        & (work["keep_label"] != "none")
    ].copy()
    if work.empty:
        raise ValueError(
            f"No ale_frechet rows remain for dataset '{dataset}' after excluding "
            "update_every=none and keep_epochs=none."
        )

    std_col = selection_metric.replace("_mean", "_std")
    sort_cols = ["update_label", selection_metric]
    ascending = [True, True]
    if std_col in work.columns:
        sort_cols.append(std_col)
        ascending.append(True)
    if "combo_index" in work.columns:
        sort_cols.append("combo_index")
        ascending.append(True)

    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    best = work.groupby("update_label", as_index=False, sort=False).head(1).copy()
    best["dataset"] = dataset
    return best.reset_index(drop=True)


def build_summary_row(
    row: pd.Series,
    metrics: Iterable[str],
    decimals: int,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "dataset": row["dataset"],
        "update": row["update_label"],
        "combo_index": int(row["combo_index"]),
        "keep": format_factor(row.get("sweep__similarity__keep_epochs")),
        "lr": format_factor(row.get("sweep__optim__lr")),
        "l2": format_factor(row.get("sweep__regularization__l2")),
    }
    for metric in metrics:
        out[metric] = format_mean_std(
            row.get(f"{metric}_mean", np.nan),
            row.get(f"{metric}_std", np.nan),
            decimals=decimals,
        )
        out[f"{metric}_value"] = row.get(f"{metric}_mean", np.nan)
    return out


def build_summary_df(
    best_rows: pd.DataFrame,
    metrics: Iterable[str],
    decimals: int,
) -> pd.DataFrame:
    rows = [build_summary_row(row, metrics=metrics, decimals=decimals) for _, row in best_rows.iterrows()]
    out = pd.DataFrame(rows)
    out["update"] = pd.Categorical(out["update"], categories=UPDATE_ORDER, ordered=True)
    out = out.sort_values(["update", "dataset"], kind="mergesort").reset_index(drop=True)
    return out


def metric_best_second_map(
    summary_df: pd.DataFrame,
    dataset: str,
    metric: str,
) -> Tuple[object, object]:
    scores = summary_df.loc[summary_df["dataset"] == dataset, ["update", f"{metric}_value"]].copy()
    scores = scores.dropna(subset=[f"{metric}_value"])
    if scores.empty:
        return None, None
    ordered = scores.sort_values(f"{metric}_value", ascending=LOWER_IS_BETTER.get(metric, True))
    best = ordered.iloc[0]["update"] if len(ordered) >= 1 else None
    second = ordered.iloc[1]["update"] if len(ordered) >= 2 else None
    return best, second


def styled_cell(
    summary_df: pd.DataFrame,
    update: str,
    dataset: str,
    metric: str,
) -> str:
    row = summary_df[(summary_df["update"] == update) & (summary_df["dataset"] == dataset)]
    if row.empty:
        return "--"
    cell = row.iloc[0][metric]
    best, second = metric_best_second_map(summary_df, dataset, metric)
    if update == best:
        return r"\textbf{" + cell + "}"
    if update == second:
        return r"\underline{" + cell + "}"
    return cell


def lookup_factor(summary_df: pd.DataFrame, update: str, dataset: str, column: str) -> str:
    row = summary_df[(summary_df["update"] == update) & (summary_df["dataset"] == dataset)]
    if row.empty:
        return "--"
    return latex_escape(row.iloc[0][column])


def build_latex_table(
    summary_df: pd.DataFrame,
    datasets: Iterable[str],
    metrics: Iterable[str],
    selection_metric: str,
) -> str:
    datasets = list(datasets)
    metrics = list(metrics)
    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Best ALE--Fr\'echet configuration for each similarity recomputation frequency. "
        rf"Within each dataset and update value, the shown row is the best configuration selected by "
        rf"{latex_escape(selection_metric)}. This table is intended to answer whether similarity should "
        rf"be recomputed every step.}}"
    )
    lines.append(r"\label{tab:similarity_update_frequency}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * (len(datasets) * (1 + len(metrics))) + r"}")
    lines.append(r"\toprule")

    header_top = ["Update"]
    for dataset in datasets:
        header_top.append(rf"\multicolumn{{{1 + len(metrics)}}}{{c}}{{{latex_escape(dataset)}}}")
    lines.append(" & ".join(header_top) + r" \\")

    cmidrules = [" "]
    start = 2
    for _ in datasets:
        end = start + len(metrics)
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append(" ".join(cmidrules))

    header_bottom = ["Update"]
    for _dataset in datasets:
        header_bottom.append("Keep")
        for metric in metrics:
            header_bottom.append(metric.replace("_", " ").upper())
    lines.append(" & ".join(header_bottom) + r" \\")
    lines.append(r"\midrule")

    for update in UPDATE_ORDER:
        row_cells = [latex_escape(update)]
        for dataset in datasets:
            row_cells.append(lookup_factor(summary_df, update, dataset, "keep"))
            for metric in metrics:
                row_cells.append(styled_cell(summary_df, update, dataset, metric))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_verdict(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in summary_df["dataset"].drop_duplicates():
        dataset_rows = summary_df[summary_df["dataset"] == dataset].copy()
        best_row = dataset_rows.sort_values("test_rmse_value", ascending=True, kind="mergesort").iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "best_update": best_row["update"],
                "best_keep": best_row["keep"],
                "best_combo_index": best_row["combo_index"],
                "best_test_rmse_mean": best_row["test_rmse_value"],
                "best_test_mae_mean": best_row["test_mae_value"] if "test_mae_value" in best_row else np.nan,
                "best_total_time_mean": best_row["total_time_value"] if "total_time_value" in best_row else np.nan,
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
            select_best_per_update(
                dataset_df,
                dataset=dataset,
                selection_metric=args.selection_metric,
            )
        )

    best_rows = pd.concat(blocks, ignore_index=True)
    summary_df = build_summary_df(best_rows, metrics=args.metrics, decimals=args.decimals)
    verdict_df = build_verdict(summary_df)

    summary_df.to_csv(args.out_dir / "similarity_update_summary.csv", index=False)
    verdict_df.to_csv(args.out_dir / "similarity_update_verdict.csv", index=False)
    (args.tables_dir / "similarity_update_table.tex").write_text(
        build_latex_table(
            summary_df=summary_df,
            datasets=args.datasets,
            metrics=args.metrics,
            selection_metric=args.selection_metric,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
