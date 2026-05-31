#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {
    "test_rmse": True,
    "test_mae": True,
    "test_mape": True,
    "total_time": True,
}

DEFAULT_DATASETS = ["polynomial", "multisine"]
DEFAULT_METRICS = ["test_rmse", "test_mae", "test_mape", "total_time"]

ABLATION_COLUMNS = {
    "combo_index": "combo",
    "sweep__optim__lr": "lr",
    "sweep__regularization__l2": "l2",
    "sweep__similarity__keep_epochs": "keep",
    "sweep__similarity__update_every": "update",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build paper-ready ALE-Frechet ablation tables for the synthetic datasets."
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
        help="Directory where ablation CSV files will be written.",
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
        help="Synthetic datasets to include.",
    )
    ap.add_argument(
        "--selection-metric",
        default="test_rmse_mean",
        help="Metric used to rank ALE-Frechet configurations.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to display in the ablation tables.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ALE-Frechet configurations to keep per dataset.",
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


def format_float(value: object, decimals: int = 4) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{decimals}f}"


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


def format_mean_std(mean: object, std: object, decimals: int) -> str:
    if pd.isna(mean):
        return "--"
    mean_text = format_float(mean, decimals=decimals)
    if pd.isna(std):
        return mean_text
    return f"{mean_text} $\\pm$ {format_float(std, decimals=decimals)}"


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


def pick_top_configs(
    df: pd.DataFrame,
    selection_metric: str,
    top_k: int,
) -> pd.DataFrame:
    work = df.copy()
    work["method"] = work["method"].astype(str).str.lower()
    work = work[work["method"] == "ale_frechet"].copy()
    if work.empty:
        raise ValueError("No ale_frechet rows found in the ablation source file.")
    if selection_metric not in work.columns:
        raise KeyError(
            f"Selection metric '{selection_metric}' is missing. "
            f"Available columns include: {', '.join(work.columns[:20])}"
        )

    sort_cols = [selection_metric]
    ascending = [True]
    std_col = selection_metric.replace("_mean", "_std")
    if std_col in work.columns:
        sort_cols.append(std_col)
        ascending.append(True)
    if "combo_index" in work.columns:
        sort_cols.append("combo_index")
        ascending.append(True)

    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort").head(top_k).copy()
    work.insert(0, "rank", np.arange(1, len(work) + 1))
    return work.reset_index(drop=True)


def build_display_df(df: pd.DataFrame, metrics: Iterable[str], decimals: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        out_row: Dict[str, object] = {"rank": int(row["rank"])}

        for src, dst in ABLATION_COLUMNS.items():
            if src == "combo_index":
                out_row[dst] = int(row[src])
            else:
                out_row[dst] = format_factor(row.get(src))

        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            out_row[metric] = format_mean_std(
                row.get(mean_col, np.nan),
                row.get(std_col, np.nan),
                decimals=decimals,
            )

        rows.append(out_row)

    return pd.DataFrame(rows)


def style_metric_cell(
    raw_df: pd.DataFrame,
    display_df: pd.DataFrame,
    row_idx: int,
    metric: str,
) -> str:
    mean_col = f"{metric}_mean"
    if mean_col not in raw_df.columns:
        return display_df.iloc[row_idx][metric]

    lower_is_better = LOWER_IS_BETTER.get(metric, True)
    scores = raw_df[mean_col].astype(float)
    ordered = scores.sort_values(ascending=lower_is_better)
    best_idx = ordered.index[0] if len(ordered) >= 1 else None
    second_idx = ordered.index[1] if len(ordered) >= 2 else None

    cell = display_df.iloc[row_idx][metric]
    current_idx = raw_df.index[row_idx]
    if current_idx == best_idx:
        return r"\textbf{" + cell + "}"
    if current_idx == second_idx:
        return r"\underline{" + cell + "}"
    return cell


def build_latex_table(
    dataset: str,
    raw_df: pd.DataFrame,
    display_df: pd.DataFrame,
    metrics: Iterable[str],
    selection_metric: str,
) -> str:
    metrics = list(metrics)
    metric_headers = [metric.replace("_", " ").upper() for metric in metrics]
    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Top ALE--Fr\'echet ablation configurations for {latex_escape(dataset)}. "
        rf"Rows are ranked by {latex_escape(selection_metric)}; best and second-best values "
        rf"within each reported metric are highlighted.}}"
    )
    lines.append(rf"\label{{tab:ablation_{latex_escape(dataset)}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * (5 + len(metrics)) + r"}")
    lines.append(r"\toprule")
    lines.append(
        "Rank & Combo & LR & L2 & Keep & Update & " + " & ".join(metric_headers) + r" \\"
    )
    lines.append(r"\midrule")

    for row_idx in range(len(display_df)):
        display_row = display_df.iloc[row_idx]
        row_cells = [
            str(display_row["rank"]),
            str(display_row["combo"]),
            latex_escape(display_row["lr"]),
            latex_escape(display_row["l2"]),
            latex_escape(display_row["keep"]),
            latex_escape(display_row["update"]),
        ]
        for metric in metrics:
            row_cells.append(style_metric_cell(raw_df, display_df, row_idx, metric))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    for dataset in args.datasets:
        results_path = args.comparisons_root / dataset / "results_mean_std.csv"
        dataset_df = read_results(results_path)
        top_df = pick_top_configs(
            dataset_df,
            selection_metric=args.selection_metric,
            top_k=args.top_k,
        )
        display_df = build_display_df(top_df, metrics=args.metrics, decimals=args.decimals)

        csv_path = args.out_dir / f"{dataset}_ablation_top{args.top_k}.csv"
        tex_path = args.tables_dir / f"{dataset}_ablation_table.tex"

        top_df.to_csv(csv_path, index=False)
        tex_path.write_text(
            build_latex_table(
                dataset=dataset,
                raw_df=top_df,
                display_df=display_df,
                metrics=args.metrics,
                selection_metric=args.selection_metric,
            )
            + "\n",
            encoding="utf-8",
        )

        best_row = top_df.iloc[0]
        summary_rows.append(
            {
                "dataset": dataset,
                "combo_index": int(best_row["combo_index"]),
                "lr": format_factor(best_row.get("sweep__optim__lr")),
                "l2": format_factor(best_row.get("sweep__regularization__l2")),
                "keep": format_factor(best_row.get("sweep__similarity__keep_epochs")),
                "update": format_factor(best_row.get("sweep__similarity__update_every")),
                "selection_metric": args.selection_metric,
                "selection_value": best_row.get(args.selection_metric, np.nan),
            }
        )

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "ablation_best_configs_summary.csv", index=False)


if __name__ == "__main__":
    main()
