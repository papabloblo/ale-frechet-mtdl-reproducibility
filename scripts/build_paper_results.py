#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script orchestrates the generation of all LaTeX tables required for the
Results and Discussion section of the paper. It acts as a single entry point
to transform experimental outputs (CSV files) into publication-ready LaTeX
artifacts.

The script aggregates results from synthetic experiments (e.g., multisine and
polynomial datasets), computes summary statistics, evaluates cross-task
behavior, and produces formatted tables for direct inclusion in the manuscript.

Specifically, it coordinates:
    - Generation of overall performance tables (mean and standard deviation)
    - Analysis of cross-task stability (min, max, dispersion metrics)
    - Ranking of methods based on performance–stability trade-offs
    - Export of LaTeX tables with consistent formatting and labels
    - Friedman tests over method ranks and paired Wilcoxon tests against
      ALE--Frechet for the strongest competing methods

Inputs:
    - CSV files located under:
        results/comparisons/<dataset>/
            - results_mean_std.csv
            - results_task_behavior.csv

Outputs:
    - LaTeX tables written to:
        reports/tables/
            - results_mean_std_table.tex
            - task_behavior_table.tex
            - method_ranking_table.tex

Usage:
    Run from the repository root:
        python -m scripts.latex.tables.build_paper_results

    or via Makefile:
        make paper-results

Notes:
    - This script assumes that all experiments have been executed and the
      corresponding CSV files are available.
    - It is designed to be reproducible and fully deterministic given the input data.
    - The generated tables are directly referenced in the main LaTeX document.

This script is part of the reproducibility pipeline of the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


# ============================================================
# Helpers
# ============================================================

LOWER_IS_BETTER_DEFAULT = {
    "test_loss": True,
    "test_rmse": True,
    "test_mae": True,
    "test_mape": True,
    "val_loss": True,
    "val_rmse": True,
    "val_mae": True,
    "val_mape": True,
    "time_train": True,
    "time_validation": True,
    "time_test": True,
    "time_ale": True,
    "time_similarity": True,
    "total_time": True,
}

DEFAULT_STAT_TEST_METRICS = ["test_rmse", "test_mae"]

METHOD_DISPLAY_NAMES = {
    "ale_frechet": r"ALE--Fr\'{e}chet",
    "crossstitch": "Cross-stitch",
    "hard": "Hard",
    "independent": "Single-task MLP",
    "mmoe": "MMoE",
    "mtan": "MTAN",
    "ple": "PLE",
    "single_task": "Single-task MLP",
    "single-task": "Single-task MLP",
    "single-task-mlp": "Single-task MLP",
    "single_task_mlp": "Single-task MLP",
    "soft": "Soft",
    "stl": "Single-task MLP",
}

METRIC_TOKEN_DISPLAY_NAMES = {
    "ale": "ALE",
    "mae": "MAE",
    "mape": "MAPE",
    "rmse": "RMSE",
}

METRIC_CONTEXT_DISPLAY_NAMES = {
    "train": "training",
    "test": "test",
    "val": "validation",
    "validation": "validation",
}

METRIC_STAT_DISPLAY_NAMES = {
    "mean": "mean",
    "std": "standard deviation",
    "n": "sample count",
}

LATEX_TEXT_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Could not read CSV: {path}\n{e}") from e


def normalize_method_name(name: str) -> str:
    name = str(name).strip().lower()
    aliases = {
        "ale": "ale_frechet",
        "ours": "ale_frechet",
        "independent": "single_task_mlp",
        "single_task": "single_task_mlp",
        "single-task": "single_task_mlp",
        "single-task-mlp": "single_task_mlp",
        "stl": "single_task_mlp",
    }
    return aliases.get(name, name)


def infer_dataset_name_from_path(path: Path) -> str:
    # expected: .../comparisons/<dataset>/results_mean_std.csv
    parent = path.parent.name
    if parent and parent != "comparisons":
        return parent
    return path.stem


def normalize_dataset_name(name: str) -> str:
    return str(name).strip().lower()


def discover_dataset_result_files(comparisons_dir: Path) -> List[Path]:
    files = []
    for p in sorted(comparisons_dir.glob("*/results_mean_std.csv")):
        if p.is_file():
            files.append(p)
    return files


def find_selection_metric(df: pd.DataFrame, preferred: str | None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)

    candidates.extend([
        "val_rmse_mean",
        "val_loss_mean",
        "val_mae_mean",
        "val_mape_mean",
        "test_rmse_mean",
        "test_loss_mean",
        "test_mae_mean",
        "test_mape_mean",
    ])

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "Could not find a metric to select the best configuration. "
        f"Available columns: {list(df.columns)}"
    )


def metric_base_name(metric: str) -> str:
    return metric[:-5] if metric.endswith("_mean") else metric


def is_lower_better(metric: str, override: Dict[str, bool] | None = None) -> bool:
    base = metric_base_name(metric)
    if override and base in override:
        return override[base]
    return LOWER_IS_BETTER_DEFAULT.get(base, True)


def select_best_row_per_method(
    df: pd.DataFrame,
    dataset_name: str,
    selection_metric: str,
    lower_is_better: bool,
) -> pd.DataFrame:
    required = {"method", selection_metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required columns: {sorted(missing)}"
        )

    work = df.copy()
    work["dataset"] = dataset_name
    work["method"] = work["method"].map(normalize_method_name)

    # Stable sorting for deterministic tie-breaking
    sort_cols = [selection_metric]
    ascending = [lower_is_better]

    # Prefer lower std when means tie
    std_col = metric_base_name(selection_metric) + "_std"
    if std_col in work.columns:
        sort_cols.append(std_col)
        ascending.append(True)

    # Prefer smaller combo index for full reproducibility
    if "combo_index" in work.columns:
        sort_cols.append("combo_index")
        ascending.append(True)

    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    best = work.groupby("method", as_index=False, sort=False).head(1).reset_index(drop=True)
    return best


def collect_best_results(
    comparisons_dir: Path,
    selection_metric: str | None,
    exclude_datasets: List[str] | None = None,
    exclude_methods: List[str] | None = None,
) -> Tuple[pd.DataFrame, str]:
    files = discover_dataset_result_files(comparisons_dir)
    if not files:
        raise FileNotFoundError(
            f"No dataset result files found under: {comparisons_dir}\n"
            "Expected files like: comparisons/<dataset>/results_mean_std.csv"
        )

    selected_blocks = []
    chosen_selection_metric = None
    excluded = {
        normalize_dataset_name(dataset)
        for dataset in (exclude_datasets or [])
        if str(dataset).strip()
    }
    excluded_methods = {
        normalize_method_name(method)
        for method in (exclude_methods or [])
        if str(method).strip()
    }

    for path in files:
        dataset_name = infer_dataset_name_from_path(path)
        if normalize_dataset_name(dataset_name) in excluded:
            continue

        df = read_csv_safe(path)
        if df.empty:
            continue

        if excluded_methods:
            if "method" not in df.columns:
                raise ValueError(
                    f"Dataset '{dataset_name}' is missing required column 'method' "
                    "needed for method exclusions."
                )
            normalized_methods = df["method"].map(normalize_method_name)
            df = df.loc[~normalized_methods.isin(excluded_methods)].copy()
            if df.empty:
                continue

        sel_metric = find_selection_metric(df, selection_metric)

        if chosen_selection_metric is None:
            chosen_selection_metric = sel_metric

        best = select_best_row_per_method(
            df=df,
            dataset_name=dataset_name,
            selection_metric=sel_metric,
            lower_is_better=is_lower_better(sel_metric),
        )
        selected_blocks.append(best)

    if not selected_blocks:
        raise ValueError(
            "No result rows remained after applying exclusions. "
            f"Excluded datasets: {sorted(excluded) if excluded else 'none'}. "
            f"Excluded methods: {sorted(excluded_methods) if excluded_methods else 'none'}."
        )

    out = pd.concat(selected_blocks, ignore_index=True)
    return out, chosen_selection_metric


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def rank_metric_within_dataset(
    df: pd.DataFrame,
    metric_mean_col: str,
    lower_is_better_metric: bool,
) -> pd.Series:
    values = df[metric_mean_col].astype(float)
    return values.rank(method="min", ascending=lower_is_better_metric)


def build_long_table(best_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "method", "combo_index", "combo_tag"]
    ordered = [c for c in cols if c in best_df.columns]

    metric_cols = []
    for c in best_df.columns:
        if c.endswith("_mean") or c.endswith("_std") or c.endswith("_n"):
            metric_cols.append(c)

    extra_cols = [c for c in best_df.columns if c not in ordered + metric_cols]
    return best_df[ordered + extra_cols + metric_cols].copy()


def build_wide_metric_table(
    best_df: pd.DataFrame,
    metric: str,
    decimals: int = 4,
) -> pd.DataFrame:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in best_df.columns:
        raise ValueError(f"Missing column: {mean_col}")

    work = best_df[["dataset", "method", mean_col] + ([std_col] if std_col in best_df.columns else [])].copy()

    if std_col not in work.columns:
        work[std_col] = np.nan

    work["formatted"] = [
        format_mean_std(m, s, decimals=decimals)
        for m, s in zip(work[mean_col], work[std_col])
    ]

    wide = work.pivot(index="dataset", columns="method", values="formatted")
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    return wide


def latex_escape_text(value: object) -> str:
    """Escape plain text for use in LaTeX text/table cells."""
    return "".join(LATEX_TEXT_REPLACEMENTS.get(char, char) for char in str(value))


def display_dataset_name(name: str) -> str:
    return latex_escape_text(name)


def display_metric_name(metric: str, *, uppercase: bool = True) -> str:
    words = []
    for token in str(metric).split("_"):
        lowered = token.lower()
        if lowered in METRIC_TOKEN_DISPLAY_NAMES:
            words.append(METRIC_TOKEN_DISPLAY_NAMES[lowered])
        elif uppercase:
            words.append(token.upper())
        else:
            words.append(token.lower())
    return latex_escape_text(" ".join(words))


def display_caption_metric_name(metric: str) -> str:
    tokens = [token.lower() for token in str(metric).split("_") if token]
    if not tokens:
        return ""

    statistic = None
    if tokens[-1] in METRIC_STAT_DISPLAY_NAMES:
        statistic = METRIC_STAT_DISPLAY_NAMES[tokens.pop()]

    if tokens and tokens[0] == "time":
        tokens = tokens[1:] + ["time"]

    words = []
    for token in tokens:
        if token in METRIC_CONTEXT_DISPLAY_NAMES:
            words.append(METRIC_CONTEXT_DISPLAY_NAMES[token])
        elif token in METRIC_TOKEN_DISPLAY_NAMES:
            words.append(METRIC_TOKEN_DISPLAY_NAMES[token])
        else:
            words.append(token)

    if statistic:
        words.insert(0, statistic)

    return latex_escape_text(" ".join(words))


def sentence_case_preserving_acronyms(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def display_method_name(method: str) -> str:
    normalized = normalize_method_name(method)
    return METHOD_DISPLAY_NAMES.get(normalized, latex_escape_text(method))


def build_latex_table(
    best_df: pd.DataFrame,
    metric: str,
    caption: str,
    label: str,
    decimals: int = 4,
    underline_second: bool = True,
) -> str:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in best_df.columns:
        raise ValueError(f"Missing column: {mean_col}")

    methods = sorted(best_df["method"].dropna().unique().tolist())
    datasets = sorted(best_df["dataset"].dropna().unique().tolist())
    lower = is_lower_better(metric)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(methods) + "}")
    lines.append(r"\toprule")
    lines.append("Dataset & " + " & ".join(display_method_name(method) for method in methods) + r" \\")
    lines.append(r"\midrule")

    for dataset in datasets:
        sub = best_df[best_df["dataset"] == dataset].copy()
        sub = sub.set_index("method")

        available_methods = [m for m in methods if m in sub.index]
        scores = sub.loc[available_methods, mean_col].astype(float)

        if lower:
            ordered = scores.sort_values(ascending=True)
        else:
            ordered = scores.sort_values(ascending=False)

        best_method = ordered.index[0] if len(ordered) >= 1 else None
        second_method = ordered.index[1] if len(ordered) >= 2 else None

        row_vals = []
        for method in methods:
            if method not in sub.index:
                row_vals.append("--")
                continue

            mean_val = float(sub.loc[method, mean_col])
            std_val = float(sub.loc[method, std_col]) if std_col in sub.columns else np.nan
            cell = format_mean_std(mean_val, std_val, decimals=decimals)

            if method == best_method:
                cell = r"\textbf{" + cell + "}"
            elif underline_second and method == second_method:
                cell = r"\underline{" + cell + "}"

            row_vals.append(cell)

        lines.append(display_dataset_name(dataset) + " & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def compute_rankings(best_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []

    for metric in metrics:
        mean_col = f"{metric}_mean"
        if mean_col not in best_df.columns:
            continue

        lower = is_lower_better(metric)

        tmp = best_df[["dataset", "method", mean_col]].copy()
        tmp["rank"] = tmp.groupby("dataset", group_keys=False)[mean_col].apply(
            lambda s: s.rank(method="min", ascending=lower)
        )

        grouped = tmp.groupby("method", dropna=False)
        summary = grouped["rank"].agg(["mean", "std", "count"]).reset_index()
        summary["metric"] = metric
        summary = summary.rename(
            columns={
                "mean": "avg_rank",
                "std": "std_rank",
                "count": "n_datasets",
            }
        )
        rows.append(summary)

    if not rows:
        return pd.DataFrame(columns=["metric", "method", "avg_rank", "std_rank", "n_datasets"])

    out = pd.concat(rows, ignore_index=True)
    return out[["metric", "method", "avg_rank", "std_rank", "n_datasets"]]


def add_method_ranking_positions(ranking_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["metric", "method_rank", "method", "avg_rank", "std_rank", "n_datasets"]
    if ranking_df.empty:
        return pd.DataFrame(columns=columns)

    required = {"metric", "method", "avg_rank", "std_rank", "n_datasets"}
    missing = required - set(ranking_df.columns)
    if missing:
        raise ValueError(f"Ranking table is missing required columns: {sorted(missing)}")

    work = ranking_df.copy()
    work["avg_rank"] = pd.to_numeric(work["avg_rank"], errors="coerce")
    work["method_rank"] = (
        work.groupby("metric", group_keys=False)["avg_rank"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )
    work = work.sort_values(
        ["metric", "method_rank", "avg_rank", "method"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return work[columns]


def format_rank_value(value: object) -> str:
    if pd.isna(value):
        return "--"
    return str(int(value))


def build_method_ranking_latex_table(
    ranking_df: pd.DataFrame,
    caption: str,
    label: str,
    decimals: int = 3,
) -> str:
    if ranking_df.empty:
        return "% No method ranking results were available.\n"

    work = add_method_ranking_positions(ranking_df)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrlrr}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Rank & Method & Avg. rank & Rank std. \\")
    lines.append(r"\midrule")

    for metric, sub in work.groupby("metric", sort=False):
        first_metric_row = True
        for _, row in sub.iterrows():
            rank_value = format_rank_value(row["method_rank"])
            method = display_method_name(row["method"])
            avg_rank = format_stat_value(row["avg_rank"], decimals=decimals)
            std_rank = format_stat_value(row["std_rank"], decimals=decimals)

            if pd.notna(row["method_rank"]) and int(row["method_rank"]) == 1:
                rank_value = r"\textbf{" + rank_value + "}"
                method = r"\textbf{" + method + "}"
                avg_rank = r"\textbf{" + avg_rank + "}"

            metric_cell = display_metric_name(metric) if first_metric_row else ""
            first_metric_row = False

            lines.append(
                " & ".join([metric_cell, rank_value, method, avg_rank, std_rank])
                + r" \\"
            )
        lines.append(r"\addlinespace")

    if lines[-1] == r"\addlinespace":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def format_stat_value(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"


def build_rank_matrix(best_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    mean_col = f"{metric}_mean"
    if mean_col not in best_df.columns:
        raise ValueError(f"Missing column: {mean_col}")

    lower = is_lower_better(metric)
    work = best_df[["dataset", "method", mean_col]].copy()
    work[mean_col] = pd.to_numeric(work[mean_col], errors="coerce")
    work = work.dropna(subset=["dataset", "method", mean_col])
    work["method"] = work["method"].map(normalize_method_name)
    work["rank"] = work.groupby("dataset", group_keys=False)[mean_col].rank(
        method="average",
        ascending=lower,
    )

    matrix = work.pivot(index="dataset", columns="method", values="rank")
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    return matrix


def run_friedman_test(rank_matrix: pd.DataFrame) -> Tuple[float, float, int, int]:
    complete = rank_matrix.dropna(axis=0, how="any")
    n_datasets = int(complete.shape[0])
    n_methods = int(complete.shape[1])

    if n_datasets < 2 or n_methods < 3:
        return np.nan, np.nan, n_datasets, n_methods

    try:
        stat, p_value = friedmanchisquare(
            *[complete[col].astype(float).to_numpy() for col in complete.columns]
        )
    except ValueError:
        return np.nan, np.nan, n_datasets, n_methods

    return float(stat), float(p_value), n_datasets, n_methods


def run_pairwise_wilcoxon(
    rank_matrix: pd.DataFrame,
    target_method: str,
    competitor_method: str,
) -> Dict[str, object]:
    target_method = normalize_method_name(target_method)
    competitor_method = normalize_method_name(competitor_method)

    pair = rank_matrix[[target_method, competitor_method]].dropna(axis=0, how="any")
    n_datasets = int(pair.shape[0])

    if n_datasets == 0:
        return {
            "wilcoxon_w": np.nan,
            "wilcoxon_p": np.nan,
            "pair_n_datasets": 0,
            "ale_wins": 0,
            "ties": 0,
            "ale_losses": 0,
            "median_rank_delta": np.nan,
            "paired_datasets": "",
        }

    target = pair[target_method].astype(float)
    competitor = pair[competitor_method].astype(float)
    diff = target - competitor

    ale_wins = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    ale_losses = int((diff > 0).sum())
    median_rank_delta = float(diff.median())

    if n_datasets < 2 or np.allclose(diff.to_numpy(), 0.0):
        stat = 0.0
        p_value = 1.0
    else:
        try:
            test = wilcoxon(
                target.to_numpy(),
                competitor.to_numpy(),
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            stat = float(test.statistic)
            p_value = float(test.pvalue)
        except ValueError:
            stat = np.nan
            p_value = np.nan

    return {
        "wilcoxon_w": stat,
        "wilcoxon_p": p_value,
        "pair_n_datasets": n_datasets,
        "ale_wins": ale_wins,
        "ties": ties,
        "ale_losses": ale_losses,
        "median_rank_delta": median_rank_delta,
        "paired_datasets": ",".join(str(x) for x in pair.index.tolist()),
    }


def add_holm_adjusted_p_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["wilcoxon_p_holm"] = []
        return df

    blocks = []
    for _, block in df.groupby("metric", sort=False):
        block = block.copy()
        block["wilcoxon_p_holm"] = np.nan

        valid = block["wilcoxon_p"].dropna().sort_values(kind="mergesort")
        m = len(valid)
        running_max = 0.0
        for rank_index, (idx, p_value) in enumerate(valid.items(), start=1):
            adjusted = min(1.0, float(p_value) * (m - rank_index + 1))
            running_max = max(running_max, adjusted)
            block.loc[idx, "wilcoxon_p_holm"] = running_max

        blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


def build_statistical_tests(
    best_df: pd.DataFrame,
    metrics: List[str],
    target_method: str = "ale_frechet",
    top_competitors: int = 3,
) -> pd.DataFrame:
    rows = []
    target_method = normalize_method_name(target_method)

    for metric in metrics:
        mean_col = f"{metric}_mean"
        if mean_col not in best_df.columns:
            continue

        rank_matrix = build_rank_matrix(best_df, metric)
        if target_method not in rank_matrix.columns:
            continue

        friedman_stat, friedman_p, friedman_n, friedman_k = run_friedman_test(rank_matrix)

        avg_ranks = rank_matrix.mean(axis=0, skipna=True).sort_values(kind="mergesort")
        competitors = [m for m in avg_ranks.index.tolist() if m != target_method]
        competitors = competitors[: max(0, int(top_competitors))]

        for competitor in competitors:
            pair_stats = run_pairwise_wilcoxon(
                rank_matrix=rank_matrix,
                target_method=target_method,
                competitor_method=competitor,
            )
            rows.append({
                "metric": metric,
                "friedman_chi2": friedman_stat,
                "friedman_p": friedman_p,
                "friedman_n_datasets": friedman_n,
                "friedman_n_methods": friedman_k,
                "target_method": target_method,
                "target_avg_rank": float(avg_ranks[target_method]),
                "competitor_method": competitor,
                "competitor_avg_rank": float(avg_ranks[competitor]),
                **pair_stats,
            })

    out = pd.DataFrame(rows)
    return add_holm_adjusted_p_values(out)


def build_statistical_tests_latex_table(
    stats_df: pd.DataFrame,
    caption: str,
    label: str,
) -> str:
    if stats_df.empty:
        return "% No statistical test results were available.\n"

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrrlrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Metric & $N_F$ & $k$ & Friedman $\chi^2$ & $p_F$ & "
        r"ALE rank & Competitor & Comp. rank & $N_W$ & $W$ & $p_W$ & "
        r"$p_{\mathrm{Holm}}$ \\"
    )
    lines.append(r"\midrule")

    for _, row in stats_df.iterrows():
        cells = [
            display_metric_name(row["metric"]),
            str(int(row["friedman_n_datasets"])),
            str(int(row["friedman_n_methods"])),
            format_stat_value(row["friedman_chi2"]),
            format_p_value(row["friedman_p"]),
            format_stat_value(row["target_avg_rank"]),
            display_method_name(row["competitor_method"]),
            format_stat_value(row["competitor_avg_rank"]),
            str(int(row["pair_n_datasets"])),
            format_stat_value(row["wilcoxon_w"]),
            format_p_value(row["wilcoxon_p"]),
            format_p_value(row["wilcoxon_p_holm"]),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True if df.index.name is not None else False)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build paper-ready result tables from sweep outputs."
    )
    ap.add_argument(
        "--comparisons-dir",
        type=Path,
        default=Path("experiments/results/comparisons"),
        help="Root directory containing comparisons/<dataset>/results_mean_std.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/results/paper_results"),
        help="Output directory for paper-ready CSV/LaTeX files.",
    )
    ap.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("paper/tables"),
        help="Table directory",
    )
    ap.add_argument(
        "--selection-metric",
        type=str,
        default=None,
        help=(
            "Metric used to choose the best hyperparameter combo per dataset and method. "
            "Example: val_rmse_mean. If omitted, it is inferred."
        ),
    )
    ap.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=[],
        help=(
            "Dataset names to exclude from paper-ready tables. "
            "Example: --exclude-datasets nn5 metrla."
        ),
    )
    ap.add_argument(
        "--exclude-methods",
        nargs="*",
        default=[],
        help=(
            "Method names to exclude from paper-ready tables. "
            "Names are normalized, so aliases such as 'ale' and 'ours' map to ale_frechet. "
            "Example: --exclude-methods mtan soft."
        ),
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["test_rmse", "test_mae", "test_mape", "total_time"],
        help="Metrics to export as wide CSV and LaTeX tables.",
    )
    ap.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimals in formatted paper tables.",
    )
    ap.add_argument(
        "--stats-metrics",
        nargs="+",
        default=DEFAULT_STAT_TEST_METRICS,
        help=(
            "Metrics used for Friedman/Wilcoxon statistical tests. "
            "Defaults to performance metrics and excludes runtime."
        ),
    )
    ap.add_argument(
        "--stats-top-competitors",
        type=int,
        default=3,
        help="Number of strongest non-ALE methods to compare against ALE--Frechet.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_df, chosen_selection_metric = collect_best_results(
        comparisons_dir=args.comparisons_dir,
        selection_metric=args.selection_metric,
        exclude_datasets=args.exclude_datasets,
        exclude_methods=args.exclude_methods,
    )

    # Save raw selected-best rows
    best_df = best_df.sort_values(["dataset", "method"], kind="mergesort").reset_index(drop=True)
    best_long = build_long_table(best_df)

    best_long.to_csv(args.out_dir / "paper_results_long.csv", index=False)
    best_df.to_csv(args.out_dir / "best_configs.csv", index=False)

    # Save wide tables per metric
    available_metrics = []
    for metric in args.metrics:
        mean_col = f"{metric}_mean"
        if mean_col not in best_df.columns:
            print(f"[WARN] Skipping metric '{metric}' because column '{mean_col}' is missing.")
            continue

        available_metrics.append(metric)

        wide = build_wide_metric_table(best_df, metric=metric, decimals=args.decimals)
        wide.index.name = "dataset"
        wide.to_csv(args.out_dir / f"{metric}_wide.csv")

        caption = (
            f"{sentence_case_preserving_acronyms(display_caption_metric_name(metric))} "
            f"by dataset and method for configurations selected on validation data. "
            f"Best values are in bold, and second-best values are underlined. "
            f"Selection was based on {display_caption_metric_name(chosen_selection_metric)}."
        )
        label = f"tab:{metric}"
        tex = build_latex_table(
            best_df=best_df,
            metric=metric,
            caption=caption,
            label=label,
            decimals=args.decimals,
            underline_second=True,
        )
        save_text(args.tables_dir / f"{metric}_table.tex", tex)

    # Ranking summary
    ranking_df = compute_rankings(best_df, available_metrics)
    ranking_df = add_method_ranking_positions(ranking_df)
    ranking_df.to_csv(args.out_dir / "method_ranking.csv", index=False)

    ranking_caption = (
        "Method ranking by average per-dataset rank for each reported metric. "
        "Lower average rank is better; the best-ranked method for each metric is shown in bold."
    )
    ranking_tex = build_method_ranking_latex_table(
        ranking_df=ranking_df,
        caption=ranking_caption,
        label="tab:method_ranking",
        decimals=3,
    )
    save_text(args.tables_dir / "method_ranking_table.tex", ranking_tex)

    # Statistical tests over per-dataset method ranks.
    stats_metrics = [m for m in args.stats_metrics if f"{m}_mean" in best_df.columns]
    skipped_stats_metrics = [m for m in args.stats_metrics if f"{m}_mean" not in best_df.columns]
    for metric in skipped_stats_metrics:
        print(f"[WARN] Skipping statistical tests for '{metric}' because column '{metric}_mean' is missing.")

    stats_df = build_statistical_tests(
        best_df=best_df,
        metrics=stats_metrics,
        target_method="ale_frechet",
        top_competitors=args.stats_top_competitors,
    )
    stats_df = stats_df.reset_index(drop=True)
    stats_df.to_csv(args.out_dir / "rank_statistical_tests.csv", index=False)

    stats_caption = (
        "Friedman rank tests and paired Wilcoxon signed-rank comparisons between "
        "ALE--Fr\\'{e}chet and the strongest competing methods. For each metric, "
        "competitors are selected by lowest average rank; Wilcoxon p-values are "
        "Holm-adjusted within metric."
    )
    stats_tex = build_statistical_tests_latex_table(
        stats_df=stats_df,
        caption=stats_caption,
        label="tab:rank_statistical_tests",
    )
    save_text(args.tables_dir / "rank_statistical_tests_table.tex", stats_tex)

    # Human-readable summary
    summary_lines = []
    summary_lines.append("Paper results summary")
    summary_lines.append("====================")
    summary_lines.append(f"Comparisons dir      : {args.comparisons_dir}")
    summary_lines.append(f"Output dir           : {args.out_dir}")
    summary_lines.append(f"Tables dir           : {args.tables_dir}")
    summary_lines.append(f"Selection metric     : {chosen_selection_metric}")
    summary_lines.append(
        f"Excluded datasets    : {' '.join(args.exclude_datasets) if args.exclude_datasets else 'none'}"
    )
    summary_lines.append(
        f"Excluded methods     : {' '.join(args.exclude_methods) if args.exclude_methods else 'none'}"
    )
    summary_lines.append(f"Datasets             : {best_df['dataset'].nunique()}")
    summary_lines.append(f"Methods              : {best_df['method'].nunique()}")
    summary_lines.append(f"Rows in best_configs : {len(best_df)}")
    summary_lines.append(f"Stat test metrics    : {' '.join(stats_metrics) if stats_metrics else 'none'}")
    summary_lines.append(f"Stat test competitors: {args.stats_top_competitors}")
    summary_lines.append("")
    summary_lines.append("Generated files:")
    summary_lines.append("- best_configs.csv")
    summary_lines.append("- paper_results_long.csv")
    summary_lines.append("- method_ranking.csv")
    summary_lines.append("- rank_statistical_tests.csv")
    summary_lines.append(f"- {args.tables_dir}/method_ranking_table.tex")
    summary_lines.append(f"- {args.tables_dir}/rank_statistical_tests_table.tex")
    for metric in available_metrics:
        summary_lines.append(f"- {metric}_wide.csv")
        summary_lines.append(f"- {args.tables_dir}/{metric}_table.tex")

    save_text(args.out_dir / "README.txt", "\n".join(summary_lines))

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
