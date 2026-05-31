#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from scripts.build_ale_similarity_figures import (
        aggregate_similarity,
        candidate_run_rows,
        latest_epoch_tensor,
        load_tracker,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from build_ale_similarity_figures import (
        aggregate_similarity,
        candidate_run_rows,
        latest_epoch_tensor,
        load_tracker,
    )


DEFAULT_DATASETS = ["multisine", "polynomial"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Validate whether learned ALE--Frechet similarities recover known "
            "synthetic task relationships."
        )
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
        help="Synthetic datasets to validate.",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/interim"),
        help="Root containing synthetic dataset CSVs.",
    )
    ap.add_argument(
        "--comparisons-root",
        type=Path,
        default=Path("results/comparisons"),
        help="Root containing comparison raw_runs and results_mean_std.csv.",
    )
    ap.add_argument(
        "--logging-root",
        type=Path,
        default=Path("results"),
        help="Root containing ale_frechet/<run_tag>/metrics.pth.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/results/interpretability_validation"),
        help="Directory for validation CSV outputs.",
    )
    ap.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("paper/tables"),
        help="Directory for the LaTeX validation table.",
    )
    ap.add_argument(
        "--ale-similarity-report",
        type=Path,
        default=Path("paper/figures/ale_similarity/ale_similarity_figure_report.csv"),
        help="Optional report whose run tags are preferred for provenance consistency.",
    )
    ap.add_argument(
        "--selection-metric",
        default="test_rmse_mean",
        help="Metric used to rank ALE--Frechet runs when no preferred run is available.",
    )
    ap.add_argument(
        "--similarity-aggregation",
        choices=["mean", "sum", "max"],
        default="mean",
        help="How to aggregate feature-wise task similarities.",
    )
    ap.add_argument(
        "--grid-size",
        type=int,
        default=2000,
        help="Number of x-grid points used for ground-truth function distances.",
    )
    return ap.parse_args()


def latex_escape(value: object) -> str:
    replacements = {
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
    return "".join(replacements.get(ch, ch) for ch in str(value))


def read_synthetic_full(dataset: str, data_root: Path) -> pd.DataFrame:
    path = data_root / dataset / f"{dataset}_full.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing synthetic dataset file: {path}")
    df = pd.read_csv(path)
    if "task" not in df.columns or "x" not in df.columns:
        raise ValueError(f"{path} must contain 'task' and 'x' columns.")
    return df


def task_parameter_table(dataset: str, df: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for task in labels:
        sub = df[df["task"].astype(str) == str(task)]
        if sub.empty:
            raise ValueError(f"Task '{task}' was not found in dataset '{dataset}'.")
        first = sub.iloc[0]
        row = {"task": str(task)}
        if dataset == "multisine":
            for col in ["amplitude", "frequency", "phase"]:
                if col not in sub.columns:
                    raise ValueError(f"Missing multisine parameter column: {col}")
                row[col] = float(first[col])
        elif dataset == "polynomial":
            coeff_cols = sorted(
                [c for c in sub.columns if re.fullmatch(r"w\d+", str(c))],
                key=lambda c: int(c[1:]),
            )
            if not coeff_cols:
                raise ValueError("Polynomial data must contain coefficient columns w0, w1, ...")
            for col in coeff_cols:
                row[col] = float(first[col])
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_true_functions(dataset: str, params: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    curves = []
    if dataset == "multisine":
        for _, row in params.iterrows():
            curves.append(row["amplitude"] * np.sin(row["frequency"] * x + row["phase"]))
    elif dataset == "polynomial":
        coeff_cols = sorted(
            [c for c in params.columns if re.fullmatch(r"w\d+", str(c))],
            key=lambda c: int(c[1:]),
        )
        for _, row in params.iterrows():
            y = np.zeros_like(x, dtype=float)
            for col in coeff_cols:
                y += float(row[col]) * (x ** int(col[1:]))
            curves.append(y)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return np.vstack(curves)


def pairwise_rmse(curves: np.ndarray) -> np.ndarray:
    n = curves.shape[0]
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(np.sqrt(np.mean((curves[i] - curves[j]) ** 2)))
            out[i, j] = value
            out[j, i] = value
    return out


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return np.asarray(matrix[idx], dtype=float)


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"


def select_preferred_run_tag(dataset: str, report_path: Path) -> str | None:
    if not report_path.exists():
        return None
    try:
        report = pd.read_csv(report_path)
    except Exception:
        return None
    if "dataset" not in report.columns or "run_tag" not in report.columns:
        return None
    work = report[report["dataset"].astype(str).str.lower() == dataset.lower()].copy()
    if "kind" in work.columns:
        similarity = work[work["kind"].astype(str).str.lower() == "similarity"]
        if not similarity.empty:
            work = similarity
    for run_tag in work["run_tag"].dropna().astype(str).tolist():
        if run_tag:
            return run_tag
    return None


def run_tag_combo_index(run_tag: str) -> int | None:
    match = re.search(r"combo_(\d+)", run_tag)
    return int(match.group(1)) if match else None


def run_tag_seed(run_tag: str) -> int | None:
    match = re.search(r"seed_(\d+)", run_tag)
    return int(match.group(1)) if match else None


def load_similarity_for_run(run_tag: str, logging_root: Path, aggregation: str) -> tuple[np.ndarray, int, Path]:
    metrics_path, payload = load_tracker(run_tag, logging_root)
    epoch, tensor = latest_epoch_tensor(payload.get("similarity", {}), "similarity")
    matrix = aggregate_similarity(tensor, aggregation)
    return matrix, int(epoch), metrics_path


def select_similarity_run(
    dataset: str,
    comparisons_root: Path,
    logging_root: Path,
    selection_metric: str,
    aggregation: str,
    preferred_report: Path,
) -> tuple[str, np.ndarray, int, Path]:
    preferred = select_preferred_run_tag(dataset, preferred_report)
    if preferred:
        try:
            matrix, epoch, metrics_path = load_similarity_for_run(preferred, logging_root, aggregation)
            return preferred, matrix, epoch, metrics_path
        except Exception:
            pass

    candidates = candidate_run_rows(
        dataset=dataset,
        comparisons_root=comparisons_root,
        selection_metric=selection_metric,
        seed=None,
    )
    errors = []
    for _, row in candidates.iterrows():
        run_tag = str(row["run_tag"])
        try:
            matrix, epoch, metrics_path = load_similarity_for_run(run_tag, logging_root, aggregation)
            return run_tag, matrix, epoch, metrics_path
        except Exception as exc:
            errors.append(f"{run_tag}: {exc}")

    raise RuntimeError(
        f"No usable ALE--Frechet similarity tracker found for dataset={dataset}. "
        f"First failures: {'; '.join(errors[:5])}"
    )


def load_same_combo_similarity_matrices(
    dataset: str,
    combo_index: int,
    comparisons_root: Path,
    logging_root: Path,
    aggregation: str,
) -> list[tuple[int, str, np.ndarray]]:
    raw = candidate_run_rows(
        dataset=dataset,
        comparisons_root=comparisons_root,
        selection_metric="test_rmse_mean",
        seed=None,
    )
    combo_token = f"combo_{combo_index:03d}"
    raw = raw[raw["run_tag"].astype(str).str.contains(combo_token, regex=False)].copy()

    matrices = []
    seen = set()
    for _, row in raw.iterrows():
        run_tag = str(row["run_tag"])
        if run_tag in seen:
            continue
        seen.add(run_tag)
        seed = run_tag_seed(run_tag)
        if seed is None:
            continue
        try:
            matrix, _, _ = load_similarity_for_run(run_tag, logging_root, aggregation)
        except Exception:
            continue
        matrices.append((seed, run_tag, matrix))

    matrices.sort(key=lambda item: item[0])
    return matrices


def nearest_neighbor_agreement(true_distance: np.ndarray, learned_similarity: np.ndarray, top_k: int) -> float:
    n = true_distance.shape[0]
    if n <= 1:
        return np.nan
    hits = 0
    for i in range(n):
        true_order = np.argsort(np.where(np.arange(n) == i, np.inf, true_distance[i]))
        learned_order = np.argsort(np.where(np.arange(n) == i, -np.inf, learned_similarity[i]))[::-1]
        true_nn = int(true_order[0])
        learned_top = set(int(x) for x in learned_order[: min(top_k, n - 1)])
        hits += int(true_nn in learned_top)
    return hits / n


def average_seed_stability(matrices: Sequence[np.ndarray]) -> tuple[float, int]:
    if len(matrices) < 2:
        return np.nan, 0
    correlations = []
    for left, right in combinations(matrices, 2):
        a = upper_triangle_values(left)
        b = upper_triangle_values(right)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            continue
        corr = spearmanr(a[mask], b[mask]).statistic
        if np.isfinite(corr):
            correlations.append(float(corr))
    if not correlations:
        return np.nan, 0
    return float(np.mean(correlations)), len(correlations)


def save_matrix(path: Path, matrix: np.ndarray, labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(matrix, index=list(labels), columns=list(labels))
    df.index.name = "task"
    df.to_csv(path)


def build_latex_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "% No interpretability validation results were available.\n"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Quantitative validation of learned ALE--Fr\'{e}chet task "
            r"relationships on synthetic datasets. Spearman correlation compares "
            r"learned similarity with ground-truth functional similarity. Top-$k$ "
            r"agreement reports whether the true nearest neighbor is recovered among "
            r"the learned top-$k$ neighbors.}"
        ),
        r"\label{tab:interpretability_validation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"Dataset & Ground-truth target & $\rho$ & $p$ & "
            r"Top-1 & Top-3 & Seed stability & Seeds \\"
        ),
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        cells = [
            latex_escape(row["dataset"]),
            latex_escape(row["ground_truth_target"]),
            f"{float(row['spearman_rho']):.3f}" if np.isfinite(row["spearman_rho"]) else "--",
            format_p_value(float(row["spearman_p_value"])),
            f"{float(row['top1_agreement']):.3f}" if np.isfinite(row["top1_agreement"]) else "--",
            f"{float(row['top3_agreement']):.3f}" if np.isfinite(row["top3_agreement"]) else "--",
            f"{float(row['seed_stability_rho_mean']):.3f}" if np.isfinite(row["seed_stability_rho_mean"]) else "--",
            str(int(row["n_seed_matrices"])),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def validate_dataset(dataset: str, args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    df = read_synthetic_full(dataset, args.data_root)
    labels = sorted(df["task"].dropna().astype(str).unique().tolist())
    params = task_parameter_table(dataset, df, labels)

    x = np.linspace(float(df["x"].min()), float(df["x"].max()), int(args.grid_size))
    true_curves = evaluate_true_functions(dataset, params, x)
    true_distance = pairwise_rmse(true_curves)
    true_similarity = -true_distance

    run_tag, learned_similarity, epoch, metrics_path = select_similarity_run(
        dataset=dataset,
        comparisons_root=args.comparisons_root,
        logging_root=args.logging_root,
        selection_metric=args.selection_metric,
        aggregation=args.similarity_aggregation,
        preferred_report=args.ale_similarity_report,
    )

    n = min(true_similarity.shape[0], learned_similarity.shape[0], len(labels))
    labels = labels[:n]
    true_distance = true_distance[:n, :n]
    true_similarity = true_similarity[:n, :n]
    learned_similarity = learned_similarity[:n, :n]

    true_vec = upper_triangle_values(true_similarity)
    learned_vec = upper_triangle_values(learned_similarity)
    mask = np.isfinite(true_vec) & np.isfinite(learned_vec)
    if mask.sum() < 2:
        rho = np.nan
        p_value = np.nan
    else:
        result = spearmanr(true_vec[mask], learned_vec[mask])
        rho = float(result.statistic)
        p_value = float(result.pvalue)

    combo_index = run_tag_combo_index(run_tag)
    seed_mats = []
    seed_details = []
    if combo_index is not None:
        for seed, seed_run_tag, matrix in load_same_combo_similarity_matrices(
            dataset=dataset,
            combo_index=combo_index,
            comparisons_root=args.comparisons_root,
            logging_root=args.logging_root,
            aggregation=args.similarity_aggregation,
        ):
            matrix = matrix[:n, :n]
            seed_mats.append(matrix)
            seed_details.append({
                "dataset": dataset,
                "combo_index": combo_index,
                "seed": seed,
                "run_tag": seed_run_tag,
            })

    stability, n_seed_pairs = average_seed_stability(seed_mats)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_matrix(args.out_dir / f"{dataset}_true_function_distance.csv", true_distance, labels)
    save_matrix(args.out_dir / f"{dataset}_true_function_similarity.csv", true_similarity, labels)
    save_matrix(args.out_dir / f"{dataset}_learned_similarity.csv", learned_similarity, labels)
    params.to_csv(args.out_dir / f"{dataset}_task_parameters.csv", index=False)

    summary = {
        "dataset": dataset,
        "ground_truth_target": "functional similarity",
        "spearman_rho": rho,
        "spearman_p_value": p_value,
        "top1_agreement": nearest_neighbor_agreement(true_distance, learned_similarity, top_k=1),
        "top3_agreement": nearest_neighbor_agreement(true_distance, learned_similarity, top_k=3),
        "seed_stability_rho_mean": stability,
        "n_seed_matrices": len(seed_mats),
        "n_seed_pairs": n_seed_pairs,
        "run_tag": run_tag,
        "combo_index": combo_index,
        "similarity_epoch": epoch,
        "metrics_path": str(metrics_path),
        "similarity_aggregation": args.similarity_aggregation,
    }
    return summary, seed_details


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    seed_rows = []
    for dataset in args.datasets:
        summary, seed_details = validate_dataset(dataset, args)
        summaries.append(summary)
        seed_rows.extend(seed_details)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.out_dir / "interpretability_validation_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(args.out_dir / "interpretability_validation_seed_runs.csv", index=False)

    tex = build_latex_table(summary_df)
    table_path = args.tables_dir / "interpretability_validation_table.tex"
    table_path.write_text(tex, encoding="utf-8")

    print(f"Saved: {args.out_dir / 'interpretability_validation_summary.csv'}")
    print(f"Saved: {args.out_dir / 'interpretability_validation_seed_runs.csv'}")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
