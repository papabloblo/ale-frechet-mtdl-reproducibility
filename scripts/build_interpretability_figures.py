#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the figures used by the interpretability subsection of paper/main.tex.

The script does not retrain models. It loads the saved ALE-Frechet trackers for
the selected ALE-Frechet runs and writes paper-ready figures:

    - paper/figures/interpretability/multisine_interpretability.{pdf,png}
      ALE curves, task similarity matrix, and induced nearest-neighbor graph.
    - paper/figures/interpretability/electricity_similarity_structure.{pdf,png}
      Full electricity similarity matrix, off-diagonal similarity distribution,
      and strongest task-pair graph.

It also writes helper CSVs and a LaTeX snippet with the figure environments.

Examples
--------
    python3 scripts/build_interpretability_figures.py

    python3 scripts/build_interpretability_figures.py --seed 9 --multisine-top-features 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import torch

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

from build_ale_similarity_figures import (
    aggregate_similarity,
    ale_y_limits,
    candidate_run_rows,
    choose_ale_features,
    centered_ale_array,
    latest_epoch_tensor,
    load_tracker,
    save_ale_curves_csv,
    save_similarity_csv,
    task_labels,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for the interpretability figure workflow.

    Defaults target the paper-ready outputs in
    ``paper/figures/interpretability``. Most arguments control where saved
    comparison artifacts are found, how an ALE--Frechet run is selected, and
    how much of the saved ALE/similarity tensors should be shown.
    """
    ap = argparse.ArgumentParser(
        description="Build paper-specific interpretability figures from saved ALE-Frechet trackers."
    )
    ap.add_argument(
        "--comparisons-root",
        type=Path,
        default=Path("results/comparisons"),
        help="Root containing <dataset>/results_mean_std.csv and raw_runs/.",
    )
    ap.add_argument(
        "--logging-root",
        type=Path,
        default=Path("results"),
        help="Root containing ale_frechet/<run_tag>/metrics.pth.",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/interim"),
        help="Root containing dataset CSVs, used only for task labels.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/figures/interpretability"),
        help="Directory where figures, helper CSVs, and the LaTeX snippet are written.",
    )
    ap.add_argument(
        "--selection-metric",
        default="test_rmse_mean",
        help="Metric in results_mean_std.csv used to choose the ALE-Frechet combo.",
    )
    ap.add_argument(
        "--ale-similarity-report",
        type=Path,
        default=Path("paper/figures/ale_similarity/ale_similarity_figure_report.csv"),
        help="Optional report from build_ale_similarity_figures.py. Existing run tags in this report are tried first.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed. If omitted, the best available seed for the selected combo is used.",
    )
    ap.add_argument(
        "--similarity-aggregation",
        choices=["mean", "sum", "max"],
        default="mean",
        help="How to aggregate per-feature similarities into a task-by-task matrix.",
    )
    ap.add_argument(
        "--multisine-top-features",
        type=int,
        default=3,
        help="Number of highest-variance ALE latent features to show in the multisine figure.",
    )
    ap.add_argument(
        "--multisine-feature-indices",
        nargs="*",
        type=int,
        default=None,
        help="Explicit ALE latent feature indices for the multisine figure.",
    )
    ap.add_argument(
        "--electricity-max-tasks",
        type=int,
        default=None,
        help="Optional cap on electricity tasks shown in the heatmap. Defaults to all tasks.",
    )
    ap.add_argument(
        "--electricity-top-pairs",
        type=int,
        default=20,
        help="Number of strongest off-diagonal task pairs shown in the electricity graph and CSV.",
    )
    ap.add_argument("--dpi", type=int, default=300, help="Figure resolution.")
    return ap.parse_args()


def select_tracker_with_tensors(
    dataset: str,
    comparisons_root: Path,
    logging_root: Path,
    selection_metric: str,
    seed: Optional[int],
    tensor_names: Sequence[str],
    preferred_report: Optional[Path] = None,
) -> tuple[str, Path, dict]:
    """
    Select the first ranked or preferred run containing all required tensors.

    When an ALE similarity report is available, its recorded run tags are tried
    first so the interpretability figures can reuse the same artifacts as the
    broader ALE-similarity figure set. If those preferred trackers are missing
    or incomplete, the function falls back to the metric-ranked candidates from
    the comparison tables. A candidate is accepted only if every tensor named in
    ``tensor_names`` has at least one loadable epoch snapshot.
    """
    preferred_run_tags: list[str] = []
    if preferred_report is not None and preferred_report.exists():
        try:
            report = pd.read_csv(preferred_report)
            preferred = report[report["dataset"].astype(str).str.lower() == dataset.lower()]
            for run_tag in preferred["run_tag"].dropna().astype(str).tolist():
                if run_tag not in preferred_run_tags:
                    preferred_run_tags.append(run_tag)
        except Exception:
            preferred_run_tags = []

    # Preferred runs keep cross-figure provenance consistent, but they are not
    # mandatory: older reports may point to removed or partial tracker files.
    preferred_errors = []
    for run_tag in preferred_run_tags:
        try:
            metrics_path, payload = load_tracker(run_tag, logging_root)
            for tensor_name in tensor_names:
                latest_epoch_tensor(payload.get(tensor_name, {}), tensor_name)
            return run_tag, metrics_path, payload
        except Exception as exc:
            preferred_errors.append(f"{run_tag}: {exc}")

    candidates = candidate_run_rows(
        dataset=dataset,
        comparisons_root=comparisons_root,
        selection_metric=selection_metric,
        seed=seed,
    )

    errors = []
    for _, row in candidates.iterrows():
        run_tag = str(row["run_tag"])
        if run_tag in preferred_run_tags:
            continue
        try:
            metrics_path, payload = load_tracker(run_tag, logging_root)
            for tensor_name in tensor_names:
                latest_epoch_tensor(payload.get(tensor_name, {}), tensor_name)
            return run_tag, metrics_path, payload
        except Exception as exc:
            errors.append(f"{run_tag}: {exc}")

    joined = "\n".join((preferred_errors + errors)[:10])
    raise RuntimeError(
        f"Could not find usable tensors {list(tensor_names)} for dataset={dataset}. "
        f"First failures:\n{joined}"
    )


def color_limits(matrix: np.ndarray) -> tuple[float, float]:
    """
    Return finite color-scale limits for a numeric matrix.

    Matplotlib colorbars should ignore NaN/Inf values. If a matrix has no
    finite entries, a neutral ``(0.0, 1.0)`` scale keeps plotting code from
    failing before the upstream artifact problem can be inspected.
    """
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    """
    Return off-diagonal values from a square similarity matrix.

    The electricity distribution panel is meant to summarize pairwise
    similarities between distinct tasks, so self-similarities on the diagonal
    are excluded. Non-square inputs are flattened as a defensive fallback.
    """
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] <= 1:
        return matrix.ravel()
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def top_pairs(matrix: np.ndarray, labels: Sequence[str], n_pairs: int) -> pd.DataFrame:
    """
    Return the strongest finite upper-triangular task pairs.

    Only ``i < j`` entries are considered so symmetric matrices do not produce
    duplicate pairs. The returned DataFrame carries both display labels and
    zero-based indices, allowing the same table to be written to CSV and passed
    directly into the graph drawing helper.
    """
    rows = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            value = float(matrix[i, j])
            if np.isfinite(value):
                rows.append(
                    {
                        "task_i": labels[i] if i < len(labels) else f"T{i + 1}",
                        "task_j": labels[j] if j < len(labels) else f"T{j + 1}",
                        "task_i_index": i,
                        "task_j_index": j,
                        "similarity": value,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("similarity", ascending=False, kind="mergesort").head(n_pairs)


def draw_task_graph(
    ax: plt.Axes,
    matrix: np.ndarray,
    labels: Sequence[str],
    pairs: Optional[pd.DataFrame] = None,
    directed_nearest_neighbor: bool = False,
    label_fontsize: float = 7,
) -> None:
    """
    Draw a circular task graph from nearest neighbors or selected pairs.

    If ``pairs`` is omitted, each task contributes one directed edge to its
    highest-similarity neighbor. If ``pairs`` is provided, those rows are drawn
    as undirected strongest-pair edges. Edge width and opacity are scaled by
    similarity so stronger relationships are visually emphasized.
    """
    n = matrix.shape[0]
    if pairs is None:
        edge_rows = []
        for i in range(n):
            row = matrix[i].copy()
            row[i] = -np.inf
            if not np.isfinite(row).any():
                continue
            j = int(np.nanargmax(row))
            edge_rows.append((i, j, float(matrix[i, j])))
    else:
        edge_rows = [
            (int(row.task_i_index), int(row.task_j_index), float(row.similarity))
            for row in pairs.itertuples(index=False)
        ]

    node_ids = sorted({idx for edge in edge_rows for idx in edge[:2]})
    if not node_ids:
        node_ids = list(range(n))

    angles = np.linspace(0, 2 * np.pi, len(node_ids), endpoint=False)
    positions = {
        node_id: np.array([np.cos(angle), np.sin(angle)])
        for node_id, angle in zip(node_ids, angles)
    }

    weights = np.array([edge[2] for edge in edge_rows], dtype=float)
    w_min, w_max = (float(weights.min()), float(weights.max())) if weights.size else (0.0, 1.0)
    # Avoid divide-by-zero when all selected edges have identical similarity.
    span = max(w_max - w_min, 1e-12)

    for i, j, weight in edge_rows:
        start = positions[i]
        end = positions[j]
        scaled = (weight - w_min) / span
        width = 0.8 + 3.0 * scaled
        alpha = 0.35 + 0.55 * scaled
        if directed_nearest_neighbor:
            ax.annotate(
                "",
                xy=end * 0.88,
                xytext=start * 0.88,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#3b5f8a",
                    "lw": width,
                    "alpha": alpha,
                    "shrinkA": 8,
                    "shrinkB": 8,
                },
            )
        else:
            ax.plot(
                [start[0] * 0.88, end[0] * 0.88],
                [start[1] * 0.88, end[1] * 0.88],
                color="#3b5f8a",
                lw=width,
                alpha=alpha,
                solid_capstyle="round",
            )

    for node_id in node_ids:
        x, y = positions[node_id]
        label = labels[node_id] if node_id < len(labels) else f"T{node_id + 1}"
        ax.text(x, y, str(label), ha="center", va="center", fontsize=label_fontsize, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")


def plot_multisine_interpretability(
    ale_tensor: torch.Tensor,
    similarity: np.ndarray,
    labels: Sequence[str],
    feature_indices: Sequence[int],
    out_base: Path,
    dpi: int,
) -> None:
    """
    Plot the complete multisine interpretability figure.

    The first row shows selected task-specific centered ALE curves. Centering
    removes arbitrary per-task vertical offsets that can otherwise make the
    y-axis much wider than the visible curve variation. The second row shows
    the aggregated task-similarity matrix and a directed nearest-neighbor graph
    derived from that matrix. The figure is saved as both PDF and PNG using
    ``out_base`` as the extensionless base path.
    """
    arr = centered_ale_array(ale_tensor)
    n_features = len(feature_indices)
    ncols = max(3, n_features)
    fig = plt.figure(figsize=(4.1 * ncols, 7.3), constrained_layout=True)
    gs = GridSpec(2, ncols, figure=fig, height_ratios=[1.0, 1.15])

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, arr.shape[0])))
    first_ale_ax = None
    for panel_idx, feature_idx in enumerate(feature_indices):
        ax = fig.add_subplot(gs[0, panel_idx])
        if first_ale_ax is None:
            first_ale_ax = ax
        for task_idx in range(arr.shape[0]):
            label = labels[task_idx] if task_idx < len(labels) else f"T{task_idx + 1}"
            ax.plot(
                arr[task_idx, feature_idx, :, 0],
                arr[task_idx, feature_idx, :, 1],
                lw=1.4,
                color=colors[task_idx % len(colors)],
                label=label,
            )
        ax.set_title(f"ALE latent feature {feature_idx}", fontsize=10)
        ax.set_xlabel("ALE grid")
        ax.set_ylabel("Centered ALE")
        ax.set_ylim(*ale_y_limits(arr[:, feature_idx, :, 1]))
        ax.grid(alpha=0.25, linewidth=0.5)

    for panel_idx in range(n_features, ncols):
        ax = fig.add_subplot(gs[0, panel_idx])
        ax.axis("off")

    heatmap_ax = fig.add_subplot(gs[1, : max(1, ncols - 1)])
    vmin, vmax = color_limits(similarity)
    im = heatmap_ax.imshow(similarity, cmap="viridis", vmin=vmin, vmax=vmax)
    heatmap_ax.set_title("ALE-Frechet task similarity", fontsize=10)
    heatmap_ax.set_xlabel("Task")
    heatmap_ax.set_ylabel("Task")
    heatmap_ax.set_xticks(np.arange(similarity.shape[0]))
    heatmap_ax.set_yticks(np.arange(similarity.shape[0]))
    heatmap_ax.set_xticklabels(labels[: similarity.shape[0]], rotation=0, fontsize=8)
    heatmap_ax.set_yticklabels(labels[: similarity.shape[0]], fontsize=8)
    for i in range(similarity.shape[0]):
        for j in range(similarity.shape[1]):
            heatmap_ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=heatmap_ax, fraction=0.046, pad=0.02, label="Similarity")

    graph_ax = fig.add_subplot(gs[1, ncols - 1])
    graph_ax.set_title("Nearest-neighbor sharing graph", fontsize=10)
    draw_task_graph(graph_ax, similarity, labels, directed_nearest_neighbor=True)

    if first_ale_ax is not None:
        first_ale_ax.legend(loc="upper right", ncol=1, frameon=True, borderpad=0.3)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def plot_electricity_structure(
    matrix: np.ndarray,
    labels: Sequence[str],
    strongest_pairs: pd.DataFrame,
    out_base: Path,
    dpi: int,
) -> None:
    """
    Plot the electricity similarity-structure figure.

    The three panels show complementary views of the same aggregated similarity
    matrix: the full heatmap, the distribution of off-diagonal task-pair
    similarities, and the graph induced by the strongest finite task pairs.
    """
    fig = plt.figure(figsize=(13.0, 5.4), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 0.9, 1.05])

    heatmap_ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = color_limits(matrix)
    im = heatmap_ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    heatmap_ax.set_title("Electricity similarity matrix", fontsize=10)
    heatmap_ax.set_xlabel("Task")
    heatmap_ax.set_ylabel("Task")
    n = matrix.shape[0]
    if n <= 40:
        ticks = np.arange(n)
        tick_labels = list(labels[:n])
    else:
        step = max(1, int(np.ceil(n / 12)))
        ticks = np.arange(0, n, step)
        tick_labels = [str(i) for i in ticks]
    heatmap_ax.set_xticks(ticks)
    heatmap_ax.set_yticks(ticks)
    heatmap_ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    heatmap_ax.set_yticklabels(tick_labels, fontsize=7)
    fig.colorbar(im, ax=heatmap_ax, fraction=0.046, pad=0.02, label="Similarity")

    hist_ax = fig.add_subplot(gs[0, 1])
    values = off_diagonal_values(matrix)
    values = values[np.isfinite(values)]
    hist_ax.hist(values, bins=40, color="#587a9d", edgecolor="white", linewidth=0.5)
    hist_ax.set_title("Off-diagonal similarities", fontsize=10)
    hist_ax.set_xlabel("Similarity")
    hist_ax.set_ylabel("Pair count")
    hist_ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    graph_ax = fig.add_subplot(gs[0, 2])
    graph_ax.set_title("Strongest task pairs", fontsize=10)
    draw_task_graph(
        graph_ax,
        matrix,
        labels,
        pairs=strongest_pairs,
        directed_nearest_neighbor=False,
        label_fontsize=7.5,
    )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def latex_include_path(path: Path) -> str:
    """
    Convert a figure base path to the relative PDF path used in LaTeX.

    Paper sources are compiled from the ``paper`` directory, so paths beginning
    with ``paper/`` are shortened before they are inserted into
    ``\\includegraphics`` commands.
    """
    text = str(path.with_suffix(".pdf"))
    if text.startswith("paper/"):
        text = text[len("paper/") :]
    return text


def write_latex_snippet(out_dir: Path, multisine_base: Path, electricity_base: Path) -> Path:
    """
    Write LaTeX figure environments for the generated plots.

    The snippet references the PDF versions of the multisine and electricity
    figures and includes captions/labels ready to paste or input from the main
    manuscript. The path to the generated ``.tex`` file is returned for the
    command-line report.
    """
    snippet = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{{latex_include_path(multisine_base)}}}
\caption{{Interpretability of ALE--Fr\'echet sharing on the multisine dataset. The panels show task-specific ALE profiles, the corresponding task-similarity matrix, and the nearest-neighbor sharing graph induced by the learned similarities.}}
\label{{fig:multisine_interpretability}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{{latex_include_path(electricity_base)}}}
\caption{{Learned ALE--Fr\'echet task-sharing structure for the electricity dataset. The similarity matrix and off-diagonal distribution show that most task pairs have low coupling, while the strongest task-pair graph highlights localized groups with higher similarity.}}
\label{{fig:electricity_interpretability}}
\end{{figure}}
"""
    path = out_dir / "interpretability_figures.tex"
    path.write_text(snippet, encoding="utf-8")
    return path


def build_multisine(args: argparse.Namespace) -> dict:
    """
    Build the multisine interpretability figure and helper CSV outputs.

    This orchestrates tracker selection, latest-epoch tensor extraction,
    feature selection, plotting, CSV export, and report-row construction for
    the multisine panel set. Both ALE and similarity tensors are required.
    """
    run_tag, metrics_path, payload = select_tracker_with_tensors(
        dataset="multisine",
        comparisons_root=args.comparisons_root,
        logging_root=args.logging_root,
        selection_metric=args.selection_metric,
        seed=args.seed,
        tensor_names=["ale", "similarity"],
        preferred_report=args.ale_similarity_report,
    )
    ale_epoch, ale_tensor = latest_epoch_tensor(payload["ale"], "ale")
    sim_epoch, sim_tensor = latest_epoch_tensor(payload["similarity"], "similarity")
    similarity = aggregate_similarity(sim_tensor, args.similarity_aggregation)
    labels = task_labels("multisine", args.data_root, int(similarity.shape[0]))
    features = choose_ale_features(
        ale_tensor,
        args.multisine_feature_indices,
        args.multisine_top_features,
    )

    out_base = args.out_dir / "multisine_interpretability"
    plot_multisine_interpretability(ale_tensor, similarity, labels, features, out_base, args.dpi)
    save_ale_curves_csv(args.out_dir / "multisine_ale_curves.csv", ale_tensor, labels)
    save_similarity_csv(args.out_dir / "multisine_similarity_matrix.csv", similarity, labels)

    return {
        "figure": "multisine_interpretability",
        "dataset": "multisine",
        "run_tag": run_tag,
        "metrics_path": str(metrics_path),
        "ale_epoch": ale_epoch,
        "similarity_epoch": sim_epoch,
        "ale_shape": tuple(ale_tensor.shape),
        "similarity_shape": tuple(sim_tensor.shape),
        "features_plotted": ",".join(str(i) for i in features),
        "pdf": str(out_base.with_suffix(".pdf")),
        "png": str(out_base.with_suffix(".png")),
    }


def build_electricity(args: argparse.Namespace) -> dict:
    """
    Build the electricity similarity-structure figure and helper CSV outputs.

    Only the saved similarity tensor is required. The aggregated matrix may be
    truncated for readability, then it is used to produce the heatmap,
    off-diagonal distribution, strongest-pair graph, matrix CSV, top-pairs CSV,
    and report metadata.
    """
    run_tag, metrics_path, payload = select_tracker_with_tensors(
        dataset="electricity",
        comparisons_root=args.comparisons_root,
        logging_root=args.logging_root,
        selection_metric=args.selection_metric,
        seed=args.seed,
        tensor_names=["similarity"],
        preferred_report=args.ale_similarity_report,
    )
    sim_epoch, sim_tensor = latest_epoch_tensor(payload["similarity"], "similarity")
    matrix = aggregate_similarity(sim_tensor, args.similarity_aggregation)
    if args.electricity_max_tasks is not None:
        max_tasks = max(1, int(args.electricity_max_tasks))
        matrix = matrix[:max_tasks, :max_tasks]

    labels = task_labels("electricity", args.data_root, matrix.shape[0])
    strongest_pairs = top_pairs(matrix, labels, args.electricity_top_pairs)

    out_base = args.out_dir / "electricity_similarity_structure"
    plot_electricity_structure(matrix, labels, strongest_pairs, out_base, args.dpi)
    save_similarity_csv(args.out_dir / "electricity_similarity_matrix.csv", matrix, labels)
    strongest_pairs.to_csv(args.out_dir / "electricity_top_pairs.csv", index=False)

    return {
        "figure": "electricity_similarity_structure",
        "dataset": "electricity",
        "run_tag": run_tag,
        "metrics_path": str(metrics_path),
        "ale_epoch": "",
        "similarity_epoch": sim_epoch,
        "ale_shape": "",
        "similarity_shape": tuple(sim_tensor.shape),
        "features_plotted": "",
        "pdf": str(out_base.with_suffix(".pdf")),
        "png": str(out_base.with_suffix(".png")),
    }


def main() -> None:
    """
    Run the interpretability figure workflow and write reports/snippets.

    The main routine intentionally delegates dataset-specific work to
    ``build_multisine`` and ``build_electricity``. It only prepares the output
    directory, collects report rows, writes the LaTeX snippet, and prints the
    generated artifact paths.
    """
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = [build_multisine(args), build_electricity(args)]
    report = pd.DataFrame(records)
    report_path = args.out_dir / "interpretability_figure_report.csv"
    report.to_csv(report_path, index=False)

    snippet_path = write_latex_snippet(
        args.out_dir,
        args.out_dir / "multisine_interpretability",
        args.out_dir / "electricity_similarity_structure",
    )

    print("Generated interpretability figures")
    print(f"Output dir: {args.out_dir}")
    print(f"Report    : {report_path}")
    print(f"LaTeX     : {snippet_path}")
    for record in records:
        print(f"- {record['figure']}: {record['pdf']}")


if __name__ == "__main__":
    main()
