#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build paper-ready ALE-curve and ALE--Frechet similarity figures from saved
ALE--Frechet tracker files.

The script does not retrain models. It selects an ALE--Frechet run from
results/comparisons/<dataset>/results_mean_std.csv and raw_runs/, loads the
corresponding results/ale_frechet/<run_tag>/metrics.pth tracker, and exports:

    - ALE curves for synthetic datasets (multisine and/or polynomial)
    - Similarity heatmaps for multisine and real datasets such as electricity
      or METR-LA

Examples
--------
Build the default requested figures:

    python scripts/build_ale_similarity_figures.py

Build ALE curves only for polynomial and a METR-LA heatmap:

    python scripts/build_ale_similarity_figures.py \\
        --ale-datasets polynomial \\
        --heatmap-datasets metrla
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DEFAULT_ALE_DATASETS = ["multisine", "polynomial"]
DEFAULT_HEATMAP_DATASETS = ["multisine", "electricity"]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for ALE curve and similarity figure export.

    The defaults mirror the paper workflow: ALE curves are generated for the
    synthetic datasets, and heatmaps are generated for multisine and
    electricity. Paths are kept relative to the repository root so the script
    can be run directly from the checkout.
    """
    ap = argparse.ArgumentParser(
        description="Build ALE-curve plots and ALE--Frechet similarity heatmaps from saved tracker files."
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
        default=Path("paper/figures/ale_similarity"),
        help="Directory where figures and helper CSVs are written.",
    )
    ap.add_argument(
        "--ale-datasets",
        nargs="*",
        default=DEFAULT_ALE_DATASETS,
        choices=["multisine", "polynomial"],
        help="Synthetic datasets for ALE-curve figures. Use an empty value to skip.",
    )
    ap.add_argument(
        "--heatmap-datasets",
        nargs="*",
        default=DEFAULT_HEATMAP_DATASETS,
        choices=["multisine", "electricity", "metrla"],
        help="Datasets for similarity heatmaps. Use an empty value to skip.",
    )
    ap.add_argument(
        "--selection-metric",
        default="test_rmse_mean",
        help="Metric in results_mean_std.csv used to choose the ALE--Frechet combo.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed. If omitted, the best available seed for the selected combo is used.",
    )
    ap.add_argument(
        "--ale-top-features",
        type=int,
        default=6,
        help="Number of highest-variance ALE latent features to plot.",
    )
    ap.add_argument(
        "--ale-feature-indices",
        nargs="*",
        type=int,
        default=None,
        help="Explicit ALE latent feature indices to plot. Overrides --ale-top-features.",
    )
    ap.add_argument(
        "--max-heatmap-tasks",
        type=int,
        default=None,
        help="Optional cap on tasks shown in heatmaps. Defaults to all tasks.",
    )
    ap.add_argument(
        "--similarity-aggregation",
        choices=["mean", "sum", "max"],
        default="mean",
        help="How to aggregate per-feature similarities into a task-by-task matrix.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution.",
    )
    return ap.parse_args()


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    """
    Return the first path in ``paths`` that exists.

    This lets callers express a priority order for possible dataset files
    without duplicating the same existence-check loop.
    """
    for path in paths:
        if path.exists():
            return path
    return None


def task_labels(dataset: str, data_root: Path, n_tasks: int) -> list[str]:
    """
    Load stable task labels for a dataset.

    The function prefers the compact metadata file, then the smaller split CSVs
    before falling back to train/full. Only the ``task`` column is needed for
    labels, so reading a multi-GB training file should be a last resort. If the
    data files are unavailable or malformed, it returns generic labels ``T1`` ...
    ``Tn`` so figure generation can still proceed.
    """
    candidates = [
        data_root / dataset / f"{dataset}_meta.csv",
        data_root / dataset / f"{dataset}_test.csv",
        data_root / dataset / f"{dataset}_val.csv",
        data_root / dataset / f"{dataset}_train.csv",
        data_root / dataset / f"{dataset}_full.csv",
    ]
    csv_path = _first_existing(candidates)
    if csv_path is None:
        return [f"T{i + 1}" for i in range(n_tasks)]

    try:
        tasks = pd.read_csv(csv_path, usecols=["task"])["task"].dropna().unique().tolist()
    except Exception:
        return [f"T{i + 1}" for i in range(n_tasks)]

    labels = [str(t) for t in sorted(tasks, key=lambda x: str(x))]
    if len(labels) >= n_tasks:
        return labels[:n_tasks]
    return labels + [f"T{i + 1}" for i in range(len(labels), n_tasks)]


def metric_to_raw_column(selection_metric: str) -> str:
    """
    Map a summary-table metric name to the corresponding raw-run column.

    Summary files store aggregate columns such as ``test_rmse_mean`` while raw
    run files usually store the per-seed value as ``test_rmse``. This helper
    strips the aggregate suffix before raw runs are sorted within a combo.
    """
    col = selection_metric
    if col.endswith("_mean"):
        col = col[:-5]
    return col


def read_raw_run_rows(raw_runs_dir: Path) -> pd.DataFrame:
    """
    Read the first row from each raw-run CSV into one DataFrame.

    Each raw-run CSV is expected to describe a single completed experiment.
    Empty or unreadable files are skipped so one bad artifact does not prevent
    selection from considering the remaining runs.
    """
    rows = []
    for path in sorted(raw_runs_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        rows.append(df.iloc[[0]].copy())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def candidate_run_rows(
    dataset: str,
    comparisons_root: Path,
    selection_metric: str,
    seed: Optional[int],
) -> pd.DataFrame:
    """
    Return ALE--Frechet raw-run rows ordered by model-selection priority.

    The summary table ranks hyperparameter combinations by ``selection_metric``.
    For each ranked combo, the matching per-seed raw runs are appended and
    optionally sorted by the corresponding raw metric. The result is an ordered
    candidate list that downstream loaders can scan until they find a tracker
    containing the requested saved tensor.
    """
    dataset_dir = comparisons_root / dataset
    summary_path = dataset_dir / "results_mean_std.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise ValueError(f"Summary file is empty: {summary_path}")

    work = summary[summary["method"].astype(str).str.lower() == "ale_frechet"].copy()
    if work.empty:
        raise ValueError(f"No ale_frechet rows found in {summary_path}")
    if selection_metric not in work.columns:
        raise KeyError(f"Missing selection metric '{selection_metric}' in {summary_path}")

    # Stable sorting keeps the summary-table order deterministic when two
    # hyperparameter combinations have the same selection metric.
    work = work.sort_values(selection_metric, ascending=True, kind="mergesort")
    raw = read_raw_run_rows(dataset_dir / "raw_runs")
    if raw.empty:
        raise FileNotFoundError(f"No raw run CSV files found in {dataset_dir / 'raw_runs'}")

    raw = raw[raw["method"].astype(str).str.lower() == "ale_frechet"].copy()
    if seed is not None:
        raw = raw[raw["seed"].astype(int) == int(seed)].copy()
        if raw.empty:
            raise ValueError(f"No ale_frechet raw run found for dataset={dataset}, seed={seed}")

    # Prefer the raw version of the selected metric; fall back to the generic
    # test-loss column used by older comparison artifacts.
    raw_col = metric_to_raw_column(selection_metric)
    if raw_col not in raw.columns:
        raw_col = "best_test_loss" if "best_test_loss" in raw.columns else None

    ordered_blocks = []
    for _, combo_row in work.iterrows():
        combo_index = int(combo_row["combo_index"])
        combo_token = f"combo_{combo_index:03d}"
        block = raw[raw["run_tag"].astype(str).str.contains(combo_token, regex=False)].copy()
        if block.empty:
            continue
        block["combo_rank_value"] = float(combo_row[selection_metric])
        if raw_col is not None:
            block = block.sort_values(raw_col, ascending=True, kind="mergesort")
        ordered_blocks.append(block)

    if not ordered_blocks:
        raise ValueError(f"No raw runs matched ALE--Frechet combos for dataset={dataset}")

    return pd.concat(ordered_blocks, ignore_index=True)


def load_tracker(run_tag: str, logging_root: Path) -> tuple[Path, dict]:
    """
    Load the saved tracker payload for an ALE--Frechet run.

    Returns both the resolved ``metrics.pth`` path and the deserialized payload
    so reporting code can record exactly which artifact produced each figure.
    """
    metrics_path = logging_root / "ale_frechet" / run_tag / "metrics.pth"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing tracker file: {metrics_path}")
    payload = torch.load(metrics_path, map_location="cpu", weights_only=False)
    return metrics_path, payload


def latest_epoch_tensor(epoch_payload: dict, required_name: str) -> tuple[int, torch.Tensor]:
    """
    Return the latest saved epoch and tensor for a tracked metric.

    Trackers store metric snapshots in dictionaries keyed by epoch. This helper
    selects the maximum epoch key, converts array-like payloads to tensors, and
    normalizes the result to detached CPU ``float`` data for plotting.
    """
    if not isinstance(epoch_payload, dict) or not epoch_payload:
        raise ValueError(f"No saved {required_name} tensors found in tracker.")
    epoch = max(int(e) for e in epoch_payload.keys())
    tensor = epoch_payload[epoch]
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    return epoch, tensor.detach().cpu().float()


def select_tracker_with_tensor(
    dataset: str,
    comparisons_root: Path,
    logging_root: Path,
    selection_metric: str,
    seed: Optional[int],
    tensor_name: str,
) -> tuple[str, Path, dict]:
    """
    Select the highest-ranked run that contains a usable tracked tensor.

    Candidate rows come from :func:`candidate_run_rows`. The first candidate
    whose tracker can be loaded and whose latest ``tensor_name`` snapshot can
    be converted to a tensor is returned. Failures are collected and included
    in the final exception to make missing artifacts diagnosable.
    """
    candidates = candidate_run_rows(
        dataset=dataset,
        comparisons_root=comparisons_root,
        selection_metric=selection_metric,
        seed=seed,
    )

    errors = []
    for _, row in candidates.iterrows():
        run_tag = str(row["run_tag"])
        try:
            metrics_path, payload = load_tracker(run_tag, logging_root)
            latest_epoch_tensor(payload.get(tensor_name, {}), tensor_name)
            return run_tag, metrics_path, payload
        except Exception as exc:
            errors.append(f"{run_tag}: {exc}")

    joined = "\n".join(errors[:10])
    raise RuntimeError(
        f"Could not find a usable saved '{tensor_name}' tensor for dataset={dataset}. "
        f"First failures:\n{joined}"
    )


def choose_ale_features(ale_tensor: torch.Tensor, explicit: Optional[Sequence[int]], top_k: int) -> list[int]:
    """
    Choose ALE latent feature indices for plotting.

    ``ale_tensor`` must have shape ``(tasks, features, intervals, 2)`` where the
    last dimension stores grid locations and ALE values. Explicit feature
    indices are validated and de-duplicated. Without explicit indices, features
    are ranked by the variance of centered ALE values across tasks and
    intervals. Centering removes arbitrary vertical offsets before scoring so
    constant high-offset curves are not selected only because they force a large
    y-axis range.
    """
    if ale_tensor.ndim != 4 or ale_tensor.shape[-1] < 2:
        raise ValueError(f"Expected ALE tensor shape (tasks, features, intervals, 2), got {tuple(ale_tensor.shape)}")

    n_features = int(ale_tensor.shape[1])
    if explicit:
        bad = [idx for idx in explicit if idx < 0 or idx >= n_features]
        if bad:
            raise ValueError(f"ALE feature indices out of range 0..{n_features - 1}: {bad}")
        return list(dict.fromkeys(int(idx) for idx in explicit))

    # Rank by centered ALE-value variance rather than grid variance; the grid
    # coordinate lives in the final-dimension slot 0 and should not affect
    # feature choice.
    values = centered_ale_array(ale_tensor)[..., 1]
    scores = np.nanvar(values, axis=(0, 2))
    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[: max(1, min(int(top_k), n_features))]]


def centered_ale_array(ale_tensor: torch.Tensor) -> np.ndarray:
    """
    Return an ALE array with each task-feature curve vertically centered.

    ALE values can contain arbitrary additive offsets for a task/feature pair.
    Plotting the raw offsets can make the y-axis hundreds of units wide even
    when the curve shape varies only slightly. This helper preserves the grid
    coordinate in the final-dimension slot 0 and subtracts the finite mean of
    each ``(task, feature)`` curve from the ALE-value slot 1.
    """
    if ale_tensor.ndim != 4 or ale_tensor.shape[-1] < 2:
        raise ValueError(f"Expected ALE tensor shape (tasks, features, intervals, 2), got {tuple(ale_tensor.shape)}")

    arr = ale_tensor.detach().cpu().float().numpy().copy()
    values = arr[..., 1]
    finite = np.isfinite(values)
    counts = finite.sum(axis=2, keepdims=True)
    sums = np.where(finite, values, 0.0).sum(axis=2, keepdims=True)
    centers = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    arr[..., 1] = np.where(finite, values - centers, values)
    return arr


def ale_y_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
    """
    Compute readable y-axis limits for centered ALE values.

    The limits use the finite min/max after centering and add a small padding.
    If all finite values are identical, the function expands around that value
    so Matplotlib does not create a visually misleading near-zero-height axis.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0

    lower = float(np.nanmin(finite))
    upper = float(np.nanmax(finite))
    span = upper - lower
    if span <= 1e-12:
        pad = max(abs(lower), 1.0) * 0.5
    else:
        pad = max(span * pad_fraction, 1e-6)
    return lower - pad, upper + pad


def save_ale_curves_csv(path: Path, ale_tensor: torch.Tensor, labels: Sequence[str]) -> None:
    """
    Write all ALE grid/value points to a long-form CSV file.

    The exported schema has one row per task, feature, and ALE interval. This
    makes the data easy to inspect, replot, or import into a statistics package
    without decoding the original tensor layout.
    """
    arr = ale_tensor.numpy()
    rows = []
    for task_idx in range(arr.shape[0]):
        task_label = labels[task_idx] if task_idx < len(labels) else f"T{task_idx + 1}"
        for feature_idx in range(arr.shape[1]):
            for interval_idx in range(arr.shape[2]):
                rows.append({
                    "task_index": task_idx,
                    "task": task_label,
                    "feature_index": feature_idx,
                    "interval": interval_idx,
                    "x": float(arr[task_idx, feature_idx, interval_idx, 0]),
                    "ale": float(arr[task_idx, feature_idx, interval_idx, 1]),
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_ale_curves(
    dataset: str,
    ale_tensor: torch.Tensor,
    labels: Sequence[str],
    feature_indices: Sequence[int],
    out_base: Path,
    dpi: int,
) -> None:
    """
    Plot task-specific ALE curves for selected latent features.

    One subplot is created per feature index. Each task is drawn as a separate
    line using centered ALE values from ``ale_tensor``. Centering keeps the
    y-axis focused on profile shape rather than arbitrary vertical offsets. The
    figure is saved as both PNG and PDF using ``out_base`` as the extensionless
    base path.
    """
    n = len(feature_indices)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    arr = centered_ale_array(ale_tensor)

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, arr.shape[0])))
    for panel_idx, feature_idx in enumerate(feature_indices):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        for task_idx in range(arr.shape[0]):
            label = labels[task_idx] if task_idx < len(labels) else f"T{task_idx + 1}"
            ax.plot(
                arr[task_idx, feature_idx, :, 0],
                arr[task_idx, feature_idx, :, 1],
                lw=1.5,
                color=colors[task_idx % len(colors)],
                label=label,
            )
        ax.set_title(f"Latent feature {feature_idx}")
        ax.set_xlabel("ALE grid")
        ax.set_ylabel("Centered ALE")
        ax.set_ylim(*ale_y_limits(arr[:, feature_idx, :, 1]))
        ax.grid(alpha=0.25, linewidth=0.5)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    if len(handles) <= 12:
        fig.legend(handles, legend_labels, loc="upper center", ncol=min(len(handles), 5), frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        fig.tight_layout()

    fig.suptitle(f"{dataset.capitalize()} ALE curves", y=0.995, fontsize=12)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def aggregate_similarity(sim_tensor: torch.Tensor, mode: str) -> np.ndarray:
    """
    Convert a saved similarity tensor into a task-by-task matrix.

    Trackers may save either a two-dimensional matrix ``(tasks, tasks)`` or a
    three-dimensional tensor ``(tasks, tasks, features)``. For the latter,
    ``mode`` controls how feature-specific similarities are collapsed. The
    diagonal is reset to a self-similarity value suitable for heatmaps.
    """
    if sim_tensor.ndim == 2:
        mat = sim_tensor.numpy()
    elif sim_tensor.ndim == 3:
        arr = sim_tensor.numpy()
        if mode == "mean":
            mat = np.nanmean(arr, axis=2)
        elif mode == "sum":
            mat = np.nansum(arr, axis=2)
        elif mode == "max":
            mat = np.nanmax(arr, axis=2)
        else:
            raise ValueError(f"Unsupported similarity aggregation: {mode}")
    else:
        raise ValueError(f"Expected similarity tensor shape (tasks, tasks) or (tasks, tasks, features), got {tuple(sim_tensor.shape)}")

    mat = np.asarray(mat, dtype=float)
    if mat.shape[0] == mat.shape[1]:
        finite = mat[np.isfinite(mat)]
        # If similarities are normalized, self-similarity is 1.0. Otherwise use
        # the observed maximum so the diagonal does not compress the colorbar.
        diag_value = 1.0 if finite.size == 0 or np.nanmax(finite) <= 1.0 else float(np.nanmax(finite))
        np.fill_diagonal(mat, diag_value)
    return mat


def save_similarity_csv(path: Path, matrix: np.ndarray, labels: Sequence[str]) -> None:
    """
    Write a labeled task-by-task similarity matrix to CSV.

    Missing labels are filled with generic task identifiers so the output
    matrix always has matching row and column names.
    """
    n = matrix.shape[0]
    use_labels = list(labels[:n])
    if len(use_labels) < n:
        use_labels.extend(f"T{i + 1}" for i in range(len(use_labels), n))
    df = pd.DataFrame(matrix, index=use_labels, columns=use_labels)
    df.index.name = "task"
    df.to_csv(path)


def plot_similarity_heatmap(
    dataset: str,
    matrix: np.ndarray,
    labels: Sequence[str],
    out_base: Path,
    dpi: int,
) -> None:
    """
    Plot and save a labeled ALE--Frechet task-similarity heatmap.

    Small matrices show every task label. Larger matrices use sparse numeric
    ticks to keep the figure legible while preserving the full matrix values in
    the accompanying CSV.
    """
    n = matrix.shape[0]
    fig_size = max(5.0, min(12.0, 0.16 * n + 4.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap="viridis", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix), aspect="auto")

    if n <= 40:
        tick_labels = list(labels[:n])
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)
    else:
        step = max(1, int(np.ceil(n / 20)))
        ticks = np.arange(0, n, step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(i) for i in ticks], rotation=90, fontsize=7)
        ax.set_yticklabels([str(i) for i in ticks], fontsize=7)

    ax.set_title(f"{dataset.capitalize()} ALE--Frechet similarity")
    ax.set_xlabel("Task")
    ax.set_ylabel("Task")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Similarity")
    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def build_ale_output(
    dataset: str,
    args: argparse.Namespace,
) -> dict:
    """
    Build ALE curve figure and CSV outputs for one synthetic dataset.

    The returned dictionary is one row of the final report CSV. It records the
    selected run, source tracker, tensor shape, plotted features, and generated
    output paths.
    """
    run_tag, metrics_path, payload = select_tracker_with_tensor(
        dataset=dataset,
        comparisons_root=args.comparisons_root,
        logging_root=args.logging_root,
        selection_metric=args.selection_metric,
        seed=args.seed,
        tensor_name="ale",
    )
    epoch, ale_tensor = latest_epoch_tensor(payload["ale"], "ale")
    labels = task_labels(dataset, args.data_root, int(ale_tensor.shape[0]))
    features = choose_ale_features(ale_tensor, args.ale_feature_indices, args.ale_top_features)

    out_base = args.out_dir / f"{dataset}_ale_curves"
    plot_ale_curves(dataset, ale_tensor, labels, features, out_base, args.dpi)
    save_ale_curves_csv(args.out_dir / f"{dataset}_ale_curves.csv", ale_tensor, labels)

    return {
        "kind": "ale",
        "dataset": dataset,
        "run_tag": run_tag,
        "metrics_path": str(metrics_path),
        "epoch": epoch,
        "shape": tuple(ale_tensor.shape),
        "features_plotted": ",".join(str(i) for i in features),
        "png": str(out_base.with_suffix(".png")),
        "pdf": str(out_base.with_suffix(".pdf")),
        "csv": str(args.out_dir / f"{dataset}_ale_curves.csv"),
    }


def build_heatmap_output(
    dataset: str,
    args: argparse.Namespace,
) -> dict:
    """
    Build similarity heatmap figure and CSV outputs for one dataset.

    The similarity tensor is loaded from the selected tracker, optionally
    truncated to ``--max-heatmap-tasks``, rendered as a heatmap, exported as a
    labeled CSV, and summarized as one report row.
    """
    run_tag, metrics_path, payload = select_tracker_with_tensor(
        dataset=dataset,
        comparisons_root=args.comparisons_root,
        logging_root=args.logging_root,
        selection_metric=args.selection_metric,
        seed=args.seed,
        tensor_name="similarity",
    )
    epoch, sim_tensor = latest_epoch_tensor(payload["similarity"], "similarity")
    matrix = aggregate_similarity(sim_tensor, args.similarity_aggregation)

    if args.max_heatmap_tasks is not None:
        max_tasks = max(1, int(args.max_heatmap_tasks))
        matrix = matrix[:max_tasks, :max_tasks]

    labels = task_labels(dataset, args.data_root, matrix.shape[0])
    out_base = args.out_dir / f"{dataset}_similarity_heatmap"
    plot_similarity_heatmap(dataset, matrix, labels, out_base, args.dpi)
    save_similarity_csv(args.out_dir / f"{dataset}_similarity_heatmap.csv", matrix, labels)

    return {
        "kind": "similarity",
        "dataset": dataset,
        "run_tag": run_tag,
        "metrics_path": str(metrics_path),
        "epoch": epoch,
        "shape": tuple(sim_tensor.shape),
        "aggregation": args.similarity_aggregation,
        "png": str(out_base.with_suffix(".png")),
        "pdf": str(out_base.with_suffix(".pdf")),
        "csv": str(args.out_dir / f"{dataset}_similarity_heatmap.csv"),
    }


def main() -> None:
    """
    Run the figure-building workflow and write a report of generated files.

    This is intentionally thin: argument parsing, output-directory creation,
    dataset loops, and final reporting live here, while all data loading and
    plotting details remain in testable helper functions above.
    """
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for dataset in args.ale_datasets:
        records.append(build_ale_output(dataset, args))

    for dataset in args.heatmap_datasets:
        records.append(build_heatmap_output(dataset, args))

    report = pd.DataFrame(records)
    report_path = args.out_dir / "ale_similarity_figure_report.csv"
    report.to_csv(report_path, index=False)

    print("Generated ALE/similarity figures")
    print(f"Output dir: {args.out_dir}")
    print(f"Report    : {report_path}")
    for record in records:
        print(f"- {record['dataset']} {record['kind']}: {record['png']}")


if __name__ == "__main__":
    main()
