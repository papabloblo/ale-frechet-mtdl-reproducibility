#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DEFAULT_METHOD_ORDER = [
    "ale_frechet",
    "hard",
    "soft",
    "mmoe",
    "crossstitch",
    "ple",
    "mtan",
]

DISPLAY_NAMES = {
    "ale_frechet": "ALE–Fréchet",
    "hard": "Hard",
    "soft": "Soft",
    "mmoe": "MMoE",
    "crossstitch": "Cross-stitch",
    "ple": "PLE",
    "mtan": "MTAN",
}

SPLIT_ALIASES = {
    "val": "validation",
    "validation": "validation",
    "valid": "validation",
    "train": "train",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a learning-curves figure for the compared methods using the saved "
            "tracker files (metrics.pth). The script selects one configuration per "
            "method from results_mean_std.csv and aggregates seed-wise curves."
        )
    )
    ap.add_argument("--dataset", required=True, help="Dataset name, e.g. electricity")
    ap.add_argument(
        "--comparisons-root",
        default="results/comparisons",
        help="Root directory containing per-dataset comparison folders.",
    )
    ap.add_argument(
        "--logging-root",
        default="results",
        help=(
            "Root directory used by training for tracker outputs. The script expects "
            "files like <logging-root>/<method>/<run_tag>/metrics.pth"
        ),
    )
    ap.add_argument(
        "--out-dir",
        default="reports/figures/learning_curves",
        help="Directory where the figure and helper files will be written.",
    )
    ap.add_argument(
        "--selection-metric",
        default="test_rmse_mean",
        help=(
            "Column in results_mean_std.csv used to choose the best configuration per method. "
            "Lower is assumed to be better."
        ),
    )
    ap.add_argument(
        "--metric",
        default="RMSE",
        help="Metric to plot from tracker files. Use LOSS or one of the saved error names, e.g. RMSE, MAE, MAPE.",
    )
    ap.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "val", "test"],
        help="Tracker split to plot.",
    )
    ap.add_argument(
        "--include-train",
        action="store_true",
        help="Overlay the train curve of the same metric as a dashed line for each method.",
    )
    ap.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional subset of methods to include.",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional hard cap on plotted epochs.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution.",
    )
    return ap.parse_args()


def normalize_split(split: str) -> str:
    out = SPLIT_ALIASES.get(split.lower())
    if out is None:
        raise ValueError(f"Unsupported split: {split}")
    return out


def normalize_method_order(methods: Iterable[str]) -> List[str]:
    methods = [m.lower() for m in methods]
    order_map = {m: i for i, m in enumerate(DEFAULT_METHOD_ORDER)}
    return sorted(methods, key=lambda m: (order_map.get(m, 10_000), m))


def find_metric_tensor(metrics_payload: Dict, split: str, metric: str) -> torch.Tensor:
    """Return the saved tensor for one split/metric.

    The in-memory tracker uses:
        tracker.track[split]["metrics"]

    but ``metrics.pth`` is saved with a flatter structure:
        saved_metrics[split] = tracker.track[split]["metrics"].for_save()

    so the saved payload usually does *not* contain an intermediate ``"metrics"`` key.
    To remain robust, this helper accepts both layouts.
    """
    if split not in metrics_payload:
        available = ", ".join(sorted(metrics_payload.keys()))
        raise KeyError(f"Split '{split}' not found in metrics payload. Available splits: {available}")

    split_payload = metrics_payload[split]
    if isinstance(split_payload, dict) and "metrics" in split_payload:
        split_payload = split_payload["metrics"]

    metric_u = metric.upper()

    if metric_u == "LOSS":
        if "loss_per_task" not in split_payload:
            available = ", ".join(sorted(split_payload.keys()))
            raise KeyError(
                f"Saved payload for split '{split}' does not contain 'loss_per_task'. "
                f"Available keys: {available}"
            )
        tensor = split_payload["loss_per_task"]
    else:
        errors = split_payload.get("errors")
        if errors is None:
            available = ", ".join(sorted(split_payload.keys()))
            raise KeyError(
                f"Saved payload for split '{split}' does not contain an 'errors' dictionary. "
                f"Available keys: {available}"
            )
        if metric_u not in errors:
            available = ", ".join(sorted(errors.keys()))
            raise KeyError(f"Metric '{metric}' not found. Available tracked errors: {available}")
        tensor = errors[metric_u]

    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.detach().cpu().float()


def reduce_metric_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Reduce a tracker tensor to one scalar per epoch.

    The saved tracker keeps the epoch axis first, then one or more extra axes
    depending on what each batch metric returns. Typical examples are:
      - [epochs]                         -> already one value per epoch
      - [epochs, batches]               -> batch-wise scalar metric
      - [epochs, batches, tasks]        -> per-task metric per batch
      - [epochs, batches, tasks, out]   -> same, with an output/channel dim

    For learning curves we want one scalar per epoch, so we average over every
    axis except the first one.
    """
    arr = tensor.detach().cpu().float().numpy()

    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr

    reduce_axes = tuple(range(1, arr.ndim))
    return arr.mean(axis=reduce_axes)


def nanmean_nanstd(curves: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not curves:
        raise ValueError("No curves to aggregate.")
    max_len = max(len(c) for c in curves)
    mat = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        mat[i, : len(curve)] = curve
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0, ddof=1 if len(curves) > 1 else 0)
    n = np.sum(~np.isnan(mat), axis=0)
    return mean, std, n


def load_seed_curve(metrics_path: Path, split: str, metric: str) -> np.ndarray:
    payload = torch.load(metrics_path, map_location="cpu", weights_only=False)
    tensor = find_metric_tensor(payload, split=split, metric=metric)
    return reduce_metric_tensor(tensor)


def choose_best_rows(results_mean_std: pd.DataFrame, selection_metric: str, methods: Optional[Sequence[str]]) -> pd.DataFrame:
    df = results_mean_std.copy()
    df["method"] = df["method"].astype(str).str.lower()

    if methods:
        keep = {m.lower() for m in methods}
        df = df[df["method"].isin(keep)].copy()

    if selection_metric not in df.columns:
        raise KeyError(
            f"Selection metric '{selection_metric}' is not present in results_mean_std.csv. "
            f"Available columns include: {', '.join(df.columns[:20])}..."
        )

    if df.empty:
        raise ValueError("No rows left after filtering methods.")

    idx = df.groupby("method", dropna=False)[selection_metric].idxmin()
    best = df.loc[idx].copy()
    best = best.sort_values(by="method", key=lambda s: s.map({m: i for i, m in enumerate(DEFAULT_METHOD_ORDER)}).fillna(10_000))
    return best.reset_index(drop=True)


def build_run_table(results_all: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = results_all.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["method"] = df["method"].astype(str).str.lower()
    df["run_tag"] = df["run_tag"].astype(str)
    return df[df["dataset"] == dataset].copy()


def tracker_path(logging_root: Path, method: str, run_tag: str) -> Path:
    return logging_root / method / run_tag / "metrics.pth"


def format_method_label(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def save_caption(caption_path: Path, dataset: str, split: str, metric: str, selection_metric: str, included_methods: List[str]) -> None:
    split_name = "validation" if split == "validation" else split
    methods_txt = ", ".join(format_method_label(m) for m in included_methods)
    caption = (
        f"Learning curves for {dataset}. The figure reports the epoch-wise {split_name} {metric.upper()} "
        f"for the best configuration of each method, where the selected configuration is the one that minimizes "
        f"{selection_metric} in the sweep summary. Solid lines denote the mean across seeds and shaded bands denote "
        f"one standard deviation. Included methods: {methods_txt}."
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    split = normalize_split(args.split)
    metric = args.metric.upper()

    dataset_dir = Path(args.comparisons_root) / dataset
    results_mean_std_path = dataset_dir / "results_mean_std.csv"
    results_all_path = dataset_dir / "results_all.csv"

    if not results_mean_std_path.exists():
        raise FileNotFoundError(f"Missing file: {results_mean_std_path}")
    if not results_all_path.exists():
        raise FileNotFoundError(f"Missing file: {results_all_path}")

    results_mean_std = pd.read_csv(results_mean_std_path)
    results_all = pd.read_csv(results_all_path)

    best_rows = choose_best_rows(results_mean_std, args.selection_metric, args.methods)
    runs_df = build_run_table(results_all, dataset=dataset)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging_root = Path(args.logging_root)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    included_methods: List[str] = []
    report_rows: List[Dict[str, object]] = []

    for _, best_row in best_rows.iterrows():
        method = str(best_row["method"]).lower()
        combo_index = int(best_row["combo_index"])

        seed_runs = runs_df[
            (runs_df["method"] == method)
            & (runs_df["combo_index"] == combo_index)
        ].copy()

        if seed_runs.empty:
            print(f"[WARN] No seed runs found in results_all.csv for method={method}, combo_index={combo_index}.")
            continue

        curves: List[np.ndarray] = []
        train_curves: List[np.ndarray] = []
        used_run_tags: List[str] = []
        missing_paths: List[str] = []

        for _, run in seed_runs.sort_values(by=[c for c in ["seed", "run_tag"] if c in seed_runs.columns]).iterrows():
            run_tag = str(run["run_tag"])
            mpath = tracker_path(logging_root=logging_root, method=method, run_tag=run_tag)
            if not mpath.exists():
                missing_paths.append(str(mpath))
                continue

            try:
                curve = load_seed_curve(mpath, split=split, metric=metric)
                if args.max_epochs is not None:
                    curve = curve[: args.max_epochs]
                curves.append(curve)
                used_run_tags.append(run_tag)

                if args.include_train:
                    train_curve = load_seed_curve(mpath, split="train", metric=metric)
                    if args.max_epochs is not None:
                        train_curve = train_curve[: args.max_epochs]
                    train_curves.append(train_curve)
            except Exception as exc:
                print(f"[WARN] Could not read {mpath}: {exc}")

        if not curves:
            print(f"[WARN] Method {method} skipped because no tracker files could be loaded.")
            continue

        mean_curve, std_curve, n_curve = nanmean_nanstd(curves)
        epochs = np.arange(1, len(mean_curve) + 1)

        line, = ax.plot(epochs, mean_curve, label=format_method_label(method), linewidth=2.0)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.18)

        if args.include_train and train_curves:
            train_mean, _, _ = nanmean_nanstd(train_curves)
            train_epochs = np.arange(1, len(train_mean) + 1)
            ax.plot(
                train_epochs,
                train_mean,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                color=line.get_color(),
                label=f"{format_method_label(method)} (train)",
            )

        included_methods.append(method)
        report_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "display_name": format_method_label(method),
                "combo_index": combo_index,
                "combo_tag": best_row.get("combo_tag", None),
                "selection_metric": args.selection_metric,
                "selection_metric_value": best_row.get(args.selection_metric, np.nan),
                "n_seed_runs_in_results_all": int(len(seed_runs)),
                "n_seed_runs_loaded": int(len(curves)),
                "max_epoch_plotted": int(len(mean_curve)),
                "run_tags_loaded": " | ".join(used_run_tags),
                "missing_tracker_paths": " | ".join(missing_paths),
                "last_epoch_n": int(n_curve[-1]),
                "last_epoch_mean": float(mean_curve[-1]),
                "last_epoch_std": float(std_curve[-1]),
            }
        )

    if not included_methods:
        raise RuntimeError(
            "No method could be plotted. Check that results_all.csv exists and that tracker files are available "
            "under <logging-root>/<method>/<run_tag>/metrics.pth."
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{split.capitalize()} {metric}")
    ax.set_title(f"{dataset}: {split.capitalize()} {metric} learning curves")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, ncol=2 if len(included_methods) > 4 else 1)
    fig.tight_layout()

    stem = f"learning_curves__{dataset}__{split}__{metric.lower()}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    report_path = out_dir / f"{stem}__report.csv"
    caption_path = out_dir / f"{stem}__caption.txt"

    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    save_caption(
        caption_path,
        dataset=dataset,
        split=split,
        metric=metric,
        selection_metric=args.selection_metric,
        included_methods=normalize_method_order(included_methods),
    )

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved caption: {caption_path}")


if __name__ == "__main__":
    main()
