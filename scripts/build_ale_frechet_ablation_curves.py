#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
            "Build ALE–Fréchet ablation learning curves from saved tracker files. "
            "Each plotted line corresponds to one ALE–Fréchet configuration aggregated across seeds."
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
        help="Root directory used by training for tracker outputs.",
    )
    ap.add_argument(
        "--out-dir",
        default="reports/figures/ablation",
        help="Directory where the figure and helper files will be written.",
    )
    ap.add_argument(
        "--selection-metric",
        default="test_rmse_mean",
        help="Column used to rank ALE–Fréchet configurations. Lower is assumed to be better.",
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
        help="Overlay the train curve of the same metric as a dashed line for each configuration.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of ALE–Fréchet configurations to plot, ranked by selection metric.",
    )
    ap.add_argument(
        "--combo-indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit combo_index values to plot. Overrides --top-k.",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional hard cap on plotted epochs.",
    )
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return ap.parse_args()


def normalize_split(split: str) -> str:
    out = SPLIT_ALIASES.get(split.lower())
    if out is None:
        raise ValueError(f"Unsupported split: {split}")
    return out


def find_metric_tensor(metrics_payload: Dict, split: str, metric: str) -> torch.Tensor:
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
                f"Saved payload for split '{split}' does not contain 'loss_per_task'. Available keys: {available}"
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
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=tuple(range(1, arr.ndim)))


def nanmean_nanstd(curves: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def tracker_path(logging_root: Path, method: str, run_tag: str) -> Path:
    return logging_root / method / run_tag / "metrics.pth"


def format_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return str(value)


def build_config_label(row: pd.Series) -> str:
    lr = format_value(row.get("optim_lr", row.get("lr", np.nan)))
    l2 = format_value(row.get("regularization_l2", row.get("l2_penalty", row.get("l2", np.nan))))
    keep = format_value(row.get("similarity_keep_epochs", row.get("keep_epochs", np.nan)))
    update = format_value(row.get("similarity_update_every", row.get("update_every", np.nan)))
    return f"c{int(row['combo_index']):03d}: lr={lr}, l2={l2}, keep={keep}, upd={update}"


def select_ale_rows(df: pd.DataFrame, selection_metric: str, combo_indices: Optional[Sequence[int]], top_k: int) -> pd.DataFrame:
    out = df.copy()
    out["method"] = out["method"].astype(str).str.lower()
    out = out[out["method"] == "ale_frechet"].copy()

    if out.empty:
        raise ValueError("No ale_frechet rows found in results_mean_std.csv.")
    if selection_metric not in out.columns:
        raise KeyError(
            f"Selection metric '{selection_metric}' is not present in results_mean_std.csv. "
            f"Available columns include: {', '.join(out.columns[:20])}..."
        )

    out = out.sort_values(selection_metric, ascending=True)

    if combo_indices:
        wanted = set(combo_indices)
        out = out[out["combo_index"].isin(wanted)].copy()
        missing = wanted.difference(set(out["combo_index"].tolist()))
        if missing:
            raise ValueError(f"Requested combo_index values not found for ale_frechet: {sorted(missing)}")
        order = {combo: i for i, combo in enumerate(combo_indices)}
        out = out.sort_values(by="combo_index", key=lambda s: s.map(order))
    else:
        out = out.head(top_k).copy()

    out["config_label"] = out.apply(build_config_label, axis=1)
    return out.reset_index(drop=True)


def save_caption(caption_path: Path, dataset: str, split: str, metric: str, selection_metric: str, n_configs: int) -> None:
    caption = (
        f"Learning-curve ablation for ALE--Fr\'echet on {dataset}. Each line reports the epoch-wise {split} "
        f"{metric.upper()} of one ALE--Fr\'echet configuration, aggregated across seeds. Configurations are ranked "
        f"by {selection_metric}, and the figure includes the top {n_configs} settings. Solid lines denote the mean "
        f"across seeds and shaded bands denote one standard deviation."
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
    results_all["dataset"] = results_all["dataset"].astype(str)
    results_all["method"] = results_all["method"].astype(str).str.lower()
    results_all["run_tag"] = results_all["run_tag"].astype(str)

    selected = select_ale_rows(
        results_mean_std,
        selection_metric=args.selection_metric,
        combo_indices=args.combo_indices,
        top_k=args.top_k,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging_root = Path(args.logging_root)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    report_rows: List[Dict[str, object]] = []

    for _, row in selected.iterrows():
        combo_index = int(row["combo_index"])
        seed_runs = results_all[
            (results_all["dataset"] == dataset)
            & (results_all["method"] == "ale_frechet")
            & (results_all["combo_index"] == combo_index)
        ].copy()

        if seed_runs.empty:
            print(f"[WARN] No seed runs found for ALE–Fréchet combo_index={combo_index}.")
            continue

        curves: List[np.ndarray] = []
        train_curves: List[np.ndarray] = []
        used_run_tags: List[str] = []
        missing_paths: List[str] = []

        sort_cols = [c for c in ["seed", "run_tag"] if c in seed_runs.columns]
        for _, run in seed_runs.sort_values(by=sort_cols).iterrows():
            run_tag = str(run["run_tag"])
            mpath = tracker_path(logging_root=logging_root, method="ale_frechet", run_tag=run_tag)
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
            print(f"[WARN] ALE–Fréchet combo_index={combo_index} skipped because no tracker files could be loaded.")
            continue

        mean_curve, std_curve, n_curve = nanmean_nanstd(curves)
        epochs = np.arange(1, len(mean_curve) + 1)
        label = str(row["config_label"])

        line, = ax.plot(epochs, mean_curve, linewidth=2.0, label=label)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.18)

        if args.include_train and train_curves:
            train_mean, _, _ = nanmean_nanstd(train_curves)
            train_epochs = np.arange(1, len(train_mean) + 1)
            ax.plot(
                train_epochs,
                train_mean,
                linestyle="--",
                linewidth=1.1,
                alpha=0.9,
                color=line.get_color(),
            )

        report_rows.append(
            {
                "dataset": dataset,
                "method": "ale_frechet",
                "combo_index": combo_index,
                "config_label": label,
                "selection_metric": args.selection_metric,
                "selection_metric_value": row.get(args.selection_metric, np.nan),
                "n_seed_runs_in_results_all": int(len(seed_runs)),
                "n_seed_runs_loaded": int(len(curves)),
                "max_epoch_plotted": int(len(mean_curve)),
                "run_tags_loaded": " | ".join(used_run_tags),
                "missing_tracker_paths": " | ".join(missing_paths),
                "last_epoch_n": int(n_curve[-1]),
                "last_epoch_mean": float(mean_curve[-1]),
                "last_epoch_std": float(std_curve[-1]),
                "optim_lr": row.get("optim_lr", np.nan),
                "regularization_l2": row.get("regularization_l2", np.nan),
                "similarity_keep_epochs": row.get("similarity_keep_epochs", np.nan),
                "similarity_update_every": row.get("similarity_update_every", np.nan),
            }
        )

    if not report_rows:
        raise RuntimeError(
            "No ALE–Fréchet configuration could be plotted. Check results_all.csv and tracker files under "
            "<logging-root>/ale_frechet/<run_tag>/metrics.pth."
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{split.capitalize()} {metric}")
    ax.set_title(f"{dataset}: ALE–Fréchet ablation learning curves")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()

    stem = f"ale_frechet_ablation__{dataset}__{split}__{metric.lower()}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    report_path = out_dir / f"{stem}__report.csv"
    caption_path = out_dir / f"{stem}__caption.txt"

    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    save_caption(
        caption_path=caption_path,
        dataset=dataset,
        split=split,
        metric=metric,
        selection_metric=args.selection_metric,
        n_configs=len(report_rows),
    )

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved caption: {caption_path}")


if __name__ == "__main__":
    main()
