#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

LOWER_IS_BETTER_TOKENS = ("loss", "mse", "mae", "rmse", "mape", "smape", "error")
HIGHER_IS_BETTER_TOKENS = ("r2", "acc", "accuracy", "auc", "f1", "precision", "recall")


def metric_direction(metric_name: str, override: str = "auto") -> str:
    if override in {"lower", "higher"}:
        return override
    metric = str(metric_name).strip().lower()
    if any(tok in metric for tok in LOWER_IS_BETTER_TOKENS):
        return "lower"
    if any(tok in metric for tok in HIGHER_IS_BETTER_TOKENS):
        return "higher"
    return "lower"



def minmax_score(series: pd.Series, lower_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if valid.empty:
        return out
    mn = valid.min()
    mx = valid.max()
    if np.isclose(mx, mn):
        out.loc[valid.index] = 1.0
        return out
    if lower_is_better:
        out.loc[valid.index] = (mx - valid) / (mx - mn)
    else:
        out.loc[valid.index] = (valid - mn) / (mx - mn)
    return out



def z_score_rank(series: pd.Series, lower_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if valid.empty:
        return out
    std = valid.std(ddof=0)
    if np.isclose(std, 0.0):
        out.loc[valid.index] = 0.0
        return out
    z = (valid - valid.mean()) / std
    out.loc[valid.index] = -z if lower_is_better else z
    return out



def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate a method ranking table from results_task_behavior.csv by "
            "balancing average performance and cross-task stability."
        )
    )
    ap.add_argument("--input", required=True, help="Path to results_task_behavior.csv")
    ap.add_argument("--output", default=None, help="Output CSV path for the ranking table")
    ap.add_argument("--split", default="test", help="Split to use, e.g. test/val/train")
    ap.add_argument("--metric", required=True, help="Metric name, e.g. loss, mse, mae, r2")
    ap.add_argument(
        "--performance-col",
        default="run_task_mean_mean",
        help="Column representing average performance across tasks (default: run_task_mean_mean)",
    )
    ap.add_argument(
        "--stability-col",
        default="run_task_std_mean",
        help="Column representing cross-task dispersion/stability (default: run_task_std_mean)",
    )
    ap.add_argument(
        "--secondary-stability-col",
        default="run_task_gap_mean",
        help="Optional second stability column such as run_task_gap_mean; use '' to disable",
    )
    ap.add_argument(
        "--group-cols",
        default="dataset,method,combo_index,combo_tag",
        help="Columns that define one compared configuration",
    )
    ap.add_argument(
        "--method-col",
        default="method",
        help="Column name used as the method identifier",
    )
    ap.add_argument(
        "--direction",
        choices=["auto", "lower", "higher"],
        default="auto",
        help="Whether lower or higher values are better for the chosen metric",
    )
    ap.add_argument(
        "--performance-weight",
        type=float,
        default=0.7,
        help="Weight for the average-performance score",
    )
    ap.add_argument(
        "--stability-weight",
        type=float,
        default=0.3,
        help="Total weight for the stability score(s)",
    )
    ap.add_argument(
        "--stability-mix",
        type=float,
        default=0.7,
        help=(
            "If two stability columns are used, fraction of stability-weight assigned to "
            "the primary stability column; the rest goes to the secondary one"
        ),
    )
    ap.add_argument(
        "--score-method",
        choices=["minmax", "zscore"],
        default="minmax",
        help="Normalization used before combining criteria",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optionally keep only the top-k ranked rows",
    )
    return ap



def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name("method_ranking.csv")

    df = pd.read_csv(input_path)
    if df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        print(f"Input is empty. Wrote empty ranking file to {output_path}")
        return

    required = {"split", "metric", args.performance_col, args.stability_col}
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    required.update(group_cols)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {input_path}: {missing}. Available columns: {list(df.columns)}"
        )

    metric_name = str(args.metric).strip().lower()
    sub = df.copy()
    sub["metric"] = sub["metric"].astype(str).str.strip().str.lower()
    sub["split"] = sub["split"].astype(str).str.strip().str.lower()
    sub = sub[(sub["split"] == args.split.lower()) & (sub["metric"] == metric_name)].copy()

    if sub.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        print(
            f"No rows found for split={args.split!r}, metric={metric_name!r}. "
            f"Wrote empty ranking file to {output_path}"
        )
        return

    direction = metric_direction(metric_name, args.direction)
    lower_is_better_perf = direction == "lower"
    lower_is_better_stability = True

    perf_weight = float(args.performance_weight)
    stab_weight = float(args.stability_weight)
    if perf_weight < 0 or stab_weight < 0:
        raise ValueError("Weights must be non-negative.")
    if np.isclose(perf_weight + stab_weight, 0.0):
        raise ValueError("At least one weight must be positive.")

    perf_weight, stab_weight = perf_weight / (perf_weight + stab_weight), stab_weight / (perf_weight + stab_weight)

    secondary_col = args.secondary_stability_col.strip()
    use_secondary = bool(secondary_col)
    if use_secondary and secondary_col not in sub.columns:
        raise ValueError(f"secondary stability column not found: {secondary_col}")

    score_fn = minmax_score if args.score_method == "minmax" else z_score_rank

    sub = sub.copy()
    sub["performance_score"] = score_fn(sub[args.performance_col], lower_is_better=lower_is_better_perf)
    sub["stability_score"] = score_fn(sub[args.stability_col], lower_is_better=lower_is_better_stability)

    if use_secondary:
        mix = float(args.stability_mix)
        if not (0.0 <= mix <= 1.0):
            raise ValueError("stability-mix must be between 0 and 1")
        sub["stability_score_secondary"] = score_fn(sub[secondary_col], lower_is_better=lower_is_better_stability)
        sub["stability_score_combined"] = (
            mix * sub["stability_score"] + (1.0 - mix) * sub["stability_score_secondary"]
        )
    else:
        sub["stability_score_secondary"] = np.nan
        sub["stability_score_combined"] = sub["stability_score"]

    sub["ranking_score"] = (
        perf_weight * sub["performance_score"] + stab_weight * sub["stability_score_combined"]
    )

    ascending_rank = False
    sub["rank"] = sub["ranking_score"].rank(method="dense", ascending=ascending_rank).astype("Int64")

    order_cols: List[str] = ["rank", "ranking_score", args.performance_col, args.stability_col]
    if use_secondary:
        order_cols.append(secondary_col)

    sub = sub.sort_values(by=["rank", "ranking_score"], ascending=[True, False]).reset_index(drop=True)

    if args.top_k is not None:
        sub = sub.head(int(args.top_k)).copy()

    # Add a compact interpretation helper
    sub["ranking_note"] = (
        "perf=" + sub[args.performance_col].round(6).astype(str)
        + ", stab=" + sub[args.stability_col].round(6).astype(str)
        + (", gap=" + sub[secondary_col].round(6).astype(str) if use_secondary else "")
    )

    preferred_cols = [
        "rank",
        *[c for c in group_cols if c in sub.columns],
        "split",
        "metric",
        args.performance_col,
        args.stability_col,
    ]
    if use_secondary:
        preferred_cols.append(secondary_col)
    preferred_cols += [
        "performance_score",
        "stability_score",
        "stability_score_secondary",
        "stability_score_combined",
        "ranking_score",
        "ranking_note",
    ]
    remaining = [c for c in sub.columns if c not in preferred_cols]
    sub = sub[preferred_cols + remaining]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)

    print(f"Saved ranking table to {output_path}")
    print(f"Rows: {len(sub)}")
    print(f"Metric direction: {direction}")
    print(f"Performance column: {args.performance_col}")
    print(f"Stability column: {args.stability_col}")
    if use_secondary:
        print(f"Secondary stability column: {secondary_col}")
    print(f"Weights -> performance: {perf_weight:.3f}, stability: {stab_weight:.3f}")


if __name__ == "__main__":
    main()
