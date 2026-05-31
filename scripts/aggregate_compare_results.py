#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np


EXCLUDE_JSON = {
    "results_all.json",
    "failures.json",
    "manifest.json",
    "completed_runs.json",
    "results.json",
}

TRACKER_METRIC_CACHE: Dict[Path, Dict[str, Any] | None] = {}


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def finite_numeric_mean(value: Any) -> float:
    if value is None:
        return np.nan

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if hasattr(value, "numel") and value.numel() == 1:
            value = value.item()
        elif hasattr(value, "reshape"):
            value = value.reshape(-1).tolist()

    if isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, pd.Series):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]

    normalized_values = []
    for item in values:
        if hasattr(item, "detach") and hasattr(item, "cpu"):
            item = item.detach().cpu()
            item = item.item() if hasattr(item, "numel") and item.numel() == 1 else item
        normalized_values.append(item)

    numeric = pd.to_numeric(pd.Series(normalized_values, dtype="object"), errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return np.nan
    return float(numeric.mean())


def load_tracker_best_model_metrics(r: Dict[str, Any]) -> Dict[str, Any] | None:
    method = r.get("method")
    run_tag = r.get("run_tag")
    if not method or not run_tag:
        return None

    path = Path("results") / str(method) / str(run_tag) / "best_model.pth"
    path = path.resolve()
    if path in TRACKER_METRIC_CACHE:
        return TRACKER_METRIC_CACHE[path]

    if not path.exists():
        TRACKER_METRIC_CACHE[path] = None
        return None

    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        TRACKER_METRIC_CACHE[path] = None
        return None

    if not isinstance(payload, dict):
        TRACKER_METRIC_CACHE[path] = None
        return None

    metrics = {
        "train": payload.get("train_metrics", {}),
        "val": payload.get("val_metrics", {}),
        "test": payload.get("test_metrics", {}),
    }
    TRACKER_METRIC_CACHE[path] = metrics
    return metrics


def record_metric_mean(r: Dict[str, Any], split: str, metric_name: str, metric_val: Any) -> float:
    value = finite_numeric_mean(metric_val)
    if np.isfinite(value):
        return value

    split_mean = r.get(split, {}).get("mean", {})
    if isinstance(split_mean, dict):
        value = finite_numeric_mean(split_mean.get(metric_name))
        if np.isfinite(value):
            return value

    # train_compare_methods stores test RMSE in this historical field name.
    if split == "test" and metric_name.upper() == "RMSE":
        return finite_numeric_mean(r.get("best_test_loss"))

    tracker_metrics = load_tracker_best_model_metrics(r)
    if tracker_metrics:
        value = finite_numeric_mean(tracker_metrics.get(split, {}).get(metric_name))
        if np.isfinite(value):
            return value

    return np.nan


def flatten_result_record(r: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "dataset": r.get("dataset"),
        "method": r.get("method"),
        "seed": r.get("seed"),
        "run_tag": r.get("run_tag"),
        "component_ablation_variant": r.get("component_ablation_variant"),
        "best_epoch": r.get("best_epoch"),
        "best_test_loss": r.get("best_test_loss"),
        "time_train": r.get("time_train"),
        "time_validation": r.get("time_validation"),
        "time_test": r.get("time_test"),
        "time_ale": r.get("time_ale"),
        "time_similarity": r.get("time_similarity"),
        "total_time": r.get("total_time"),
    }

    for split in ("train", "val", "test"):
        #mean_metrics = r.get(split, {}).get("mean", {})
        mean_metrics = r.get(split, {}).get("per_task", {})
        for metric_name, metric_val in mean_metrics.items():
            row[f"{split}_{metric_name.lower()}"] = record_metric_mean(
                r,
                split,
                metric_name,
                metric_val,
            )

    if "test_rmse" not in row:
        fallback_test_rmse = record_metric_mean(r, "test", "RMSE", None)
        if np.isfinite(fallback_test_rmse):
            row["test_rmse"] = fallback_test_rmse

    combo = r.get("sweep_combo", {})
    for k, v in combo.items():
        row[f"sweep__{k.replace('.', '__')}"] = v

    row["combo_index"] = r.get("combo_index")
    row["combo_tag"] = r.get("combo_tag")
    return row


def aggregate_mean_std(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        c for c in df.columns
        if c not in group_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    grouped = df.groupby(group_cols, dropna=False)

    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: val for col, val in zip(group_cols, keys)}
        for c in metric_cols:
            row[f"{c}_mean"] = sub[c].mean()
            row[f"{c}_std"] = sub[c].std(ddof=1) if len(sub) > 1 else 0.0
            row[f"{c}_n"] = int(sub[c].notna().sum())
        rows.append(row)

    return pd.DataFrame(rows)


def load_run_records(run_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in EXCLUDE_JSON:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for rec in payload:
            if isinstance(rec, dict):
                rec = dict(rec)
                rec["__source_file"] = path.name
                records.append(rec)
    return records


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        c for c in [
            "dataset",
            "method",
            "seed",
            "run_tag",
            "combo_index",
            "combo_tag",
        ]
        if c in df.columns
    ]
    if not key_cols:
        return df
    return df.drop_duplicates(subset=key_cols, keep="last").copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Directory with enriched per-run JSON files.")
    ap.add_argument("--out-dir", required=True, help="Directory where results_all/results_mean_std will be written.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_run_records(run_dir)
    (out_dir / "results_all.json").write_text(
        json.dumps(sanitize_for_json(all_results), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if not all_results:
        pd.DataFrame().to_csv(out_dir / "results_all.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "results_mean_std.csv", index=False)
        print(f"No run JSON files found in {run_dir}")
        return

    df = pd.DataFrame([flatten_result_record(r) for r in all_results])
    df = deduplicate(df)
    sort_cols = [c for c in ["dataset", "method", "combo_index", "seed", "run_tag"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)
    df.to_csv(out_dir / "results_all.csv", index=False)

    group_cols = [c for c in ["dataset", "method", "combo_index", "combo_tag"] if c in df.columns]
    group_cols += [c for c in df.columns if c.startswith("sweep__")]

    df_mean_std = aggregate_mean_std(df, group_cols=group_cols)
    if not df_mean_std.empty:
        sort_mean_std_cols = [c for c in ["dataset", "method", "combo_index"] if c in df_mean_std.columns]
        if sort_mean_std_cols:
            df_mean_std = df_mean_std.sort_values(by=sort_mean_std_cols).reset_index(drop=True)
    df_mean_std.to_csv(out_dir / "results_mean_std.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "n_records": int(len(df)),
        "n_group_rows": int(len(df_mean_std)),
    }
    (out_dir / "aggregate_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Saved: {out_dir / 'results_all.json'}")
    print(f"Saved: {out_dir / 'results_all.csv'}")
    print(f"Saved: {out_dir / 'results_mean_std.csv'}")
    print(f"Saved: {out_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
