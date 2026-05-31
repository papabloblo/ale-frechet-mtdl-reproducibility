# download_real.py
# =================
#
# Download and normalize REAL multi-task time-series datasets for the
# ALE–Fréchet MTL experiments repository, and transform them into
# supervised panels with forecast targets, lags, rolling features,
# plus domain-specific contextual features (no leakage).

from __future__ import annotations

import io
import zipfile
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Iterable, List

import json
import tarfile
import gzip
import pickle
import yaml
import random

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Split utility (order-preserving per task)
# -------------------------------------------------------------------------
def temporal_split_per_task(
    df: pd.DataFrame,
    task_col: str = "task",
    time_col: str = "time",
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split each task sequentially into train/val/test based on time order."""
    assert abs(train + val + test - 1.0) < 1e-8
    parts, meta = [], []

    for task, g in df.groupby(task_col, sort=True):
        g = g.sort_values(time_col).reset_index(drop=True)
        n = len(g)
        n_tr = int(n * train)
        n_val = int(n * val)
        n_te = n - n_tr - n_val

        parts += [
            g.iloc[:n_tr].assign(split="train"),
            g.iloc[n_tr:n_tr + n_val].assign(split="val"),
            g.iloc[n_tr + n_val:].assign(split="test"),
        ]
        meta.append({"task": task, "n_total": n, "n_train": n_tr, "n_val": n_val, "n_test": n_te})

    combined = pd.concat(parts, ignore_index=True)
    meta_df = pd.DataFrame(meta)
    return combined, meta_df


# -------------------------------------------------------------------------
# Supervised transformation: target, lags, rolling (no leakage)
# -------------------------------------------------------------------------
def make_supervised_panel(
    df: pd.DataFrame,
    task_col: str = "task",
    time_col: str = "time",
    y_col: str = "y",
    covariates: Optional[List[str]] = None,
    horizon: int = 1,
    y_lags: Optional[List[int]] = None,
    cov_lags: Optional[List[int]] = None,
    roll_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Convert long-form multitask panel into supervised rows with:
      - y_target = y.shift(-horizon)
      - y_lag_L for L in y_lags
      - <cov>_lag_L for covariates and L in cov_lags
      - y_rollmean_W, y_rollstd_W from y.shift(1).rolling(W)  (strictly past)
    All ops applied per task in chronological order; rows with NA are dropped.
    """
    df = df.copy().sort_values([task_col, time_col])

    if covariates is None:
        exclude = {task_col, time_col, y_col, "split"}
        covariates = [c for c in df.columns if c not in exclude]

    y_lags = y_lags or [1]
    cov_lags = cov_lags or []
    roll_windows = roll_windows or []

    def _per_task(g: pd.DataFrame, task_value: object) -> pd.DataFrame:
        g = g.sort_values(time_col).copy()
        if task_col not in g.columns:
            g[task_col] = task_value
        g["y_target"] = g[y_col].shift(-horizon)

        for L in y_lags:
            g[f"y_lag_{L}"] = g[y_col].shift(L)

        for c in covariates:
            if not pd.api.types.is_numeric_dtype(g[c]):
                continue
            for L in cov_lags:
                g[f"{c}_lag_{L}"] = g[c].shift(L)

        if roll_windows:
            y_shift = g[y_col].shift(1)
            for W in roll_windows:
                g[f"y_rollmean_{W}"] = y_shift.rolling(W, min_periods=W).mean()
                g[f"y_rollstd_{W}"] = y_shift.rolling(W, min_periods=W).std()

        need_cols = ["y_target"] + [
            c for c in g.columns
            if c.startswith("y_lag_")
            or c.startswith("y_roll")
            or any(c.endswith(f"_lag_{L}") for L in cov_lags)
        ]
        g = g.dropna(subset=need_cols)
        return g

    parts = []
    for task_value, group in df.groupby(task_col, sort=False):
        task_out = _per_task(group, task_value)
        if not task_out.empty:
            parts.append(task_out)

    if not parts:
        return pd.DataFrame(columns=list(df.columns) + ["y_target"])

    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


# -------------------------------------------------------------------------
# Optional: Per-task standardization using TRAIN stats only (no leakage)
# -------------------------------------------------------------------------
def standardize_per_task(
    combined: pd.DataFrame,
    task_col: str = "task",
    split_col: str = "split",
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Z-score numeric columns PER TASK using TRAIN stats only.
    Returns (scaled_df, stats_df with mean/std per feature per task).
    """
    df = combined.copy()
    if exclude_cols is None:
        exclude_cols = [task_col, split_col, "time", "y", "y_target", "hour", "dow", "month"]

    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude_cols]
    if not num_cols:
        return df, pd.DataFrame(columns=["task", "feature", "mean", "std"])
    df[num_cols] = df[num_cols].astype("float64")

    stats_frames = []
    for task, g in df.groupby(task_col):
        tr = g[g[split_col] == "train"]
        if tr.empty:
            continue
        means = tr[num_cols].mean()
        stds = tr[num_cols].std(ddof=0).replace(0, 1.0)

        idx = g.index
        df.loc[idx, num_cols] = (g[num_cols] - means) / stds

        s = means.to_frame("mean").join(stds.to_frame("std"))
        s["task"] = task
        s = s.reset_index().rename(columns={"index": "feature"})
        stats_frames.append(s)

    stats_df = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame(columns=["task","feature","mean","std"])
    return df, stats_df


# -------------------------------------------------------------------------
# Helper: common calendar encodings
# -------------------------------------------------------------------------
def add_calendar_encodings(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add hour/dow/month and their trig encodings; weekend flag if applicable."""
    df["hour"] = df[time_col].dt.hour
    df["dow"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_daily_calendar_encodings(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add daily calendar features and cyclical encodings."""
    df["day"] = df[time_col].dt.day
    df["dow"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["year"] = df[time_col].dt.year
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_month_start"] = df[time_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[time_col].dt.is_month_end.astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_weekly_calendar_encodings(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add weekly calendar features and cyclical encodings."""
    iso = df[time_col].dt.isocalendar()
    df["weekofyear"] = iso.week.astype(int)
    df["month"] = df[time_col].dt.month
    df["quarter"] = df[time_col].dt.quarter
    df["year"] = df[time_col].dt.year
    df["week_sin"] = np.sin(2 * np.pi * (df["weekofyear"] - 1) / 52)
    df["week_cos"] = np.cos(2 * np.pi * (df["weekofyear"] - 1) / 52)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def safe_task_name(value: object) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
    )


def safe_column_name(value: object) -> str:
    out = str(value).strip().lower()
    replacements = {
        ".": "_",
        " ": "_",
        "-": "_",
        "/": "_",
        "\\": "_",
        "(": "_",
        ")": "",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def download_file(url: str, path: Path, timeout: int = 300) -> Path:
    """Download a file only when it is not already cached."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import requests
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    return path


def write_standard_dataset_outputs(
    combined: pd.DataFrame,
    meta: pd.DataFrame,
    scale_stats: pd.DataFrame,
    out_dir: Path,
    name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "by_task").mkdir(exist_ok=True)

    combined.to_csv(out_dir / f"{name}_full.csv", index=False)
    combined.query("split=='train'").to_csv(out_dir / f"{name}_train.csv", index=False)
    combined.query("split=='val'").to_csv(out_dir / f"{name}_val.csv", index=False)
    combined.query("split=='test'").to_csv(out_dir / f"{name}_test.csv", index=False)
    meta.to_csv(out_dir / f"{name}_meta.csv", index=False)
    scale_stats.to_csv(out_dir / f"{name}_scaler_stats.csv", index=False)

    for task, g in combined.groupby("task"):
        g.to_csv(out_dir / "by_task" / f"{name}_{safe_task_name(task)}.csv", index=False)


def parse_tsf_file(path: Path) -> tuple[pd.DataFrame, Dict[str, str]]:
    """Parse Monash .ts/.tsf forecasting files into a long panel."""
    metadata: Dict[str, str] = {}
    attribute_names: list[str] = []
    rows: list[dict[str, Any]] = []
    in_data = False

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()

            if lower.startswith("@data"):
                in_data = True
                continue

            if not in_data:
                if lower.startswith("@attribute"):
                    parts = line.split()
                    if len(parts) >= 2:
                        attribute_names.append(parts[1])
                elif lower.startswith("@"):
                    parts = line.split(maxsplit=1)
                    metadata[parts[0][1:].lower()] = parts[1] if len(parts) > 1 else ""
                continue

            parts = line.split(":")
            expected = len(attribute_names) + 1
            if len(parts) < expected:
                continue
            attr_values = parts[: len(attribute_names)]
            series_values = ":".join(parts[len(attribute_names):])
            values = [
                np.nan if item.strip() in {"?", "NaN", "nan", ""} else float(item)
                for item in series_values.split(",")
            ]
            row = dict(zip(attribute_names, attr_values))
            row["series_value"] = values
            rows.append(row)

    records: list[pd.DataFrame] = []
    frequency = metadata.get("frequency", "").strip().lower()
    freq_map = {
        "daily": "D",
        "weekly": "W",
        "monthly": "MS",
        "hourly": "H",
    }
    pandas_freq = freq_map.get(frequency, "D")

    for idx, row in enumerate(rows):
        values = pd.Series(row["series_value"], dtype="float32")
        task = row.get("series_name") or row.get("series_id") or row.get("item_id") or f"series_{idx:03d}"
        start_raw = row.get("start_timestamp") or row.get("start")
        if start_raw:
            start = pd.to_datetime(start_raw, errors="coerce")
        else:
            start = pd.Timestamp("2000-01-01")
        if pd.isna(start):
            start = pd.Timestamp("2000-01-01")
        time = pd.date_range(start=start, periods=len(values), freq=pandas_freq)
        records.append(pd.DataFrame({"time": time, "task": str(task), "y": values}))

    if not records:
        raise RuntimeError(f"No series found in TSF file: {path}")
    return pd.concat(records, ignore_index=True), metadata


# -------------------------------------------------------------------------
# ELECTRICITY (UCI)
# -------------------------------------------------------------------------
def _download_ucielec_zip() -> bytes:
    """Download the UCI Electricity Load Diagrams (2011–2014) ZIP as bytes."""
    import requests  # lazy import
    url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def prepare_electricity(
    out_dir: Path = Path("data/interim/electricity"),
    resample_rule: str = "1H",
    chunksize: int = 200_000,
    max_ntasks: Optional[int] = None,
) -> None:
    """
    Robust + memory-efficient preparation of UCI Electricity Load Diagrams.

    Fixes:
      - Handles malformed lines / variable columns with engine='python' + on_bad_lines='skip'
      - Correct resampling across chunks using hourly (sum,count) aggregation merged across chunks
        (prevents duplicate-hour artifacts / stitched datasets)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "by_task").mkdir(exist_ok=True)

    raw_zip = _download_ucielec_zip()
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".txt")]
        if not members:
            raise RuntimeError("UCI electricity ZIP: no .txt file found.")
        main_member = members[0]

        # Read header only to discover columns (use tolerant parser)
        with zf.open(main_member) as fh:
            hdr_df = pd.read_csv(
                fh,
                sep=";",
                decimal=",",
                nrows=0,
                engine="python",
                on_bad_lines="skip",
            )

        cols = list(hdr_df.columns)
        if not cols:
            raise RuntimeError("Could not read header from UCI electricity file.")
        time_col = cols[0]
        meter_cols = cols[1:]
        if not meter_cols:
            raise RuntimeError("No meter columns found in UCI electricity file.")

    # Optionally limit number of tasks for faster testing
    if max_ntasks is not None:
        meter_cols = select_random_tasks(tasks_id=meter_cols, max_ntasks=max_ntasks)

    # Output paths
    full_path  = out_dir / "electricity_full.csv"
    train_path = out_dir / "electricity_train.csv"
    val_path   = out_dir / "electricity_val.csv"
    test_path  = out_dir / "electricity_test.csv"
    meta_path  = out_dir / "electricity_meta.csv"

    final_columns = None
    wrote_headers = {"full": False, "train": False, "val": False, "test": False}
    meta_rows = []

    def _append(df: pd.DataFrame, path: Path, which: str):
        nonlocal final_columns, wrote_headers
        if final_columns is None:
            final_columns = list(df.columns)
        df = df.reindex(columns=final_columns)
        header = not wrote_headers[which]
        df.to_csv(path, mode="a", index=False, header=header)
        wrote_headers[which] = True

    # Process each meter independently
    for idx, meter in enumerate(meter_cols, start=1):
        print(f"[{idx}/{len(meter_cols)}] Processing meter: {meter}")

        # We'll aggregate per chunk into hourly sum+count, then merge at the end.
        hour_sum_parts: list[pd.DataFrame] = []
        hour_cnt_parts: list[pd.DataFrame] = []

        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
            with zf.open(main_member) as fh:
                reader = pd.read_csv(
                    fh,
                    sep=";",
                    decimal=",",
                    usecols=[time_col, meter],
                    parse_dates=[time_col],
                    chunksize=chunksize,
                    low_memory=True,
                    engine="python",
                    on_bad_lines="skip",
                )

                for chunk in reader:
                    chunk = chunk.rename(columns={time_col: "time", meter: "y"})
                    chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
                    chunk["y"] = pd.to_numeric(chunk["y"], errors="coerce")
                    chunk = chunk.dropna(subset=["time", "y"])
                    if chunk.empty:
                        continue

                    # Put into hourly bins; compute sum and count (correct across chunk boundaries)
                    chunk = chunk.set_index("time").sort_index()
                    # sum/count per hour in this chunk
                    h_sum = chunk["y"].resample(resample_rule).sum(min_count=1).to_frame("y_sum")
                    h_cnt = chunk["y"].resample(resample_rule).count().to_frame("y_cnt")

                    # Keep only hours with at least 1 obs
                    h = h_sum.join(h_cnt, how="inner")
                    h = h[h["y_cnt"] > 0]
                    if not h.empty:
                        hour_sum_parts.append(h[["y_sum"]].reset_index())
                        hour_cnt_parts.append(h[["y_cnt"]].reset_index())

        if not hour_sum_parts:
            print(f"  -> no data found for meter {meter}, skipping.")
            continue

        # Merge chunk-level (sum,count) and compute final hourly mean
        sum_df = pd.concat(hour_sum_parts, ignore_index=True)
        cnt_df = pd.concat(hour_cnt_parts, ignore_index=True)

        # Combine across chunks by hour
        sum_df = sum_df.groupby("time", as_index=False)["y_sum"].sum()
        cnt_df = cnt_df.groupby("time", as_index=False)["y_cnt"].sum()

        g = sum_df.merge(cnt_df, on="time", how="inner")
        g["y"] = (g["y_sum"] / g["y_cnt"]).astype("float32")
        g = g.drop(columns=["y_sum", "y_cnt"]).dropna(subset=["time", "y"]).sort_values("time").reset_index(drop=True)

        # Sanity check: there should be no duplicate hours now
        dup = int(g["time"].duplicated().sum())
        if dup:
            print(f"  ⚠️  Unexpected: {dup} duplicated hourly timestamps for {meter} (should be 0).")
            g = g.groupby("time", as_index=False)["y"].mean().sort_values("time").reset_index(drop=True)

        # Build per-task long-form
        g["task"] = meter

        # Calendar encodings
        g = add_calendar_encodings(g, "time")

        # Domain features (past-only rolling baselines)
        g = g.sort_values(["task", "time"]).reset_index(drop=True)
        g["y_day_mean"] = g.groupby("task")["y"].transform(
            lambda s: s.shift(1).rolling(24, min_periods=24).mean()
        ).astype("float32")
        g["y_week_mean"] = g.groupby("task")["y"].transform(
            lambda s: s.shift(1).rolling(168, min_periods=168).mean()
        ).astype("float32")

        # Supervised panel
        covs = [c for c in g.columns if c not in ["time", "task", "y", "split"]]
        g_sup = make_supervised_panel(
            g,
            task_col="task",
            time_col="time",
            y_col="y",
            covariates=covs,
            horizon=1,
            y_lags=[1, 24, 168],
            cov_lags=[1, 24],
            roll_windows=[24, 168],
        )

        if g_sup.empty:
            print(f"  -> supervised set empty for meter {meter}, skipping.")
            continue

        # Split this task only
        combined_task, meta_task = temporal_split_per_task(g_sup, task_col="task", time_col="time")

        # Append to global CSVs
        _append(combined_task, full_path,  "full")
        _append(combined_task.query("split=='train'"), train_path, "train")
        _append(combined_task.query("split=='val'"),   val_path,   "val")
        _append(combined_task.query("split=='test'"),  test_path,  "test")

        # Per-task CSV
        safe_task = str(meter).replace("/", "_").replace(" ", "_")
        combined_task.to_csv(out_dir / "by_task" / f"electricity_{safe_task}.csv", index=False)

        meta_rows.append(meta_task.iloc[0].to_dict())

        # Free memory
        del hour_sum_parts, hour_cnt_parts, sum_df, cnt_df, g, g_sup, combined_task, meta_task

    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    print(f"✅ Electricity dataset (robust + correct chunked resampling) prepared at: {out_dir.resolve()}")


# -------------------------------------------------------------------------
# EXCHANGE (ECB)
# -------------------------------------------------------------------------
def _download_ecb_hist_zip() -> bytes:
    """Download the ECB historical EUR FX rates ZIP (CSV inside)."""
    import requests  # lazy import
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def prepare_exchange(
    out_dir: Path = Path("data/interim/exchange"),
    currencies: Optional[Iterable[str]] = None,
) -> None:
    """Download → normalize → derive features → supervised → split → scale → save."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "by_task").mkdir(exist_ok=True)

    raw_zip = _download_ecb_hist_zip()
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise RuntimeError("ECB FX ZIP: no .csv file found.")
        with zf.open(members[0]) as fh:
            df = pd.read_csv(fh)

    df = df.rename(columns={df.columns[0]: "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    if currencies is None:
        currencies = ["USD", "GBP", "JPY", "CHF", "AUD", "CAD", "CNY"]
    keep_cols = ["time"] + [c for c in currencies if c in df.columns]
    df = df[keep_cols].sort_values("time")

    df_long = df.melt(id_vars=["time"], var_name="task", value_name="y").dropna(subset=["y"])

    # calendar features (daily)
    df_long["day"] = df_long["time"].dt.day
    df_long["month"] = df_long["time"].dt.month
    df_long["year"] = df_long["time"].dt.year
    df_long["is_month_start"] = df_long["time"].dt.is_month_start.astype(int)
    df_long["is_month_end"] = df_long["time"].dt.is_month_end.astype(int)

    # --- domain features (finance) with TRANSFORM to keep index alignment ---
    df_long = df_long.sort_values(["task", "time"]).reset_index(drop=True)

    # robust log and returns (guard tiny values)
    df_long["log_y"] = np.log(df_long["y"].astype(float).clip(lower=1e-12))
    df_long["ret"] = df_long.groupby("task")["log_y"].transform(lambda s: s.diff())

    # rolling vol on PAST returns
    df_long["vol_5"] = df_long.groupby("task")["ret"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=5).std()
    )
    df_long["vol_21"] = df_long.groupby("task")["ret"].transform(
        lambda s: s.shift(1).rolling(21, min_periods=21).std()
    )

    df_long = df_long.drop(columns=["log_y"])

    covs = [c for c in df_long.columns if c not in ["time", "task", "y", "split"]]
    df_feats = make_supervised_panel(
        df_long, task_col="task", time_col="time", y_col="y",
        covariates=covs, horizon=1,
        y_lags=[1, 5, 21], cov_lags=[1, 5], roll_windows=[5, 21],
    )

    combined, meta = temporal_split_per_task(df_feats, task_col="task", time_col="time")
    combined_scaled, scale_stats = standardize_per_task(combined)

    name = "exchange"
    combined_scaled.to_csv(out_dir / f"{name}_full.csv", index=False)
    combined_scaled.query("split=='train'").to_csv(out_dir / f"{name}_train.csv", index=False)
    combined_scaled.query("split=='val'").to_csv(out_dir / f"{name}_val.csv", index=False)
    combined_scaled.query("split=='test'").to_csv(out_dir / f"{name}_test.csv", index=False)
    meta.to_csv(out_dir / f"{name}_meta.csv", index=False)
    scale_stats.to_csv(out_dir / f"{name}_scaler_stats.csv", index=False)

    for task, g in combined_scaled.groupby("task"):
        g.to_csv(out_dir / "by_task" / f"{name}_{str(task).replace('/','_').replace(' ','_')}.csv", index=False)

    print(f"✅ Exchange dataset prepared at: {out_dir.resolve()}")


# -------------------------------------------------------------------------
# METR-LA — Zenodo CSV mirror (5-min)
# -------------------------------------------------------------------------
def _download_metrla_zenodo_csv(raw_dir: Path = Path("data/raw/metrla")) -> Path:
    """Download METR-LA (record 5146275) → METR-LA.csv in raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "METR-LA.csv"
    if csv_path.exists():
        return csv_path
    import requests
    rec_api = "https://zenodo.org/api/records/5146275"
    r = requests.get(rec_api, timeout=60)
    r.raise_for_status()
    rec = r.json()
    files = rec.get("files", [])
    if not files:
        raise RuntimeError("Zenodo 5146275 JSON: no files.")
    target = None
    for f in files:
        name = f.get("key") or f.get("filename") or ""
        if name.lower() == "metr-la.csv":
            target = f; break
    if target is None:
        raise RuntimeError("METR-LA.csv not found in Zenodo record.")
    url = target.get("links", {}).get("self") or target.get("links", {}).get("download")
    if not url:
        raise RuntimeError("Zenodo file entry lacks direct download link.")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(csv_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk: fh.write(chunk)
    return csv_path


def prepare_metrla(
    out_dir: Path = Path("data/interim/metrla"),
    max_ntasks: Optional[int] = None,
) -> None:
    """Download → normalize → derive features → supervised → split → scale → save."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "by_task").mkdir(exist_ok=True)

    raw_dir = Path("data/raw/metrla")
    csv_path = raw_dir / "METR-LA.csv"
    if not csv_path.exists():
        print(">>> Downloading METR-LA from Zenodo (CSV mirror) …")
        csv_path = _download_metrla_zenodo_csv(raw_dir)

    df_wide = pd.read_csv(csv_path, low_memory=False)
    if df_wide.empty: raise RuntimeError(f"Empty METR-LA CSV at {csv_path}")

    cols = list(df_wide.columns)
    if cols[0].lower() != "time":
        df_wide = df_wide.rename(columns={cols[0]: "time"})

    df_wide["time"] = pd.to_datetime(df_wide["time"], errors="coerce")
    df_wide = df_wide.dropna(subset=["time"]).sort_values("time")

    value_cols = [c for c in df_wide.columns if c != "time"]
    if not value_cols: raise RuntimeError("No sensor columns found in METR-LA CSV.")

    # Optionally limit number of tasks for faster testing
    if max_ntasks is not None:
        value_cols = select_random_tasks(tasks_id=value_cols, max_ntasks=max_ntasks)

    df_long = df_wide.melt(
        id_vars=["time"],
        value_vars=value_cols,
        var_name="task",
        value_name="y",
    ).dropna(subset=["y"])
    df_long = add_calendar_encodings(df_long, "time")

    # --- domain features (traffic) ---
    df_long = df_long.sort_values(["task", "time"])
    df_long["delta_speed"] = df_long.groupby("task")["y"].transform(lambda s: s.diff())

    # optional spatial degree if adjacency is available
    adj_pkl = Path("data/raw/metrla/adj_mx.pkl")
    if adj_pkl.exists():
        try:
            with open(adj_pkl, "rb") as f:
                adj_tuple = pickle.load(f)
                adj = adj_tuple[2] if isinstance(adj_tuple, tuple) and len(adj_tuple) >= 3 else adj_tuple
            deg = np.asarray(adj).sum(axis=1).ravel()
            unique_tasks = sorted(df_long["task"].unique())
            if len(unique_tasks) == len(deg):
                degree_map = {t: float(deg[i]) for i, t in enumerate(unique_tasks)}
                df_long["degree"] = df_long["task"].map(degree_map)
        except Exception:
            pass  # skip if incompatible

    covs = [c for c in df_long.columns if c not in ["time", "task", "y", "split"]]
    df_feats = make_supervised_panel(
        df_long, task_col="task", time_col="time", y_col="y",
        covariates=covs, horizon=12,          # 12 * 5min ≈ 1 hour ahead
        y_lags=[1, 12, 288], cov_lags=[1, 12], roll_windows=[12, 288],
    )

    combined, meta = temporal_split_per_task(df_feats, task_col="task", time_col="time")
    combined_scaled, scale_stats = standardize_per_task(combined)

    name = "metrla"
    combined_scaled.to_csv(out_dir / f"{name}_full.csv", index=False)
    combined_scaled.query("split=='train'").to_csv(out_dir / f"{name}_train.csv", index=False)
    combined_scaled.query("split=='val'").to_csv(out_dir / f"{name}_val.csv", index=False)
    combined_scaled.query("split=='test'").to_csv(out_dir / f"{name}_test.csv", index=False)
    meta.to_csv(out_dir / f"{name}_meta.csv", index=False)
    scale_stats.to_csv(out_dir / f"{name}_scaler_stats.csv", index=False)

    for task, g in combined_scaled.groupby("task"):
        g.to_csv(out_dir / "by_task" / f"{name}_{str(task).replace('/','_').replace(' ','_')}.csv", index=False)

    print(f"✅ METR-LA dataset prepared at: {out_dir.resolve()}")


# -------------------------------------------------------------------------
# NN5 Daily — Monash Forecasting Repository
# -------------------------------------------------------------------------
def _download_nn5_daily(raw_dir: Path = Path("data/raw/nn5")) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_file(
        "https://zenodo.org/records/4656117/files/nn5_daily_dataset_without_missing_values.zip?download=1",
        raw_dir / "nn5_daily_dataset_without_missing_values.zip",
    )
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".ts", ".tsf"))]
        if not members:
            raise RuntimeError("NN5 ZIP did not contain a .ts/.tsf file.")
        out_path = raw_dir / Path(members[0]).name
        if not out_path.exists():
            with zf.open(members[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    return out_path


def prepare_nn5(
    out_dir: Path = Path("data/interim/nn5"),
    max_ntasks: Optional[int] = None,
) -> None:
    """Prepare the Monash NN5 Daily ATM withdrawals dataset."""
    ts_path = _download_nn5_daily()
    df_long, _ = parse_tsf_file(ts_path)
    df_long = df_long.dropna(subset=["y"]).sort_values(["task", "time"]).reset_index(drop=True)

    if max_ntasks is not None:
        keep = select_random_tasks(df_long["task"].drop_duplicates().tolist(), max_ntasks=max_ntasks)
        df_long = df_long[df_long["task"].isin(keep)].copy()

    df_long = add_daily_calendar_encodings(df_long, "time")
    covs = [c for c in df_long.columns if c not in ["time", "task", "y", "split"]]
    df_feats = make_supervised_panel(
        df_long,
        task_col="task",
        time_col="time",
        y_col="y",
        covariates=covs,
        horizon=1,
        y_lags=[1, 7, 14, 28],
        cov_lags=[1, 7],
        roll_windows=[7, 28],
    )

    combined, meta = temporal_split_per_task(df_feats, task_col="task", time_col="time")
    combined_scaled, scale_stats = standardize_per_task(combined)
    write_standard_dataset_outputs(combined_scaled, meta, scale_stats, out_dir, "nn5")
    print(f"✅ NN5 Daily dataset prepared at: {out_dir.resolve()} ({meta.shape[0]} tasks)")




# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download, normalize, and create supervised multitask datasets (targets + lags + domain features)."
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["electricity", "exchange", "metrla", "nn5"],
        help="Which dataset to prepare.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Custom output directory under data/interim/<dataset>.",
    )
    p.add_argument(
        "--currencies",
        type=str,
        default=None,
        help="Comma-separated currency codes for 'exchange' (e.g., 'USD,GBP,JPY').",
    )
    p.add_argument("--config", type=str, default=None, required=False)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)


def select_random_tasks(tasks_id: List[str], max_ntasks: int) -> List[str]:
    if max_ntasks < len(tasks_id):
        print(f"Limiting to {max_ntasks} tasks for testing. (Originally {len(tasks_id)} tasks available.)")
        return np.random.choice(tasks_id, size=max_ntasks, replace=False).tolist()
    print(f"The number of tasks ({len(tasks_id)}) is less than or equal to max_ntasks ({max_ntasks}), so no limiting applied.")
    return tasks_id


def main():
    args = build_argparser().parse_args()
    dataset = args.dataset

    if args.config is not None and Path(args.config).exists():
        config = load_yaml(args.config)
        max_ntasks = config.get("max_ntasks", None)
        seed = config.get("seed", None)
        dataset = config.get("dataset", args.dataset)
        if seed is not None:
            set_seed(seed)
    else:
        max_ntasks = None

    if dataset == "electricity":
        out = args.outdir or Path("data/interim/electricity")
        prepare_electricity(out_dir=out, max_ntasks=max_ntasks)
    elif dataset == "exchange":
        out = args.outdir or Path("data/interim/exchange")
        curr = None
        if args.currencies:
            curr = [c.strip().upper() for c in args.currencies.split(",") if c.strip()]
        prepare_exchange(out_dir=out, currencies=curr)
    elif dataset == "metrla":
        out = args.outdir or Path("data/interim/metrla")
        prepare_metrla(out_dir=out, max_ntasks=max_ntasks)
    elif dataset == "nn5":
        out = args.outdir or Path("data/interim/nn5")
        prepare_nn5(out_dir=out, max_ntasks=max_ntasks)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
