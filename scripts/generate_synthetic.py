"""
generate_synthetic.py
=====================

Generate synthetic multi-task regression datasets for evaluating the
ALE–Fréchet Multi-Task Learning (MTL) framework.

This script produces two deterministic and interpretable datasets:

1. **Multi-Sine Regression**
2. **Polynomial Family Regression**

Each dataset defines a collection of *related but non-identical tasks*,
used to evaluate how the ALE–Fréchet similarity captures inter-task
relationships, functional smoothness, and shared feature importance.

What's new (supervised features)
--------------------------------
For consistency with the real datasets, we transform the synthetic panels
into supervised rows with:
- Forecast target: ``y_target = y.shift(-horizon)``
- Lags of the target: ``y_lag_L`` for L in a chosen list
- Optional lags of covariates (here simple functions of x): ``<cov>_lag_L``
- Rolling baselines on *past* ``y``: ``y_rollmean_W`` and ``y_rollstd_W``,
  computed from ``y.shift(1).rolling(W)`` to avoid look-ahead leakage.

Default hyperparameters (can be changed in code):
- Multi-Sine:  horizon=1, y_lags=[1, 10, 50], cov_lags=[1], roll=[10, 50]
- Polynomial:  horizon=1, y_lags=[1, 10, 50], cov_lags=[1], roll=[10, 50]

Differences Across Tasks
------------------------
- **Multi-Sine Regression**
  Each task corresponds to a sinusoidal function parameterized by:
  - Amplitude (:math:`a_t`) — scales the oscillation magnitude.
  - Frequency (:math:`b_t`) — controls oscillation rate.
  - Phase (:math:`\\phi_t`) — shifts the waveform horizontally.

  Tasks differ in their amplitude, frequency, and phase following evenly
  spaced parameter grids. Thus, some tasks are highly correlated (e.g.,
  similar phase/frequency), while others are less related. The dataset
  evaluates whether the ALE–Fréchet similarity correctly identifies such
  task affinities based on local feature effects.

- **Polynomial Family Regression**
  Each task corresponds to a polynomial of degree *d*, with coefficients
  :math:`w_0, w_1, \\ldots, w_d` sampled uniformly in [-1.5, 1.5].

  Tasks differ in both the sign and magnitude of coefficients, producing
  distinct nonlinear mappings between input :math:`x` and target :math:`y`.
  This setup tests whether the framework captures geometric similarity
  between functions (e.g., convex vs. concave shapes) rather than relying
  on explicit parameter sharing.

Outputs
-------
Each dataset includes:
- Input `x` and simple encodings (`x_sin`, `x_cos`)
- Target `y` and forecast `y_target`
- Lags/rolling features (e.g., `y_lag_1`, `y_rollmean_10`)
- Task identifier `task`
- Generating parameters (e.g., amplitude, polynomial coefficients)
- Split label `split` ∈ {train, val, test}

The script saves CSV files under `data/interim/`, organized as:

data/interim/
├── multisine/
│   ├── multisine_full.csv
│   ├── multisine_train.csv
│   ├── multisine_val.csv
│   ├── multisine_test.csv
│   ├── multisine_meta.csv
│   └── by_task/
│       ├── multisine_task_1.csv
│       ├── ...
└── polynomial/
    ├── polynomial_full.csv
    ├── polynomial_train.csv
    ├── polynomial_val.csv
    ├── polynomial_test.csv
    ├── polynomial_meta.csv
    └── by_task/
        ├── polynomial_task_1.csv
        ├── ...

Usage
-----
Run directly from the repository root:

>>> python scripts/generate_synthetic.py

Examples
--------
Visualize the generated Multi-Sine tasks:

>>> import pandas as pd, matplotlib.pyplot as plt
>>> df = pd.read_csv("data/interim/multisine/multisine_full.csv")
>>> for t in df["task"].unique():
...     d = df[(df["task"] == t) & (df["split"] == "train")]
...     plt.plot(d["x"], d["y"], label=t)
>>> plt.legend(); plt.show()

License
-------
Released under the MIT License.
"""

from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Utility: temporal split per task (order by x)
# -------------------------------------------------------------------------
def temporal_split_per_task(
    df: pd.DataFrame,
    task_col: str = "task",
    order_col: str = "x",
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each task's samples sequentially into train, validation, and test sets.

    Notes
    -----
    The split preserves the original order of samples by `order_col`,
    simulating temporal consistency for time-series experiments.
    """
    assert abs(train + val + test - 1.0) < 1e-8
    parts, meta = [], []
    for task, g in df.groupby(task_col, sort=True):
        g = g.sort_values(order_col).reset_index(drop=True)
        n = len(g)
        n_tr = int(n * train)
        n_val = int(n * val)
        n_te = n - n_tr - n_val
        parts += [
            g.iloc[:n_tr].assign(split="train"),
            g.iloc[n_tr:n_tr + n_val].assign(split="val"),
            g.iloc[n_tr + n_val:].assign(split="test"),
        ]
        meta.append({
            "task": task,
            "n_total": n,
            "n_train": n_tr,
            "n_val": n_val,
            "n_test": n_te,
        })
    return pd.concat(parts, ignore_index=True), pd.DataFrame(meta)


# -------------------------------------------------------------------------
# Helper: Supervised sequence transform (per-task; no leakage)
# -------------------------------------------------------------------------
def make_supervised_sequence(
    df: pd.DataFrame,
    task_col: str = "task",
    order_col: str = "x",
    y_col: str = "y",
    covariates: Optional[List[str]] = None,
    horizon: int = 1,
    y_lags: Optional[List[int]] = None,
    cov_lags: Optional[List[int]] = None,
    roll_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Turn a long-form panel ordered by `order_col` into supervised rows with
    y_target, lags, and rolling features (computed from strictly past values).

    Implementation note
    -------------------
    We iterate groups with a for-loop instead of `groupby().apply(...)` to avoid
    the pandas 2.x FutureWarning about grouping columns.
    """
    df = df.copy().sort_values([task_col, order_col])

    y_lags = y_lags or [1]
    cov_lags = cov_lags or []
    roll_windows = roll_windows or []
    covariates = covariates or []

    def _per_task(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(order_col).copy()
        g["y_target"] = g[y_col].shift(-horizon)

        # Target lags
        for L in y_lags:
            g[f"y_lag_{L}"] = g[y_col].shift(L)

        # Covariate lags
        for c in covariates:
            if c not in g.columns:
                continue
            if not pd.api.types.is_numeric_dtype(g[c]):
                continue
            for L in cov_lags:
                g[f"{c}_lag_{L}"] = g[c].shift(L)

        # Rolling features (strictly past)
        for W in roll_windows:
            past = g[y_col].shift(1)
            g[f"y_rollmean_{W}"] = past.rolling(W, min_periods=W).mean()
            g[f"y_rollstd_{W}"] = past.rolling(W, min_periods=W).std()

        # Drop rows without full features/target
        need_cols = ["y_target"] + [
            c for c in g.columns
            if c.startswith(("y_lag_", "y_roll"))
            or any(c.endswith(f"_lag_{L}") for L in cov_lags)
        ]
        g = g.dropna(subset=need_cols)
        return g

    # Iterate groups explicitly — no FutureWarning and we keep `task` column
    out_frames: List[pd.DataFrame] = []
    for task, g in df.groupby(task_col, sort=True):
        gg = _per_task(g)
        # Ensure task column is present (safe even if already there)
        gg[task_col] = task
        out_frames.append(gg)

    if not out_frames:
        return pd.DataFrame(columns=list(df.columns) + ["y_target"])

    return pd.concat(out_frames, ignore_index=True)


# -------------------------------------------------------------------------
# 1) Multi-Sine Regression Dataset (raw panel)
# -------------------------------------------------------------------------
def generate_multisine(
    n_tasks: int = 5,
    n_samples: int = 2000,
    noise_std: float = 0.05,
    seed: int = 42,
    horizon: int = 1,
    y_lags: Optional[List[int]] = None,
    cov_lags: Optional[List[int]] = None,
    roll_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Generate sinusoidal regression tasks with different amplitudes, frequencies,
    and phases.

    Note
    ----
    We intentionally return the *raw panel* (x, y, task, params, covariates).
    Supervised features/targets are built **after** temporal splitting (see
    `save_dataset`) to avoid leakage of labels across split boundaries.
    """
    # defaults similar to hourly real data (but count in sample steps)
    y_lags = y_lags or [1, 10, 50]
    cov_lags = cov_lags or [1]
    roll_windows = roll_windows or [10, 50]

    rng = np.random.default_rng(seed)
    X = np.linspace(0, 2 * np.pi, n_samples)
    amplitudes = np.linspace(0.5, 1.5, n_tasks)
    frequencies = np.linspace(1.0, 2.0, n_tasks)
    phases = np.linspace(0, np.pi / 2, n_tasks)

    frames = []
    for t in range(n_tasks):
        y = amplitudes[t] * np.sin(frequencies[t] * X + phases[t])
        y += rng.normal(0, noise_std, size=n_samples)

        base = pd.DataFrame({
            "x": X.astype(float),
            "x_sin": np.sin(X),
            "x_cos": np.cos(X),
            "y": y.astype(float),
            "task": f"task_{t+1}",
            "amplitude": amplitudes[t],
            "frequency": frequencies[t],
            "phase": phases[t],
            "seed": seed
        })
        frames.append(base)

    return pd.concat(frames, ignore_index=True)


# -------------------------------------------------------------------------
# 2) Polynomial Family Dataset (raw panel)
# -------------------------------------------------------------------------
def generate_polynomial(
    n_tasks: int = 5,
    n_samples: int = 2000,
    degree: int = 3,
    noise_std: float = 0.05,
    seed: int = 123,
    horizon: int = 1,
    y_lags: Optional[List[int]] = None,
    cov_lags: Optional[List[int]] = None,
    roll_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Generate polynomial regression tasks with random coefficients, then build
    a raw panel.

    Note
    ----
    We intentionally return the *raw panel*. Supervised features/targets are
    built **after** temporal splitting (see `save_dataset`) to avoid leakage of
    labels across split boundaries.
    """
    # defaults similar to multisine
    y_lags = y_lags or [1, 10, 50]
    cov_lags = cov_lags or [1]
    roll_windows = roll_windows or [10, 50]

    rng = np.random.default_rng(seed)
    X = np.linspace(-2, 2, n_samples)
    frames = []

    for t in range(n_tasks):
        coeffs = rng.uniform(-1.5, 1.5, size=degree + 1)
        y = sum(coeffs[k] * (X ** k) for k in range(degree + 1))
        y += rng.normal(0, noise_std, size=n_samples)

        base = pd.DataFrame({
            "x": X.astype(float),
            "x_sin": np.sin(X),  # simple periodic encoding analogous to calendar features
            "x_cos": np.cos(X),
            "y": y.astype(float),
            "task": f"task_{t+1}",
            "degree": degree,
            "seed": seed,
            **{f"w{k}": coeffs[k] for k in range(degree + 1)}
        })

        frames.append(base)

    return pd.concat(frames, ignore_index=True)


# -------------------------------------------------------------------------
# File saving helper
# -------------------------------------------------------------------------
def save_dataset(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    *,
    horizon: int = 1,
    y_lags: Optional[List[int]] = None,
    cov_lags: Optional[List[int]] = None,
    roll_windows: Optional[List[int]] = None,
    covariates: Optional[List[str]] = None,
) -> None:
    """
    Save dataset to disk with temporal splits and per-task CSV files.

    Option A (strict no-leakage)
    ---------------------------
    We split each task *first* on the raw panel (ordered by `x`) and then build
    supervised rows **within each split independently**. This guarantees that:
    - Train targets never reference validation/test labels.
    - Validation targets never reference test labels.
    - Lags/rolling features are computed only from history inside the same split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "by_task").mkdir(exist_ok=True)

    # 1) Split raw panel per task
    split_raw, _ = temporal_split_per_task(df, order_col="x")

    # 2) Build supervised rows *within each split* (no cross-split leakage)
    y_lags = y_lags or [1]
    cov_lags = cov_lags or []
    roll_windows = roll_windows or []
    covariates = covariates or []

    split_frames: List[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        part = split_raw.query("split==@split_name").copy()
        if part.empty:
            continue
        sup = make_supervised_sequence(
            part,
            task_col="task",
            order_col="x",
            y_col="y",
            covariates=covariates,
            horizon=horizon,
            y_lags=y_lags,
            cov_lags=cov_lags,
            roll_windows=roll_windows,
        )
        # `split` column is preserved by make_supervised_sequence; be defensive.
        if "split" not in sup.columns:
            sup["split"] = split_name
        split_frames.append(sup)

    combined = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()

    # 3) Meta computed on the final supervised table (counts after dropping NA)
    meta_rows = []
    if not combined.empty:
        for task, g in combined.groupby("task", sort=True):
            meta_rows.append({
                "task": task,
                "n_total": int(len(g)),
                "n_train": int((g["split"] == "train").sum()),
                "n_val": int((g["split"] == "val").sum()),
                "n_test": int((g["split"] == "test").sum()),
            })
    meta = pd.DataFrame(meta_rows)

    # 4) Save
    combined.to_csv(out_dir / f"{name}_full.csv", index=False)
    combined.query("split=='train'").to_csv(out_dir / f"{name}_train.csv", index=False)
    combined.query("split=='val'").to_csv(out_dir / f"{name}_val.csv", index=False)
    combined.query("split=='test'").to_csv(out_dir / f"{name}_test.csv", index=False)
    meta.to_csv(out_dir / f"{name}_meta.csv", index=False)

    for task, g in combined.groupby("task"):
        g.to_csv(out_dir / "by_task" / f"{name}_{task}.csv", index=False)

    print(f"Saved dataset '{name}' to {out_dir.resolve()}")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    base = Path("data/interim")

    # Multi-Sine (you can tweak horizon/lags/windows here)
    multisine_df = generate_multisine(
        n_tasks=5, n_samples=2000, noise_std=0.05, seed=42,
        horizon=1, y_lags=[1, 10, 50], cov_lags=[1], roll_windows=[10, 50],
    )
    save_dataset(
        multisine_df,
        base / "multisine",
        "multisine",
        horizon=1,
        y_lags=[1, 10, 50],
        cov_lags=[1],
        roll_windows=[10, 50],
        covariates=["x", "x_sin", "x_cos"],
    )

    # Polynomial (you can tweak horizon/lags/windows here)
    polynomial_df = generate_polynomial(
        n_tasks=5, n_samples=2000, degree=3, noise_std=0.05, seed=123,
        horizon=1, y_lags=[1, 10, 50], cov_lags=[1], roll_windows=[10, 50],
    )
    save_dataset(
        polynomial_df,
        base / "polynomial",
        "polynomial",
        horizon=1,
        y_lags=[1, 10, 50],
        cov_lags=[1],
        roll_windows=[10, 50],
        covariates=["x", "x_sin", "x_cos"],
    )

    print("\nAll synthetic datasets successfully generated.")
