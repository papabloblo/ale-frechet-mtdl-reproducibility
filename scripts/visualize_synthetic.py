"""
visualize_synthetic.py
======================

Visualize the synthetic multi-task regression datasets used to evaluate the
ALE–Fréchet Multi-Task Learning (MTL) framework.

This script is **only for visualization**. It assumes the datasets have
already been generated and saved under `data/interim/` by `generate_synthetic.py`.

It produces:
    - An overlay plot per dataset (all tasks on one figure).
    - (Optional) A per-task grid of plots.

Differences Across Tasks
------------------------
- Multi-Sine Regression
  Each task corresponds to a sinusoidal function with distinct amplitude a_t,
  frequency b_t, and phase φ_t, arranged on evenly spaced grids. Tasks with
  similar parameters exhibit similar waveforms, which should manifest as higher
  explainable similarity in ALE–Fréchet space.

- Polynomial Family Regression
  Each task is a polynomial of degree d with coefficients w_0..w_d sampled
  independently in [-1.5, 1.5]. Tasks therefore differ in curvature, offset,
  and trend (convex, concave, mixed), enabling geometric similarity analysis.

Inputs
------
The script expects the following files to exist:

data/interim/
├── multisine/
│   └── multisine_full.csv
└── polynomial/
    └── polynomial_full.csv

Each CSV must include: `x`, `y`, `task`, and (optionally) a `split` column.

Outputs
-------
PNG figures will be saved to the directory passed via `--outdir` (default:
`reports/figures/`):

- multisine_overlay.png
- polynomial_overlay.png
- (optional) multisine_per_task.png
- (optional) polynomial_per_task.png

Usage
-----
Visualize both datasets and save figures:

    $ python scripts/visualize_synthetic.py --dataset all --outdir reports/figures

Visualize only Multi-Sine with per-task grid and limit to 6 tasks:

    $ python scripts/visualize_synthetic.py --dataset multisine --per-task --max-tasks 6

Select only the training split (if a `split` column exists):

    $ python scripts/visualize_synthetic.py --dataset polynomial --split train

License
-------
MIT License.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------ I/O ------------------------------------ #
def _dataset_paths(root: Path) -> dict[str, Path]:
    return {
        "multisine": root / "multisine" / "multisine_full.csv",
        "polynomial": root / "polynomial" / "polynomial_full.csv",
    }


def load_dataset(
    dataset: str,
    data_root: Path = Path("data/interim"),
    split: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a synthetic dataset CSV into a DataFrame.

    Parameters
    ----------
    dataset : {'multisine', 'polynomial'}
        Name of the dataset to load.
    data_root : pathlib.Path, default='data/interim'
        Root directory containing the dataset subfolders.
    split : {'train','val','test', None}, optional
        If provided and the CSV contains a 'split' column, filter
        rows to the selected split.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with at least the columns ['x','y','task'] and
        optionally ['split', generating parameters].

    Raises
    ------
    FileNotFoundError
        If the expected CSV file is not found.

    Notes
    -----
    This function does not perform any generation; it only reads
    precomputed CSV files created by `generate_synthetic.py`.
    """
    paths = _dataset_paths(data_root)
    if dataset not in paths:
        raise ValueError("dataset must be one of {'multisine','polynomial'}")
    csv_path = paths[dataset]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {csv_path}\n"
            "Run `python scripts/generate_synthetic.py` first."
        )
    df = pd.read_csv(csv_path)
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split].copy()
    return df


# ---------------------------- Plotting --------------------------------- #
def plot_overlay(
    df: pd.DataFrame,
    title: str,
    outfile: Path,
    max_tasks: Optional[int] = None,
) -> None:
    """
    Plot all tasks on a single figure (overlay).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with columns ['x','y','task'].
    title : str
        Title for the plot.
    outfile : pathlib.Path
        Where to save the PNG figure.
    max_tasks : int, optional
        If provided, limit to the first N tasks (alphabetical).
    """
    tasks = sorted(df["task"].unique())
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    plt.figure(figsize=(10, 5))
    for t in tasks:
        d = df[df["task"] == t]
        plt.plot(d["x"], d["y"], label=t)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_per_task_grid(
    df: pd.DataFrame,
    title: str,
    outfile: Path,
    max_tasks: Optional[int] = None,
    cols: int = 4,
) -> None:
    """
    Plot each task in its own small axes on a grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with columns ['x','y','task'].
    title : str
        Super-title for the grid.
    outfile : pathlib.Path
        Where to save the PNG figure.
    max_tasks : int, optional
        If provided, limit to the first N tasks (alphabetical).
    cols : int, default=4
        Number of columns in the grid layout.

    Notes
    -----
    The grid size automatically adapts to the number of tasks shown.
    """
    tasks = sorted(df["task"].unique())
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    n = len(tasks)
    if n == 0:
        raise ValueError("No tasks found to plot.")

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for idx, t in enumerate(tasks):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        d = df[df["task"] == t]
        ax.plot(d["x"], d["y"])
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Hide any unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


# ----------------------------- CLI ------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize synthetic datasets (overlay and optional per-task grids)."
    )
    p.add_argument(
        "--dataset",
        choices=["multisine", "polynomial", "all"],
        default="all",
        help="Which dataset to visualize."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/interim"),
        help="Root directory containing dataset CSVs."
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory to save output figures."
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="If CSV includes a 'split' column, filter to this split."
    )
    p.add_argument(
        "--per-task",
        action="store_true",
        help="Also produce a per-task grid figure."
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit the number of tasks shown (useful for large T)."
    )
    p.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Columns in per-task grid (only if --per-task)."
    )
    return p


def main():
    args = build_argparser().parse_args()
    outdir: Path = args.outdir

    # Visualize Multi-Sine
    if args.dataset in ("multisine", "all"):
        df_ms = load_dataset("multisine", data_root=args.data_root, split=args.split)
        plot_overlay(
            df_ms,
            title="Multi-Sine Regression Tasks",
            outfile=outdir / "multisine_overlay.png",
            max_tasks=args.max_tasks,
        )
        if args.per_task:
            plot_per_task_grid(
                df_ms,
                title="Multi-Sine: Per-Task Curves",
                outfile=outdir / "multisine_per_task.png",
                max_tasks=args.max_tasks,
                cols=args.grid_cols,
            )

    # Visualize Polynomial
    if args.dataset in ("polynomial", "all"):
        df_pf = load_dataset("polynomial", data_root=args.data_root, split=args.split)
        plot_overlay(
            df_pf,
            title="Polynomial Family Tasks",
            outfile=outdir / "polynomial_overlay.png",
            max_tasks=args.max_tasks,
        )
        if args.per_task:
            plot_per_task_grid(
                df_pf,
                title="Polynomial: Per-Task Curves",
                outfile=outdir / "polynomial_per_task.png",
                max_tasks=args.max_tasks,
                cols=args.grid_cols,
            )

    print(f"Figures saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
