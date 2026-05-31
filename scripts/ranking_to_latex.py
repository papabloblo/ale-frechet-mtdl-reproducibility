#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_COLS = [
    "rank",
    "method",
    "combo_index",
    "run_task_mean_mean",
    "run_task_std_mean",
    "run_task_gap_mean",
    "ranking_score",
]

DEFAULT_HEADERS = [
    "Rank",
    "Method",
    "Combo",
    "Avg.",
    "Std.",
    "Gap",
    "Score",
]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate a LaTeX table from method_ranking.csv"
    )
    ap.add_argument("--input", required=True, help="Path to method_ranking.csv")
    ap.add_argument("--output", default=None, help="Output .tex path")
    ap.add_argument("--top-k", type=int, default=10, help="Number of top rows to include")
    ap.add_argument(
        "--columns",
        default=",".join(DEFAULT_COLS),
        help="Comma-separated columns to include",
    )
    ap.add_argument(
        "--headers",
        default=",".join(DEFAULT_HEADERS),
        help="Comma-separated header names aligned with --columns",
    )
    ap.add_argument("--caption", default=None, help="Table caption")
    ap.add_argument("--label", default=None, help="LaTeX label")
    ap.add_argument(
        "--float-format",
        default="%.4f",
        help="Printf-style float format used in the table",
    )
    ap.add_argument(
        "--table-env",
        choices=["table", "table*", "none"],
        default="table",
        help="Wrap in a LaTeX floating table environment",
    )
    ap.add_argument(
        "--position",
        default="tbp",
        help="LaTeX float position, only used when table-env is not none",
    )
    ap.add_argument(
        "--use-booktabs",
        action="store_true",
        help="Use booktabs style rules",
    )
    ap.add_argument(
        "--bold-best",
        action="store_true",
        help="Bold the best value in performance/stability/score columns inside the displayed rows",
    )
    return ap



def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]



def _best_direction(col: str) -> str:
    name = col.lower()
    if "score" in name:
        return "max"
    return "min"



def _format_value(value, fmt: str) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, float):
        return fmt % value
    return str(value)



def _latex_escape(text: str) -> str:
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
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out



def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".tex")

    df = pd.read_csv(input_path)
    if df.empty:
        output_path.write_text("% Empty ranking table\n", encoding="utf-8")
        print(f"Input is empty. Wrote empty LaTeX file to {output_path}")
        return

    columns = _parse_csv_list(args.columns)
    headers = _parse_csv_list(args.headers)
    if len(columns) != len(headers):
        raise ValueError("--columns and --headers must have the same number of items")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {input_path}: {missing}")

    work = df.copy()
    if "rank" in work.columns:
        work = work.sort_values(by=["rank", "ranking_score" if "ranking_score" in work.columns else columns[0]], ascending=[True, False])
    work = work.head(int(args.top_k)).copy()
    work = work[columns]

    # Decide which visible columns can be bolded as "best"
    highlight_cols = [
        c for c in work.columns
        if c in {"run_task_mean_mean", "run_task_std_mean", "run_task_gap_mean", "ranking_score"}
    ]
    best_values = {}
    if args.bold_best:
        for col in highlight_cols:
            numeric = pd.to_numeric(work[col], errors="coerce")
            if numeric.notna().any():
                best_values[col] = numeric.max() if _best_direction(col) == "max" else numeric.min()

    # Build tabular content manually so bold formatting is robust
    align = []
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            align.append("r")
        else:
            align.append("l")
    colspec = "".join(align)

    lines: List[str] = []
    if args.table_env != "none":
        lines.append(f"\\begin{{{args.table_env}}}[{args.position}]")
        lines.append("\\centering")

    if args.caption:
        lines.append(f"\\caption{{{_latex_escape(args.caption)}}}")
    if args.label:
        lines.append(f"\\label{{{_latex_escape(args.label)}}}")

    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    if args.use_booktabs:
        lines.append("\\toprule")
    else:
        lines.append("\\hline")

    header_row = " & ".join(_latex_escape(h) for h in headers) + r" \\" 
    lines.append(header_row)
    if args.use_booktabs:
        lines.append("\\midrule")
    else:
        lines.append("\\hline")

    for _, row in work.iterrows():
        row_cells = []
        for col in work.columns:
            value = row[col]
            if pd.api.types.is_numeric_dtype(work[col]):
                cell = _format_value(value, args.float_format)
                if args.bold_best and col in best_values and pd.notna(value):
                    is_best = abs(float(value) - float(best_values[col])) < 1e-12
                    if is_best:
                        cell = rf"\textbf{{{cell}}}"
            else:
                cell = _latex_escape(_format_value(value, args.float_format))
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    if args.use_booktabs:
        lines.append("\\bottomrule")
    else:
        lines.append("\\hline")
    lines.append("\\end{tabular}")

    if args.table_env != "none":
        lines.append(f"\\end{{{args.table_env}}}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote LaTeX table to {output_path}")


if __name__ == "__main__":
    main()
