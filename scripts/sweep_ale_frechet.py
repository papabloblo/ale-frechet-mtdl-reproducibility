#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

import json


def canonical_config_hash(cfg: dict) -> str:
    """
    Create a stable hashable representation of the config.
    """
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(obj: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def iter_sweep_axes(
    obj: Any,
    prefix: Tuple[str, ...] = (),
    *,
    exclude_top_keys: set[str] | None = None,
) -> List[Tuple[Tuple[str, ...], List[Any]]]:
    """
    Find all list-valued leaves in a nested dict structure.
    Returns: [((path, ...), [values...]), ...]
    """
    exclude_top_keys = exclude_top_keys or set()
    axes: List[Tuple[Tuple[str, ...], List[Any]]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if len(prefix) == 0 and k in exclude_top_keys:
                continue
            axes.extend(iter_sweep_axes(v, prefix + (str(k),), exclude_top_keys=exclude_top_keys))
        return axes

    # Treat only plain lists as sweep axes (not strings)
    if isinstance(obj, list):
        axes.append((prefix, obj))
        return axes

    return axes


def deep_set(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur: Dict[str, Any] = d
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def get_in(d: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def sanitize_value_for_tag(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        # keep short but stable
        return f"{v:.6g}"
    return str(v).replace("/", "-").replace(" ", "")


def make_tag(assignment: List[Tuple[Tuple[str, ...], Any]]) -> str:
    parts = []
    for path, val in assignment:
        key = "_".join(path)
        parts.append(f"{key}={sanitize_value_for_tag(val)}")
    return "__".join(parts) if parts else "default"


def apply_special_constraints(cfg: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    """
    Returns True if cfg is valid, False if it should be skipped.
    Keeps your original logic but only applies when the relevant fields exist.
    """
    # Your tie flag (still supported)
    tie = bool(constraints.get("tie_ale_and_similarity_updates", False))

    ue_path = ("similarity", "update_every")
    keep_path = ("similarity", "keep_epochs")
    l2_path = ("regularization", "l2")

    update_every = get_in(cfg, ue_path)
    keep_epochs = get_in(cfg, keep_path)
    l2 = get_in(cfg, l2_path)

    # If similarity keys are absent, don't enforce these constraints
    if update_every is None and keep_epochs is None and l2 is None:
        return True

    # Mirror your behavior: if either update_every or keep_epochs is None -> force both None and l2=0.0
    if (update_every is None) or (keep_epochs is None):
        deep_set(cfg, ue_path, None)
        deep_set(cfg, keep_path, None)
        deep_set(cfg, l2_path, 0.0)
        update_every = None
        keep_epochs = None
        l2 = 0.0

    # If both not None, require update_every >= keep_epochs
    if update_every is not None and keep_epochs is not None:
        if update_every < keep_epochs:
            return False

    # Apply tie if requested (and if update_every exists in cfg)
    if tie:
        deep_set(cfg, ("ale_update_every",), get_in(cfg, ue_path))

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-config", required=True)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--base-method-config", required=True)
    ap.add_argument("--sweep-config", required=True)
    ap.add_argument("--outdir", default="configs/methods/sweeps/ale_frechet")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = load_yaml(Path(args.base_method_config))
    sweep = load_yaml(Path(args.sweep_config))

    constraints = sweep.get("constraints", {}) if isinstance(sweep, dict) else {}

    axes = iter_sweep_axes(sweep, exclude_top_keys={"constraints"})
    # Sort axes for stable ordering/tags
    axes.sort(key=lambda x: x[0])

    if not axes:
        combos = [()]
        print(">>> No sweep axes found (no list-valued leaves). Running single config.")
    else:
        value_lists = [vals for _, vals in axes]
        combos = itertools.product(*value_lists)
        print(f">>> Sweep axes: {len(axes)}")
        for path, vals in axes:
            print(f"    - {'.'.join(path)}: {len(vals)} values")
        # We will iterate combos below; total size can be huge, so just compute a count safely:
        total = 1
        for _, vals in axes:
            total *= max(1, len(vals))
        print(f">>> Sweep size: {total} runs")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for combo in combos:
        cfg = deepcopy(base)

        assignment: List[Tuple[Tuple[str, ...], Any]] = []
        for (path, _vals), val in zip(axes, combo):
            deep_set(cfg, path, val)
            assignment.append((path, val))

        # Apply constraints
        if not apply_special_constraints(cfg, constraints):
            continue

        # Deduplicate AFTER constraints normalization
        cfg_hash = canonical_config_hash(cfg)
        if cfg_hash in seen:
            continue
        seen.add(cfg_hash)

        tag = make_tag(assignment)
        method_path = outdir / f"ale_frechet__{tag}.yaml"
        dump_yaml(cfg, method_path)

        cmd = [
            "python", "-m", "scripts.train_ale_frechet",
            "--global-config", args.global_config,
            "--dataset-config", args.dataset_config,
            "--model-config", args.model_config,
            "--method-config", str(method_path),
            "--tag", tag,
        ]

        print(">>>", " ".join(cmd))
        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
