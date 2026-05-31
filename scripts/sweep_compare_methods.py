#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import queue
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml


ALE_METHOD_NAMES = {"ale_frechet", "ale", "ours"}
L2_ACTIVE_METHOD_NAMES = {"ale_frechet"}
COMMON_SWEEP_PREFIXES = ("optim.", "regularization.", "loss.", "errors.")
ALE_ONLY_SWEEP_PREFIXES = ("similarity.", "ale.")
ALE_ONLY_SWEEP_KEYS = {"ale_update_every", "shared_input_data", "ale_device"}
ALE_COMPONENT_SWEEP_PREFIXES = ("component_ablation.",)
L2_SWEEP_KEY = "regularization.l2"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_for_json(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def resolve_logging_root(global_cfg: Dict[str, Any], default: str = "./results") -> Path:
    return Path(global_cfg.get("logging_dir", global_cfg.get("log_dir", default)))


def _to_list(x: Any) -> List[Any]:
    if x is None:
        return [None]
    if isinstance(x, list):
        return x
    return [x]


def split_filter_values(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.append(item)
    return out


def parse_positive_int_filter(values: Sequence[str] | None, name: str) -> set[int]:
    out: set[int] = set()
    for value in split_filter_values(values):
        try:
            parsed = int(value)
        except ValueError as e:
            raise ValueError(f"{name} expects positive integer values; got {value!r}") from e
        if parsed <= 0:
            raise ValueError(f"{name} expects positive integer values; got {value!r}")
        out.add(parsed)
    return out


def parse_optional_positive_int(value: str | None, name: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except ValueError as e:
        raise ValueError(f"{name} expects a positive integer; got {value!r}") from e
    if parsed <= 0:
        raise ValueError(f"{name} expects a positive integer; got {value!r}")
    return parsed


def _norm_scalar(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v):
            return None
        return float(v)
    return v


def _combo_key(combo: Dict[str, Any]) -> Tuple[Any, ...]:
    items = []
    for k in sorted(combo.keys()):
        v = combo[k]
        if isinstance(v, dict):
            items.append((k, tuple((kk, _norm_scalar(vv)) for kk, vv in sorted(v.items()))))
        elif isinstance(v, list):
            items.append((k, tuple(_norm_scalar(x) for x in v)))
        else:
            items.append((k, _norm_scalar(v)))
    return tuple(items)


def valid_combos_from_keys(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values_product = itertools.product(*[grid[k] for k in keys])

    combos: List[Dict[str, Any]] = []
    seen = set()

    for values in values_product:
        combo = dict(zip(keys, values))

        sim_update = combo.get("similarity.update_every", None)
        sim_keep = combo.get("similarity.keep_epochs", None)
        reg_l2 = combo.get("regularization.l2", 0.0)

        one_none = (sim_update is None) or (sim_keep is None)
        both_none = sim_update is None and sim_keep is None

        if one_none or both_none:
            combo["similarity.update_every"] = None
            combo["similarity.keep_epochs"] = None
            if "regularization.l2" in combo:
                combo["regularization.l2"] = 0.0
        elif "regularization.l2" in combo:
            combo["regularization.l2"] = float(reg_l2)

        if not one_none and combo["similarity.keep_epochs"] > combo["similarity.update_every"]:
            combo["similarity.keep_epochs"] = combo["similarity.update_every"]

        key = _combo_key(combo)
        if key not in seen:
            seen.add(key)
            combos.append(combo)

    return combos


def _flatten_sweep_tree(
    obj: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, List[Any]]:
    flat: Dict[str, List[Any]] = {}

    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            nested = _flatten_sweep_tree(value, prefix=full_key)
            flat.update(nested)
        else:
            flat[full_key] = _to_list(value)

    return flat


def expand_sweep_config(sweep_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata_keys = {"name", "description", "notes", "seeds", "baselines", "constraints"}

    if "values" in sweep_cfg:
        raw_values = sweep_cfg.get("values", {})
        if not isinstance(raw_values, dict):
            raise ValueError("'values' must be a dictionary in the sweep config.")
        flat_grid = _flatten_sweep_tree(raw_values)
    else:
        raw_top = {k: v for k, v in sweep_cfg.items() if k not in metadata_keys}
        flat_grid = _flatten_sweep_tree(raw_top)

    if not flat_grid:
        raise ValueError("No sweep values found in the sweep config.")

    return valid_combos_from_keys(flat_grid)


def is_ale_method(method_name: str) -> bool:
    return method_name.lower() in ALE_METHOD_NAMES


def combo_applies_to_method(method_name: str, dotted_key: str) -> bool:
    if dotted_key == L2_SWEEP_KEY:
        return method_name.lower() in L2_ACTIVE_METHOD_NAMES
    if dotted_key.startswith(COMMON_SWEEP_PREFIXES):
        return True
    if is_ale_method(method_name):
        return (
            dotted_key.startswith(ALE_ONLY_SWEEP_PREFIXES)
            or dotted_key.startswith(ALE_COMPONENT_SWEEP_PREFIXES)
            or dotted_key in ALE_ONLY_SWEEP_KEYS
        )
    return not (
        dotted_key.startswith(ALE_ONLY_SWEEP_PREFIXES)
        or dotted_key.startswith(ALE_COMPONENT_SWEEP_PREFIXES)
        or dotted_key in ALE_ONLY_SWEEP_KEYS
    )


def filter_combo_for_method(method_name: str, combo: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in combo.items()
        if combo_applies_to_method(method_name, k)
    }


def build_method_specific_combos(method_name: str, combos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    method_combos: List[Dict[str, Any]] = []
    seen = set()
    for combo in combos:
        filtered = filter_combo_for_method(method_name, combo)
        key = _combo_key(filtered)
        if key not in seen:
            seen.add(key)
            method_combos.append(filtered)
    return method_combos


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_combo_to_method_cfg(method_cfg: Dict[str, Any], combo: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(method_cfg))
    for k, v in combo.items():
        set_nested(out, k, v)
    return out


def combo_to_tag(combo: Dict[str, Any]) -> str:
    if not combo:
        return "default"
    parts = []
    for k in sorted(combo.keys()):
        v = combo[k]
        key = k.replace(".", "-")
        sval = "none" if v is None else str(v).replace("/", "_").replace(" ", "")
        parts.append(f"{key}_{sval}")
    return "__".join(parts)


def read_log_tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    tail: deque[str] = deque(maxlen=max_lines)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            tail.append(line.rstrip("\n"))
    return "\n".join(tail)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def format_task_progress(task: Dict[str, Any], total_runs: int) -> str:
    return (
        f"[{task['task_index']}/{total_runs}] {task['method_name']} "
        f"combo={task['combo_idx']:03d} seed={task['seed']}"
    )


def detect_cuda_devices() -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [
            item.strip()
            for item in visible.split(",")
            if item.strip() and item.strip() != "-1"
        ]
        return devices

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []

    if proc.returncode != 0:
        return []

    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def set_cuda_device_for_run(
    global_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
    cuda_device: str | None,
) -> None:
    if cuda_device is None:
        return

    device_pref = str(global_cfg.get("device", "auto"))
    if device_pref in {"auto", "cuda"} or device_pref.startswith("cuda:"):
        global_cfg["device"] = "cuda"

    ale_pref = method_cfg.get("ale_device", None)
    if ale_pref is None or str(ale_pref) in {"auto", "cuda"} or str(ale_pref).startswith("cuda:"):
        method_cfg["ale_device"] = "cuda"


def run_one(
    python_exec: str,
    train_module: str,
    global_config: Path,
    dataset_config: Path,
    ale_model_config: Path | None,
    baseline_model_config: Path | None,
    method_config: Path,
    baselines: Sequence[str],
    out_dir: Path,
    run_tag: str,
    log_path: Path,
    cuda_device: str | None = None,
) -> int:
    cmd = [
        python_exec,
        "-m",
        train_module,
        "--global-config", str(global_config),
        "--dataset-config", str(dataset_config),
        "--method-config", str(method_config),
        "--out-dir", str(out_dir),
        "--run-tag", run_tag,
        "--no-aggregate",
    ]

    if ale_model_config is not None:
        cmd += ["--ale-model-config", str(ale_model_config)]
    if baseline_model_config is not None:
        cmd += ["--baseline-model-config", str(baseline_model_config)]

    for b in baselines:
        cmd += ["--baseline", b]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"[sweep] parent_pid={os.getpid()} "
            f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')} "
            f"run_tag={run_tag}\n"
        )
        log_file.flush()
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return int(proc.returncode)


def run_sweep_task(
    task: Dict[str, Any],
    gpu_pool: "queue.Queue[str | None]",
    progress_queue: "queue.Queue[Dict[str, Any]] | None" = None,
) -> Dict[str, Any]:
    cuda_device = gpu_pool.get()
    start_time = time.monotonic()
    try:
        if progress_queue is not None:
            progress_queue.put(
                {
                    "event": "started",
                    "run_tag": task["run_tag"],
                    "task_index": task["task_index"],
                    "method": task["method_name"],
                    "combo_index": task["combo_idx"],
                    "seed": task["seed"],
                    "gpu": cuda_device,
                    "time": start_time,
                }
            )

        method_cfg = apply_combo_to_method_cfg(task["base_method_cfg"], task["combo"])

        run_global_cfg = json.loads(json.dumps(task["global_cfg"]))
        run_global_cfg["seed"] = int(task["seed"])
        set_cuda_device_for_run(run_global_cfg, method_cfg, cuda_device)

        run_global_path = task["temp_cfg_dir"] / f"{task['run_tag']}__global.yaml"
        run_method_path = task["temp_cfg_dir"] / f"{task['run_tag']}__method.yaml"

        dump_yaml(run_global_cfg, run_global_path)
        dump_yaml(method_cfg, run_method_path)

        log_path = task["log_dir"] / f"{task['run_tag']}.log"
        returncode = run_one(
            python_exec=task["python_exec"],
            train_module=task["train_module"],
            global_config=run_global_path,
            dataset_config=task["dataset_config"],
            ale_model_config=task["ale_model_config"],
            baseline_model_config=task["baseline_model_config"],
            method_config=run_method_path,
            baselines=[task["method_name"]],
            out_dir=task["raw_run_dir"],
            run_tag=task["run_tag"],
            log_path=log_path,
            cuda_device=cuda_device,
        )

        run_json = task["raw_run_dir"] / f"{task['run_tag']}.json"
        base_payload = {
            "method": task["method_name"],
            "combo_index": task["combo_idx"],
            "combo_tag": task["combo_tag"],
            "seed": task["seed"],
            "run_tag": task["run_tag"],
            "gpu": cuda_device,
            "elapsed_seconds": round(time.monotonic() - start_time, 3),
            "log_file": str(log_path),
        }

        if returncode != 0:
            return {
                "status": "failed",
                "failure": {
                    **base_payload,
                    "returncode": returncode,
                    "log_tail": read_log_tail(log_path),
                },
            }

        if not run_json.exists():
            return {
                "status": "failed",
                "failure": {
                    **base_payload,
                    "returncode": None,
                    "log_tail": read_log_tail(log_path),
                    "stderr": f"Expected run file not found: {run_json}",
                },
            }

        try:
            records = json.loads(run_json.read_text(encoding="utf-8"))
        except Exception as e:
            return {
                "status": "failed",
                "failure": {
                    **base_payload,
                    "returncode": None,
                    "log_tail": read_log_tail(log_path),
                    "stderr": f"Could not parse run JSON: {e}",
                },
            }

        enriched_records: List[Dict[str, Any]] = []
        for r in records:
            rr = dict(r)
            rr["sweep_combo"] = task["combo"]
            rr["combo_index"] = task["combo_idx"]
            rr["combo_tag"] = task["combo_tag"]
            rr["sweep_method"] = task["method_name"]
            enriched_records.append(rr)

        sweep_run_json = task["sweep_run_dir"] / f"{task['run_tag']}.json"
        write_json(sweep_run_json, enriched_records)

        return {
            "status": "ok",
            "completed": {
                **base_payload,
                "raw_run_json": str(run_json),
                "sweep_run_json": str(sweep_run_json),
            },
        }
    finally:
        gpu_pool.put(cuda_device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-config", required=True)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--ale-model-config", default=None)
    ap.add_argument("--baseline-model-config", default=None)
    ap.add_argument("--method-config", required=True)
    ap.add_argument("--sweep-config", required=True)
    ap.add_argument("--train-script", default="scripts/train_compare_methods.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--keep-temp-configs", action="store_true")
    ap.add_argument("--train-module", default="scripts.train_compare_methods")
    ap.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated GPU ids to use for parallel sweep runs. "
            "Defaults to all GPUs visible through CUDA_VISIBLE_DEVICES or nvidia-smi."
        ),
    )
    ap.add_argument(
        "--max-workers",
        default=None,
        help=(
            "Maximum number of concurrent training subprocesses. "
            "Defaults to one worker per selected GPU, or one CPU worker."
        ),
    )
    ap.add_argument(
        "--progress-interval",
        type=float,
        default=300.0,
        help="Seconds between progress summaries while jobs are running. Use 0 to disable periodic summaries.",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Optional method subset to execute. Accepts space-separated and/or comma-separated values, "
            "for example: --methods ale_frechet hard."
        ),
    )
    ap.add_argument(
        "--combo-indices",
        nargs="+",
        default=None,
        help=(
            "Optional method-specific combo indices to execute, using the printed combo numbers "
            "such as 1 or 003. Accepts space-separated and/or comma-separated values."
        ),
    )
    ap.add_argument(
        "--combo-tags",
        nargs="+",
        default=None,
        help=(
            "Optional method-specific combo tags to execute, for example "
            "optim-lr_0.001__similarity-keep_epochs_10."
        ),
    )
    ap.add_argument(
        "--list-combos",
        action="store_true",
        help="Print selected method-specific combo indices/tags and exit without running training.",
    )
    args = ap.parse_args()

    global_cfg = load_yaml(args.global_config)
    dataset_cfg = load_yaml(args.dataset_config)
    base_method_cfg = load_yaml(args.method_config)
    sweep_cfg = load_yaml(args.sweep_config)

    logging_root = resolve_logging_root(global_cfg, default="./results")
    out_dir = Path(args.out_dir) if args.out_dir else logging_root / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_cfg_dir = out_dir / "generated_method_configs"
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)

    raw_run_dir = out_dir / "raw_runs"
    raw_run_dir.mkdir(parents=True, exist_ok=True)

    sweep_run_dir = out_dir / "sweep_runs"
    sweep_run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_combos = expand_sweep_config(sweep_cfg)
    seeds = _to_list(sweep_cfg.get("seeds", global_cfg.get("seed", 0)))
    configured_baselines = [str(b) for b in _to_list(sweep_cfg.get("baselines", ["ale_frechet"]))]
    method_filter = [m.lower() for m in split_filter_values(args.methods)]
    if method_filter:
        configured_lookup = {method.lower(): method for method in configured_baselines}
        missing_methods = [method for method in method_filter if method not in configured_lookup]
        if missing_methods:
            raise ValueError(
                f"Requested method(s) not found in sweep config: {missing_methods}. "
                f"Available methods: {configured_baselines}"
            )
        baselines = [method for method in configured_baselines if method.lower() in set(method_filter)]
    else:
        baselines = configured_baselines

    combo_index_filter = parse_positive_int_filter(args.combo_indices, "--combo-indices")
    combo_tag_filter = set(split_filter_values(args.combo_tags))
    cuda_devices = (
        [item.strip() for item in str(args.gpus).split(",") if item.strip()]
        if args.gpus
        else detect_cuda_devices()
    )
    requested_max_workers = parse_optional_positive_int(args.max_workers, "--max-workers")
    worker_devices: List[str | None]
    if cuda_devices:
        if requested_max_workers is None:
            worker_devices = list(cuda_devices)
        else:
            worker_devices = [
                cuda_devices[idx % len(cuda_devices)]
                for idx in range(requested_max_workers)
            ]
    else:
        worker_devices = [None] * (requested_max_workers or 1)
    max_workers = len(worker_devices)

    dataset = dataset_cfg.get("name", "dataset")
    combos_by_method_all = {
        method_name: build_method_specific_combos(method_name, all_combos)
        for method_name in baselines
    }
    combos_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method_name, method_combos in combos_by_method_all.items():
        selected: List[Dict[str, Any]] = []
        for combo_idx, combo in enumerate(method_combos, start=1):
            combo_tag = combo_to_tag(combo)
            if combo_index_filter and combo_idx not in combo_index_filter:
                continue
            if combo_tag_filter and combo_tag not in combo_tag_filter:
                continue
            selected.append(
                {
                    "combo_idx": combo_idx,
                    "combo": combo,
                    "combo_tag": combo_tag,
                }
            )
        combos_by_method[method_name] = selected

    print(f"Dataset: {dataset}")
    print(f"Sweep controller PID: {os.getpid()}")
    print(f"Baselines: {baselines}")
    print(f"Seeds: {seeds}")
    if method_filter:
        print(f"Method filter: {method_filter}")
    if combo_index_filter:
        print(f"Combo index filter: {sorted(combo_index_filter)}")
    if combo_tag_filter:
        print(f"Combo tag filter: {sorted(combo_tag_filter)}")
    if cuda_devices:
        print(f"Selected CUDA devices: {', '.join(cuda_devices)}")
    else:
        print("Selected CUDA devices: none; CPU/no CUDA GPU detected")
    print(f"Concurrent training subprocesses: {max_workers}")
    print(f"Worker device slots: {worker_devices}")
    print(f"Raw sweep combinations: {len(all_combos)}")
    for method_name in baselines:
        print(
            f"  - {method_name}: {len(combos_by_method[method_name])}/"
            f"{len(combos_by_method_all[method_name])} method-specific combinations selected"
        )

    if args.list_combos:
        available = {}
        print("\nSelected method-specific combos:")
        for method_name in baselines:
            entries = []
            print(f"\n[{method_name}]")
            if not combos_by_method[method_name]:
                print("  (none)")
            for combo_entry in combos_by_method[method_name]:
                entry = {
                    "combo_index": int(combo_entry["combo_idx"]),
                    "combo_tag": str(combo_entry["combo_tag"]),
                    "combo": combo_entry["combo"],
                }
                entries.append(entry)
                print(
                    f"  combo={entry['combo_index']:03d} "
                    f"tag={entry['combo_tag']} "
                    f"values={entry['combo']}"
                )
            available[method_name] = entries
        write_json(out_dir / "available_combos.json", available)
        print(f"\nWrote: {out_dir / 'available_combos.json'}")
        return

    failures: List[Dict[str, Any]] = []
    completed_runs: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []

    for method_name in baselines:
        method_combos = combos_by_method[method_name]
        for combo_entry in method_combos:
            combo_idx = int(combo_entry["combo_idx"])
            combo = combo_entry["combo"]
            combo_tag = str(combo_entry["combo_tag"])

            for seed in seeds:
                run_tag = (
                    f"{dataset}__{method_name}__combo_{combo_idx:03d}"
                    f"__seed_{seed}__{combo_tag}"
                )
                tasks.append(
                    {
                        "python_exec": args.python,
                        "train_module": args.train_module,
                        "global_cfg": global_cfg,
                        "dataset_config": Path(args.dataset_config),
                        "ale_model_config": Path(args.ale_model_config) if args.ale_model_config else None,
                        "baseline_model_config": Path(args.baseline_model_config) if args.baseline_model_config else None,
                        "base_method_cfg": base_method_cfg,
                        "method_name": method_name,
                        "combo": combo,
                        "combo_idx": combo_idx,
                        "combo_tag": combo_tag,
                        "seed": seed,
                        "run_tag": run_tag,
                        "temp_cfg_dir": temp_cfg_dir,
                        "raw_run_dir": raw_run_dir,
                        "sweep_run_dir": sweep_run_dir,
                        "log_dir": log_dir,
                    }
                )

    total_runs = len(tasks)
    if total_runs == 0:
        details = {
            method_name: [
                {
                    "combo_index": idx,
                    "combo_tag": combo_to_tag(combo),
                    "combo": combo,
                }
                for idx, combo in enumerate(combos_by_method_all[method_name], start=1)
            ]
            for method_name in baselines
        }
        write_json(out_dir / "available_combos.json", details)
        raise ValueError(
            "No sweep runs matched the requested filters. "
            f"Wrote available method-specific combos to {out_dir / 'available_combos.json'}."
        )
    for task_index, task in enumerate(tasks, start=1):
        task["task_index"] = task_index

    print(f"\nScheduled runs: {total_runs}")
    gpu_pool: queue.Queue[str | None] = queue.Queue()
    for device in worker_devices:
        gpu_pool.put(device)

    progress_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    running_runs: Dict[str, Dict[str, Any]] = {}
    finished_run_tags: set[str] = set()
    sweep_start_time = time.monotonic()
    last_progress_print = sweep_start_time

    def drain_progress_events() -> None:
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue.Empty:
                break

            if event.get("event") != "started":
                continue

            if event["run_tag"] in finished_run_tags:
                continue

            running_runs[event["run_tag"]] = event
            queued = total_runs - len(completed_runs) - len(failures) - len(running_runs)
            gpu = event.get("gpu")
            gpu_msg = f" gpu={gpu}" if gpu is not None else ""
            print(
                f"  START [{event['task_index']}/{total_runs}] "
                f"{event['method']} combo={event['combo_index']:03d} seed={event['seed']}"
                f"{gpu_msg} | running={len(running_runs)} queued={max(0, queued)}",
                flush=True,
            )

    def maybe_print_progress(force: bool = False) -> None:
        nonlocal last_progress_print
        if args.progress_interval <= 0 and not force:
            return

        now = time.monotonic()
        if not force and now - last_progress_print < args.progress_interval:
            return

        done = len(completed_runs) + len(failures)
        queued = total_runs - done - len(running_runs)
        print(
            "  PROGRESS "
            f"done={done}/{total_runs} ok={len(completed_runs)} failed={len(failures)} "
            f"running={len(running_runs)} queued={max(0, queued)} "
            f"elapsed={format_duration(now - sweep_start_time)}",
            flush=True,
        )
        last_progress_print = now

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_sweep_task, task, gpu_pool, progress_queue): task
            for task in tasks
        }

        pending = set(future_to_task.keys())
        while pending:
            done_futures, pending = concurrent.futures.wait(
                pending,
                timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done_futures:
                task = future_to_task[future]
                prefix = format_task_progress(task, total_runs)
                finished_run_tags.add(task["run_tag"])
                running_runs.pop(task["run_tag"], None)
                elapsed_msg = ""
                try:
                    result = future.result()
                except Exception as e:
                    log_path = log_dir / f"{task['run_tag']}.log"
                    failures.append(
                        {
                            "method": task["method_name"],
                            "combo_index": task["combo_idx"],
                            "combo_tag": task["combo_tag"],
                            "seed": task["seed"],
                            "run_tag": task["run_tag"],
                            "returncode": None,
                            "log_file": str(log_path),
                            "log_tail": read_log_tail(log_path),
                            "stderr": f"Sweep worker failed before completing run: {e}",
                        }
                    )
                    print(f"  END {prefix} FAILED (worker error)", flush=True)
                    maybe_print_progress(force=True)
                    continue

                if result["status"] != "ok":
                    failures.append(result["failure"])
                    gpu = result["failure"].get("gpu")
                    gpu_msg = f" gpu={gpu}" if gpu is not None else ""
                    if "elapsed_seconds" in result["failure"]:
                        elapsed_msg = f" elapsed={format_duration(result['failure']['elapsed_seconds'])}"
                    print(f"  END {prefix}{gpu_msg}{elapsed_msg} FAILED", flush=True)
                    maybe_print_progress(force=True)
                    continue

                completed_runs.append(result["completed"])
                gpu = result["completed"].get("gpu")
                gpu_msg = f" gpu={gpu}" if gpu is not None else ""
                if "elapsed_seconds" in result["completed"]:
                    elapsed_msg = f" elapsed={format_duration(result['completed']['elapsed_seconds'])}"
                print(f"  END {prefix}{gpu_msg}{elapsed_msg} OK", flush=True)
                maybe_print_progress(force=True)

            drain_progress_events()
            maybe_print_progress()

        drain_progress_events()
        if total_runs == 0:
            maybe_print_progress(force=True)

    completed_runs = sorted(
        completed_runs,
        key=lambda r: (str(r["method"]), int(r["combo_index"]), int(r["seed"])),
    )
    failures = sorted(
        failures,
        key=lambda r: (str(r["method"]), int(r["combo_index"]), int(r["seed"])),
    )

    write_json(out_dir / "completed_runs.json", completed_runs)
    write_json(out_dir / "failures.json", failures)

    manifest = {
        "dataset": dataset_cfg.get("name", "dataset"),
        "baselines": list(baselines),
        "configured_baselines": list(configured_baselines),
        "seeds": list(seeds),
        "raw_n_combos": len(all_combos),
        "n_combos_by_method": {k: len(v) for k, v in combos_by_method.items()},
        "raw_n_combos_by_method": {k: len(v) for k, v in combos_by_method_all.items()},
        "method_filter": list(method_filter),
        "combo_index_filter": sorted(combo_index_filter),
        "combo_tag_filter": sorted(combo_tag_filter),
        "n_scheduled_runs": total_runs,
        "n_completed_runs": len(completed_runs),
        "n_failures": len(failures),
        "parallel_run_workers": max_workers,
        "requested_max_workers": requested_max_workers,
        "progress_interval": args.progress_interval,
        "cuda_devices": list(cuda_devices),
        "worker_devices": list(worker_devices),
        "controller_pid": os.getpid(),
        "elapsed_seconds": round(time.monotonic() - sweep_start_time, 3),
        "out_dir": str(out_dir),
        "raw_run_dir": str(raw_run_dir),
        "sweep_run_dir": str(sweep_run_dir),
        "log_dir": str(log_dir),
        "aggregate_after_run": (
            "python -m scripts.aggregate_compare_results "
            f"--run-dir {sweep_run_dir} --out-dir {out_dir}"
        ),
    }
    write_json(out_dir / "manifest.json", manifest)

    if not args.keep_temp_configs:
        for p in temp_cfg_dir.glob("*.yaml"):
            p.unlink(missing_ok=True)
        try:
            temp_cfg_dir.rmdir()
        except OSError:
            pass

    print("\nSaved:")
    print(f"  {out_dir / 'completed_runs.json'}")
    print(f"  {out_dir / 'failures.json'}")
    print(f"  {out_dir / 'manifest.json'}")
    print(f"  raw runs in {raw_run_dir}")
    print(f"  enriched sweep runs in {sweep_run_dir}")
    print("\nBuild aggregate files after all executions with:")
    print(f"  python -m scripts.aggregate_compare_results --run-dir {sweep_run_dir} --out-dir {out_dir}")

    if failures:
        print(f"\nCompleted with {len(failures)} failed runs.")
    else:
        print("\nCompleted without failures.")


if __name__ == "__main__":
    main()
