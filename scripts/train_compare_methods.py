#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from MultiTaskDeepLearning.data import MultitaskDatasetDF, MultitaskDataloader
from MultiTaskDeepLearning.loss import MultiTaskLoss, rmse_loss, mae_loss, mape_loss
from MultiTaskDeepLearning.trainer import MultiTaskTrainer
from MultiTaskDeepLearning.similarity.ale import MultiTaskALE
from MultiTaskDeepLearning.similarity.similarity import MultitaskSimilarity
from MultiTaskDeepLearning.baselines import (
    HardSharing,
    SingleTaskMLP,
    SoftSharing,
    MMoE,
    CrossStitch,
    PLE,
    MTAN,
)
from MultiTaskDeepLearning.tracking import Tracker

from scripts.train_ale_frechet import build_model


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(pref: str | None) -> torch.device:
    if pref in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def resolve_logging_root(global_cfg: Dict[str, Any], default: str = "./results") -> Path:
    return Path(global_cfg.get("logging_dir", global_cfg.get("log_dir", default)))


def resolve_trainer_cfg(global_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return global_cfg.get("trainer", {}) or {}


def resolve_early_stopping_epochs(global_cfg: Dict[str, Any], default: int = 20) -> int:
    trainer_cfg = resolve_trainer_cfg(global_cfg)
    return int(trainer_cfg.get("early_stopping_epochs", trainer_cfg.get("early_stopping", default)))


def resolve_print_each_epochs(global_cfg: Dict[str, Any], default: int = 1) -> int:
    trainer_cfg = resolve_trainer_cfg(global_cfg)
    return int(trainer_cfg.get("print_each", default))


def resolve_epochs(global_cfg: Dict[str, Any], default: int = 50) -> int:
    trainer_cfg = resolve_trainer_cfg(global_cfg)
    return int(trainer_cfg.get("epochs", default))


def resolve_ale_device(method_cfg: Dict[str, Any]) -> str:
    pref = method_cfg.get("ale_device", None)
    if pref in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(pref)


def sanitize_tag(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_.-")
    return s or "run"


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_loader_cfg(dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    loader_cfg = dataset_cfg.get("loader", {}) or {}
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", False))
    persistent_workers = bool(loader_cfg.get("persistent_workers", False))
    prefetch_factor = loader_cfg.get("prefetch_factor", 2 if num_workers > 0 else None)
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
    }


def build_loader(dataset, batch_size: int, shuffle: bool, loader_cfg: Dict[str, Any]) -> MultitaskDataloader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg["pin_memory"]),
        persistent_workers=bool(loader_cfg["persistent_workers"]),
    )
    if loader_cfg.get("prefetch_factor") is not None and int(loader_cfg["num_workers"]) > 0:
        kwargs["prefetch_factor"] = int(loader_cfg["prefetch_factor"])
    return MultitaskDataloader(dataset, **kwargs)

@torch.no_grad()
def evaluate_per_task(
    model: torch.nn.Module,
    dataloader: Optional[MultitaskDataloader],
    loss: MultiTaskLoss,
) -> Dict[str, Any]:
    if dataloader is None:
        return {"per_task": {}, "mean": {}, "n_obs": 0}

    model.eval()
    n_tasks = dataloader._dataset.n_tasks
    use_amp = bool(torch.cuda.is_available() and str(model.device).startswith("cuda"))

    totals: Dict[str, torch.Tensor] = {}
    totals["LOSS"] = torch.zeros(n_tasks, device=model.device)
    for k in loss.errors_dict.keys():
        totals[k] = torch.zeros(n_tasks, device=model.device)

    n_obs = 0
    for X, y in dataloader:
        X = X.to(model.device, non_blocking=True)
        y = y.to(model.device, non_blocking=True)
        with (torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()):
            y_pred = model(X)

        lp = loss.loss_per_task(y, y_pred).detach().view(-1)
        errs = loss.errors_per_task(y, y_pred)

        bsz = int(X.size(1))
        n_obs += bsz
        totals["LOSS"] += lp * bsz
        for k, v in errs.items():
            totals[k] += v.detach().view(-1) * bsz

    if n_obs == 0:
        return {"per_task": {}, "mean": {}, "n_obs": 0}

    per_task = {}
    mean = {}
    for k, v in totals.items():
        vals = (v / n_obs).detach().cpu()
        finite = torch.isfinite(vals)
        per_task[k] = [
            float(x) if bool(ok) else None
            for x, ok in zip(vals.tolist(), finite.tolist())
        ]
        mean[k] = float(vals[finite].mean().item()) if bool(finite.any()) else None
    return {"per_task": per_task, "mean": mean, "n_obs": int(n_obs)}


def build_dataloaders(dataset_cfg: Dict[str, Any]) -> Dict[str, MultitaskDataloader]:
    paths = dataset_cfg["paths"]
    train_path = paths["train"]
    val_path = paths.get("validation", None)
    test_path = paths["test"]

    task_id_col = dataset_cfg.get("task_id_col", "task")
    target_cols = dataset_cfg.get("target_cols", ["y_target"])
    feature_cols = dataset_cfg["feature_cols"]
    cols = [task_id_col] + target_cols + feature_cols
    numeric_cols = target_cols + feature_cols

    def _load_split_csv(path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path, usecols=cols)
        df[numeric_cols] = df[numeric_cols].astype(np.float32, copy=False)
        return df

    df_train = _load_split_csv(train_path)
    df_test = _load_split_csv(test_path)
    df_val = _load_split_csv(val_path) if val_path and Path(val_path).exists() else None

    ds_train = MultitaskDatasetDF(data=df_train, task_id=task_id_col, target_names=target_cols)
    ds_test = MultitaskDatasetDF(data=df_test, task_id=task_id_col, target_names=target_cols)
    ds_val = (
        MultitaskDatasetDF(data=df_val, task_id=task_id_col, target_names=target_cols)
        if df_val is not None
        else None
    )

    del df_train, df_test, df_val
    gc.collect()

    bs = dataset_cfg.get("batch", {})
    val_bs = int(bs.get("validation", bs.get("val", 128)))
    loader_cfg = resolve_loader_cfg(dataset_cfg)

    dl_train = build_loader(ds_train, batch_size=int(bs.get("train", 128)), shuffle=True, loader_cfg=loader_cfg)
    dl_test = build_loader(ds_test, batch_size=int(bs.get("test", 128)), shuffle=False, loader_cfg=loader_cfg)
    dl_val = build_loader(ds_val, batch_size=val_bs, shuffle=False, loader_cfg=loader_cfg) if ds_val else None
    dl_ale = build_loader(ds_train, batch_size=int(bs.get("ale", 256)), shuffle=False, loader_cfg=loader_cfg)

    dls = {"train": dl_train, "test": dl_test, "ale": dl_ale}
    if dl_val is not None:
        dls["val"] = dl_val
    return dls


class _DummySimilarity:
    def __init__(self, n_tasks: int, device: torch.device):
        self.n_tasks = n_tasks
        self.device = device

    def tasks_groups(self):
        return torch.ones(self.n_tasks, device=self.device), None


class _FixedTaskGraphSimilarity:
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        mode: str = "random",
        seed: int = 0,
    ):
        self.n_tasks = int(n_tasks)
        self.device = device
        self.mode = str(mode).lower()
        self.seed = int(seed)
        self.similarity_tasks_features = torch.zeros(
            (self.n_tasks, self.n_tasks, 1),
            device=self.device,
            dtype=torch.float32,
        )
        self._weights, self._groups = self._build_graph()

    def _build_graph(self) -> tuple[torch.Tensor, list[list[int]]]:
        if self.n_tasks <= 1:
            return torch.ones(1, device=self.device), [[0, 0]]

        rng = random.Random(self.seed)
        groups: list[list[int]] = []
        for task in range(self.n_tasks):
            candidates = [idx for idx in range(self.n_tasks) if idx != task]
            neighbor = rng.choice(candidates)
            groups.append([task, neighbor])
            self.similarity_tasks_features[task, neighbor, 0] = 1.0
            self.similarity_tasks_features[neighbor, task, 0] = 1.0

        return torch.ones(len(groups), device=self.device), groups

    def compute(self) -> None:
        return None

    def tasks_groups(self):
        return self._weights, self._groups


class _DenseTaskPairSimilarity:
    def __init__(self, n_tasks: int, device: torch.device):
        self.n_tasks = int(n_tasks)
        self.device = device
        self._groups = self._build_groups()
        self._weights = torch.full(
            (len(self._groups),),
            1.0 / max(1, len(self._groups)),
            device=self.device,
        )
        self.similarity_tasks_features = torch.ones(
            (self.n_tasks, self.n_tasks, 1),
            device=self.device,
            dtype=torch.float32,
        )

    def _build_groups(self) -> list[list[int]]:
        if self.n_tasks <= 1:
            return [[0, 0]]
        return [
            [left, right]
            for left in range(self.n_tasks)
            for right in range(left + 1, self.n_tasks)
        ]

    def compute(self) -> None:
        return None

    def tasks_groups(self):
        return self._weights, self._groups


def extract_times(tracker: Tracker) -> Dict[str, float]:
    times = {}
    for key in ["train", "validation", "test", "ale", "similarity"]:
        timing = tracker.track.get(key, {}).get("timing", None)
        if timing is not None:
            times[key] = timing.total_time()
    times["total_time"] = sum(times.values())
    return times


def build_baseline(
    name: str,
    n_tasks: int,
    input_dim: int,
    output_dim: int,
    model_cfg: Dict[str, Any],
    device: torch.device,
):
    name = name.lower()
    act = model_cfg.get("activation", "relu")
    drop = float(model_cfg.get("dropout", 0.0))

    if name == "soft":
        return SoftSharing(
            n_tasks=n_tasks,
            input_dim=input_dim,
            hidden=model_cfg.get("soft_hidden", [128, 128]),
            output_dim=output_dim,
            activation=act,
            dropout=drop,
            device=device,
        )

    if name in {"single_task", "single_task_mlp", "single-task", "single-task-mlp", "stl", "independent"}:
        return SingleTaskMLP(
            n_tasks=n_tasks,
            input_dim=input_dim,
            hidden=model_cfg.get("single_task_hidden", model_cfg.get("soft_hidden", [128, 128])),
            output_dim=output_dim,
            activation=act,
            dropout=drop,
            device=device,
        )

    if name == "hard":
        return HardSharing(
            n_tasks=n_tasks,
            input_dim=input_dim,
            trunk=model_cfg.get("hard_trunk", [128, 128]),
            head=model_cfg.get("hard_head", [64]),
            output_dim=output_dim,
            activation=act,
            dropout=drop,
            device=device,
        )

    if name == "mmoe":
        return MMoE(
            n_tasks=n_tasks,
            input_dim=input_dim,
            n_experts=int(model_cfg.get("n_experts", 8)),
            expert_hidden=model_cfg.get("expert_hidden", [128]),
            tower_hidden=model_cfg.get("tower_hidden", [64]),
            output_dim=output_dim,
            activation=act,
            dropout=drop,
            device=device,
        )

    if name in {"crossstitch", "cross_stitch"}:
        return CrossStitch(
            n_tasks=n_tasks,
            input_dim=input_dim,
            shared_dims=model_cfg.get("shared_dims", [128, 128]),
            output_dim=output_dim,
            activation=act,
            dropout=drop,
            init_identity=bool(model_cfg.get("init_identity", True)),
            device=device,
        )

    if name == "ple":
        return PLE(
            n_tasks=n_tasks,
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=int(model_cfg.get("ple_layers", 2)),
            n_shared_experts=int(model_cfg.get("ple_shared_experts", 4)),
            n_task_experts=int(model_cfg.get("ple_task_experts", 2)),
            expert_hidden=model_cfg.get("ple_expert_hidden", [128]),
            tower_hidden=model_cfg.get("ple_tower_hidden", [64]),
            activation=act,
            dropout=drop,
            device=device,
        )

    if name == "mtan":
        return MTAN(
            n_tasks=n_tasks,
            input_dim=input_dim,
            shared_dims=model_cfg.get("mtan_shared_dims", [128, 128]),
            output_dim=output_dim,
            tower_hidden=model_cfg.get("mtan_tower_hidden", [64]),
            activation=act,
            dropout=drop,
            device=device,
        )

    raise ValueError(f"Unknown baseline '{name}'.")


def _safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _summary_value(r: Dict[str, Any], split: str, metric: str) -> float:
    try:
        return float(r[split]["mean"].get(metric, float("nan")))
    except Exception:
        return float("nan")


def results_to_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metric_names = ["LOSS", "RMSE", "MAE", "MAPE"]
    rows = []
    for r in results:
        row = {
            "dataset": r["dataset"],
            "method": r["method"],
            "seed": r["seed"],
            "run_tag": r.get("run_tag"),
            "component_ablation_variant": r.get("component_ablation_variant"),
            "best_epoch": r["best_epoch"],
            "best_test_loss": r["best_test_loss"],
            "time_train": r.get("time_train"),
            "time_validation": r.get("time_validation"),
            "time_test": r.get("time_test"),
            "time_ale": r.get("time_ale"),
            "time_similarity": r.get("time_similarity"),
            "total_time": r.get("total_time"),
        }
        for split in ("train", "val", "test"):
            means = r.get(split, {}).get("mean", {})
            for m in metric_names:
                row[f"{split}_{m.lower()}"] = means.get(m, None)
        rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_for_json(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def append_aggregate_files(results: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results.json"
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError("results.json must contain a list")
    else:
        existing = []
    existing.extend(results)
    write_json(json_path, existing)

    csv_path = out_dir / "results.csv"
    new_rows = results_to_rows(results)

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(new_rows)
    df.to_csv(csv_path, index=False)


def main() -> List[Dict[str, Any]]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-config", required=True)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--ale-model-config", default=None)
    ap.add_argument("--baseline-model-config", default=None)
    ap.add_argument("--method-config", required=True)
    ap.add_argument("--baseline", action="append", default=[])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not append to shared results.json/results.csv; only write run-specific files.",
    )
    args = ap.parse_args()

    global_cfg = load_yaml(args.global_config)
    dataset_cfg = load_yaml(args.dataset_config)
    method_cfg = load_yaml(args.method_config)
    ale_model_cfg = load_yaml(args.ale_model_config) if args.ale_model_config else {}
    baseline_model_cfg = load_yaml(args.baseline_model_config) if args.baseline_model_config else {}

    device = resolve_device(global_cfg.get("device", "auto"))
    seed = int(global_cfg.get("seed", 0))
    deterministic = bool(global_cfg.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    dls = build_dataloaders(dataset_cfg)
    n_tasks = dls["train"]._dataloader.dataset.n_tasks
    input_dim = dls["train"]._dataloader.dataset.n_features
    output_dim = dls["train"]._dataloader.dataset.n_targets

    raw_l2 = float(method_cfg.get("regularization", {}).get("l2", 0.0))
    logging_root = resolve_logging_root(global_cfg)
    out_dir = Path(args.out_dir) if args.out_dir else logging_root / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = args.baseline or ["ale_frechet"]
    dataset_name = dataset_cfg.get("name", "dataset")
    #run_tag = sanitize_tag(args.run_tag or f"{dataset_name}__seed_{seed}")
    run_tag = args.run_tag

    results: List[Dict[str, Any]] = []

    for method_name in baselines:
        method_name_l = method_name.lower()
        sim_cfg: Dict[str, Any] = {}
        component_cfg = method_cfg.get("component_ablation", {}) or {}
        component_variant = str(component_cfg.get("variant", "full")).strip().lower()
        component_aliases = {
            "none": "full",
            "lambda_0": "no_similarity",
            "lambda=0": "no_similarity",
            "no_regularization": "no_similarity",
            "no_similarity_regularization": "no_similarity",
            "random": "random_graph",
        }
        component_variant = component_aliases.get(component_variant, component_variant)
        if component_variant not in {"full", "no_similarity", "random_graph"}:
            raise ValueError(
                "Unsupported component_ablation.variant="
                f"{component_variant!r}. Expected one of: full, no_similarity, random_graph."
            )
        soft_l2 = float(baseline_model_cfg.get("soft_l2", raw_l2))
        if method_name_l == "ale_frechet" and component_variant != "no_similarity":
            l2 = raw_l2
        elif method_name_l == "soft":
            l2 = soft_l2
        else:
            l2 = 0.0

        if method_name_l in {"ale_frechet", "ale", "ours"}:
            mcfg = ale_model_cfg
            if not mcfg:
                raise ValueError("For ale_frechet, provide --ale-model-config.")
            if not isinstance(mcfg.get("modules", None), list) or len(mcfg["modules"]) == 0:
                raise ValueError(
                    f"--ale-model-config must point to an ALE architecture YAML with a non-empty "
                    f"'modules' list. Got: {args.ale_model_config}"
                )

            model = build_model(mcfg, n_tasks=n_tasks, device=device)

            ale_cfg = method_cfg.get("ale", {})
            sim_cfg = method_cfg.get("similarity", {})
            if component_variant == "full":
                ale = MultiTaskALE(
                    model=model,
                    dataloader=dls["ale"],
                    n_tasks=n_tasks,
                    shared_input_data=bool(method_cfg.get("shared_input_data", False)),
                    n_features_out=int(ale_cfg.get("n_features_out", 1)),
                    num_intervals=int(ale_cfg.get("n_intervals", 30)),
                    device=resolve_ale_device(method_cfg),
                    epsilon=float(ale_cfg.get("epsilon", 1e-4)),
                    n_guess=int(ale_cfg.get("n_guess", 1000)),
                )
                sim = MultitaskSimilarity(ale_curves=ale)
            elif component_variant == "random_graph":
                ale = None
                sim = _FixedTaskGraphSimilarity(
                    n_tasks=n_tasks,
                    device=device,
                    mode="random",
                    seed=int(component_cfg.get("random_seed", seed)),
                )
            else:
                ale = None
                sim = None
        else:
            model = build_baseline(
                method_name_l,
                n_tasks,
                input_dim,
                output_dim,
                baseline_model_cfg,
                device,
            )
            ale = None
            if method_name_l == "soft":
                sim = _DenseTaskPairSimilarity(n_tasks=n_tasks, device=device)
            else:
                sim = _DummySimilarity(n_tasks=n_tasks, device=device)

        loss = MultiTaskLoss(
            model=model,
            loss_fn=rmse_loss,
            errors_fn={"RMSE": rmse_loss, "MAE": mae_loss, "MAPE": mape_loss},
            l2_penalty=l2,
        )
        if method_name_l == "soft":
            _, groups = sim.tasks_groups()
            loss.update_tasks_groups(groups)

        opt_cfg = method_cfg.get("optim", {})
        lr = float(opt_cfg.get("lr", 1e-3))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        trainer = MultiTaskTrainer(
            model=model,
            train_dataloader=dls["train"],
            test_dataloader=dls["test"],
            validation_dataloader=dls.get("val", None),
            optimizer=optimizer,
            loss=loss,
            ale=ale,
            multitask_similarity=sim,
            early_stopping_epochs=resolve_early_stopping_epochs(global_cfg, default=20),
            print_each_epochs=resolve_print_each_epochs(global_cfg, default=1),
            ale_each_epochs=int(method_cfg.get("ale_update_every", 1)),
            similarity_each_epochs=(
                int(sim_cfg.get("update_every", 1))
                if method_name_l in {"ale_frechet", "ale", "ours"}
                   and sim_cfg.get("update_every", None) is not None
                else None
            ),
            keep_similarity_epochs=(
                int(sim_cfg.get("keep_epochs", 0))
                if method_name_l in {"ale_frechet", "ale", "ours"}
                   and sim_cfg.get("keep_epochs", None) is not None
                else 0
            ),
            logging_dir=str(logging_root / method_name_l / run_tag),
            seed=seed,
            dataset_name=dataset_name,
            learning_type=method_name_l,
            architecture=method_name_l,
            train_batch_size=dls["train"]._dataloader.batch_size,
            test_batch_size=dls["test"]._dataloader.batch_size,
            ale_batch_size=dls["ale"]._dataloader.batch_size,
            amp=bool(global_cfg.get("amp", False)),
            amp_dtype=str(global_cfg.get("amp_dtype", "float16")),
        )

        epochs = resolve_epochs(global_cfg, default=50)
        trainer.train(epochs=epochs)

        best_model_info = trainer.get_best_model()
        if (
            best_model_info is not None
            and "test_metrics" in best_model_info
            and "RMSE" in best_model_info["test_metrics"]
        ):
            best = float(best_model_info["test_metrics"]["RMSE"])
        else:
            best = float("nan")

        best_epoch = int(getattr(trainer.tracking, "best_epoch", -1))

        if trainer.tracking.best_model_parameters is not None:
            model.load_state_dict(trainer.tracking.best_model_parameters)

        train_metrics = evaluate_per_task(model, dls["train"], loss)
        test_metrics = evaluate_per_task(model, dls["test"], loss)
        val_metrics = evaluate_per_task(model, dls.get("val", None), loss)
        elapsed_times = extract_times(trainer.tracking)

        results.append(
            {
                "dataset": dataset_name,
                "method": method_name_l,
                "seed": seed,
                "run_tag": run_tag,
                "best_test_loss": best,
                "best_epoch": best_epoch,
                "component_ablation_variant": component_variant,
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "time_train": elapsed_times.get("train", None),
                "time_validation": elapsed_times.get("validation", None),
                "time_test": elapsed_times.get("test", None),
                "time_ale": elapsed_times.get("ale", None),
                "time_similarity": elapsed_times.get("similarity", None),
                "total_time": elapsed_times.get("total_time", None),
            }
        )

        del trainer, optimizer, loss, model, sim, ale, train_metrics, test_metrics, val_metrics, elapsed_times
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Summary ===")
    for r in results:
        test_rmse = _summary_value(r, "test", "RMSE")
        val_rmse = _summary_value(r, "val", "RMSE")
        print(
            f"{r['method']:>12s} : "
            f"best_val_loss={val_rmse:.6f} | test_rmse={test_rmse:.6f}"
        )

    run_json = out_dir / f"{run_tag}.json"
    run_csv = out_dir / f"{run_tag}.csv"
    write_json(run_json, results)
    write_csv(run_csv, results_to_rows(results))

    if not args.no_aggregate:
        append_aggregate_files(results, out_dir=out_dir)

    print(f"\nSaved run file: {run_json}")
    print(f"Saved run file: {run_csv}")
    if not args.no_aggregate:
        print(f"Updated aggregate: {out_dir / 'results.json'}")
        print(f"Updated aggregate: {out_dir / 'results.csv'}")

    return results


if __name__ == "__main__":
    main()
