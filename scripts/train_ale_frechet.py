#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generalized trainer for the ALE–Fréchet MTL method with train/val/test splits.

This script reads configurations from YAML files to execute reproducible
experiments using the ALE–Fréchet-based multi-task deep learning framework.

It expects four YAML files:
    - Global config: seed, device, logging paths
    - Dataset config: data paths, column names, batch sizes
    - Model config: modules (arbitrary depth), sharing type, similarity hooks
    - Method config: ALE + similarity parameters, LR, L2, epochs

Examples
--------
Run a full experiment:

>>> python -m scripts.train_ale_frechet \\
...   --global-config configs/global.yaml \\
...   --dataset-config configs/datasets/multi_sine.yaml \\
...   --model-config configs/models/lstm_soft.yaml \\
...   --method-config configs/methods/ale_frechet.yaml
"""

from __future__ import annotations
import argparse
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional, Tuple

# --- Project imports ---
from MultiTaskDeepLearning.data import MultitaskDatasetDF, MultitaskDataloader
from MultiTaskDeepLearning.model import MultiTaskModel
from MultiTaskDeepLearning.similarity.ale import MultiTaskALE
from MultiTaskDeepLearning.similarity.similarity import MultitaskSimilarity
from MultiTaskDeepLearning.trainer import MultiTaskTrainer
from MultiTaskDeepLearning.loss import MultiTaskLoss, rmse_loss, mae_loss, mape_loss
from MultiTaskDeepLearning.tracking import Tracker

# --- Third-party ---
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml


# =========================================================
# Utility functions
# =========================================================
def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file into a Python dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents, or an empty dict if the file is empty.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(pref: str | None) -> torch.device:
    """
    Resolve torch device from a preference string.

    Parameters
    ----------
    pref : str or None
        Preferred device name ('cuda', 'cpu', or 'auto').

    Returns
    -------
    torch.device
        The resolved PyTorch device.
    """
    if pref is None or pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set global random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value.
    deterministic : bool, default=True
        If True, enables deterministic cuDNN backend.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def now_stamp() -> str:
    """Return a compact timestamp (YYYYMMDD-HHMMSS)."""
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: str | Path) -> Path:
    """
    Ensure a directory exists.

    Parameters
    ----------
    p : str or Path
        Directory path.

    Returns
    -------
    Path
        The same path, guaranteed to exist.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def merge_dicts(*ds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple dictionaries (recursive).

    Parameters
    ----------
    *ds : dict
        Dictionaries to merge.

    Returns
    -------
    dict
        Merged dictionary with last values winning conflicts.
    """
    out: Dict[str, Any] = {}
    for d in ds:
        for k, v in (d or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge_dicts(out[k], v)
            else:
                out[k] = v
    return out

def infer_input_dim(dataset_cfg: Dict[str, Any]) -> int:
    """
    Infer the input feature dimension from the dataset configuration.

    Parameters
    ----------
    dataset_cfg : dict
        Dataset configuration containing the key ``feature_cols``.

    Returns
    -------
    int
        Number of input features.

    Raises
    ------
    ValueError
        If ``feature_cols`` is missing.
    """
    feature_cols = dataset_cfg.get("feature_cols", None)
    if feature_cols is None:
        raise ValueError(
            "dataset_cfg must define 'feature_cols' to infer model input dimension."
        )
    return int(len(feature_cols))

# =========================================================
# Data handling
# =========================================================
def build_dataloaders(dataset_cfg: Dict[str, Any]) -> Dict[str, MultitaskDataloader]:
    """
    Build multitask dataloaders for train, validation, test, and ALE passes.

    Parameters
    ----------
    dataset_cfg : dict
        Dataset configuration with keys:
        - ``paths.train`` : str, path to training data
        - ``paths.val``   : str, path to validation data (optional)
        - ``paths.test``  : str, path to test data
        - ``task_id_col`` : str, column name for task IDs
        - ``target_cols`` : list[str], target columns
        - ``feature_cols`` : list[str], feature columns
        - ``batch.train/test/val/ale`` : int, batch sizes

    Returns
    -------
    dict[str, MultitaskDataloader]
        Mapping {"train", "val", "test", "ale"} → corresponding dataloaders.

    Notes
    -----
    The ALE dataloader defaults to using the training dataset.
    """
    paths = dataset_cfg["paths"]
    train_path = paths["train"]
    val_path = paths.get("validation", None)
    test_path = paths["test"]

    task_id_col = dataset_cfg.get("task_id_col", "task")
    target_cols = dataset_cfg.get("target_cols", ["y_target"])
    feature_cols = dataset_cfg["feature_cols"]
    cols = [task_id_col] + target_cols + feature_cols

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_val = pd.read_csv(val_path) if val_path and Path(val_path).exists() else None

    # Keep only expected columns (guards against accidental extras)
    df_train = df_train[cols]
    df_test = df_test[cols]
    if df_val is not None:
        df_val = df_val[cols]

    # Your dataset API (bikes-style args)
    ds_train = MultitaskDatasetDF(data=df_train, task_id=task_id_col, target_names=target_cols)
    ds_test  = MultitaskDatasetDF(data=df_test,  task_id=task_id_col, target_names=target_cols)
    ds_val   = MultitaskDatasetDF(data=df_val,   task_id=task_id_col, target_names=target_cols) if df_val is not None else None

    bs = dataset_cfg.get("batch", {})
    dl_train = MultitaskDataloader(ds_train, batch_size=int(bs.get("train", 128)), shuffle=True)
    dl_test  = MultitaskDataloader(ds_test,  batch_size=int(bs.get("test", 128)),  shuffle=False)
    dl_val   = MultitaskDataloader(ds_val,   batch_size=int(bs.get("validation", 128)),   shuffle=False) if ds_val else None
    dl_ale   = MultitaskDataloader(ds_train, batch_size=int(bs.get("ale", 256)),   shuffle=False)

    dls = {"train": dl_train, "test": dl_test, "ale": dl_ale}
    if dl_val:
        dls["val"] = dl_val
    return dls


# =========================================================
# Module factory and model builder (architecture-agnostic)
# =========================================================
def _act(name: str) -> nn.Module:
    """
    Small activation factory.

    Parameters
    ----------
    name : str
        One of {'relu', 'gelu', 'tanh'}.

    Returns
    -------
    nn.Module
        Activation instance (default 'relu' if unknown).
    """
    name = (name or "relu").lower()
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}.get(name, nn.ReLU())


def _layer_from_spec(spec: Dict[str, Any]) -> nn.Module:
    """
    Build a single nn.Module from a minimal spec.

    Supported ``type`` values:
      - linear:    in_features, out_features, bias (opt)
      - relu|gelu|tanh
      - dropout:   p
      - batchnorm1d: num_features, eps (opt), momentum (opt), affine (opt)
      - flatten
      - identity
      - lstm:      input_size, hidden_size, num_layers (opt), batch_first (opt)

    Parameters
    ----------
    spec : dict
        Minimal layer specification with at least a ``type`` key.

    Returns
    -------
    nn.Module
    """
    t = spec.get("type", "").lower()
    if t == "linear":
        return nn.Linear(**{k: v for k, v in spec.items() if k in ("in_features", "out_features", "bias")})
    if t in ("relu", "gelu", "tanh"):
        return _act(t)
    if t == "dropout":
        return nn.Dropout(p=float(spec.get("p", 0.0)))
    if t == "batchnorm1d":
        kwargs = {k: v for k, v in spec.items() if k in ("num_features", "eps", "momentum", "affine")}
        return nn.BatchNorm1d(**kwargs)
    if t == "flatten":
        return nn.Flatten()
    if t == "identity":
        return nn.Identity()
    if t == "lstm":
        # Return a module that outputs the full sequence (like your HardLSTM)
        input_size = int(spec["input_size"])
        hidden_size = int(spec["hidden_size"])
        num_layers = int(spec.get("num_layers", 1))
        batch_first = bool(spec.get("batch_first", True))
        class _LSTMSeq(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
            def forward(self, x):
                h, _ = self.lstm(x)
                return h
        return _LSTMSeq()
    raise ValueError(f"Unsupported layer type: {t!r}")


def _module_factory_from_layers(layers_spec: List[Dict[str, Any]]) -> Callable[[], nn.Module]:
    """
    Convert a list of layer specs into a factory producing an nn.Sequential.

    Parameters
    ----------
    layers_spec : list of dict
        Each item is passed to `_layer_from_spec`.

    Returns
    -------
    callable
        Zero-arg callable that returns a fresh nn.Module.
    """
    def build() -> nn.Module:
        blocks: List[nn.Module] = []
        for s in layers_spec:
            blocks.append(_layer_from_spec(s))
        return nn.Sequential(*blocks) if len(blocks) > 1 else (blocks[0] if blocks else nn.Identity())
    return build


def build_modules_layout(model_cfg: Dict[str, Any]) -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Optional[str]]]:
    """
    Build `modules_layout` and `similarity_layers` for MultiTaskModel.

    Parameters
    ----------
    model_cfg : dict
        Must contain:
        - modules: list of {name: str, shared: 'hard'|'soft'|None, layers: [...]}
        - similarity_layers: {in: str | null, out: str}

    Returns
    -------
    (modules_layout, similarity_layers) : tuple
        - modules_layout: list of (name, {"shared": ..., "module": factory})
        - similarity_layers: dict with keys "in" (optional) and "out"
    """
    modules = model_cfg.get("modules", [])
    if not isinstance(modules, list) or not modules:
        raise ValueError("model_cfg.modules must be a non-empty list.")

    layout: List[Tuple[str, Dict[str, Any]]] = []
    for m in modules:
        name = m["name"]
        shared = m.get("shared", None)
        layers_spec = m.get("layers", [])
        factory = _module_factory_from_layers(layers_spec)
        layout.append((name, {"shared": shared, "module": factory}))

    sim_layers = model_cfg.get("similarity_layers", {})
    if "out" not in sim_layers:
        raise ValueError("model_cfg.similarity_layers must define an 'out' layer name.")
    # Accept None for "in" (start from the first layer)
    return layout, {"in": sim_layers.get("in", None), "out": sim_layers["out"]}


def build_model(model_cfg: Dict[str, Any], n_tasks: int, device: torch.device) -> MultiTaskModel:
    """
    Instantiate MultiTaskModel with *arbitrary* number of modules.

    Parameters
    ----------
    model_cfg : dict
        Contains:
        - modules: list of layer groups (see `build_modules_layout`)
        - similarity_layers: names used to slice model for similarity
        - same_parameters: bool, whether soft-shared layers tie weights
        - shared_input_data: bool
        - weight_init: str ('kaiming_uniform'|...)
        - bias_range: float
    n_tasks : int
        Number of tasks in the dataset.
    device : torch.device
        Target device.

    Returns
    -------
    MultiTaskModel
    """
    modules_layout, similarity_layers = build_modules_layout(model_cfg)
    same_parameters = bool(model_cfg.get("same_parameters", False))
    shared_input_data = bool(model_cfg.get("shared_input_data", False))
    weight_init = model_cfg.get("weight_init", "kaiming_uniform")
    bias_range = float(model_cfg.get("bias_range", 0.1))

    model = MultiTaskModel(
        n_tasks=n_tasks,
        modules_layout=modules_layout,
        similarity_layers=similarity_layers,
        device=device,
        same_parameters=same_parameters,
        shared_input_data=shared_input_data,
        weight_init=weight_init,
        bias_range=bias_range,
    )
    return model


# =========================================================
# ALE, similarity, loss, optimizer/scheduler, trainer
# =========================================================
def build_losses(model: MultiTaskModel, method_cfg: Dict[str, Any]) -> MultiTaskLoss:
    """
    Build MultiTaskLoss compatible with the project's loss.py implementation.

    Parameters
    ----------
    model : MultiTaskModel
        The multitask model instance (required by MultiTaskLoss for penalty()).
    method_cfg : dict
        Method configuration. Expected keys:
        - loss.name : {"rmse","mae","mape"} (default "rmse")
        - errors.names : list[str] (default ["rmse","mae","mape"])
        - regularization.l2 : float (default 1e-3)

    Returns
    -------
    MultiTaskLoss
        Configured loss object with a primary loss function and a dictionary
        of error functions.
    """
    # Primary loss
    loss_name = (method_cfg.get("loss", {}) or {}).get("name", "rmse")
    loss_fn_map = {"rmse": rmse_loss, "mae": mae_loss, "mape": mape_loss}
    if loss_name not in loss_fn_map:
        raise ValueError(f"Unsupported loss.name={loss_name!r}. Choose from {list(loss_fn_map)}.")
    loss_fn = loss_fn_map[loss_name]

    # Errors dictionary
    err_names = ((method_cfg.get("errors", {}) or {}).get("names", None))
    if err_names is None:
        err_names = ["rmse", "mae", "mape"]

    errors_fn: Dict[str, Callable] = {}
    for n in err_names:
        if n not in loss_fn_map:
            raise ValueError(f"Unsupported errors.names entry {n!r}. Choose from {list(loss_fn_map)}.")
        errors_fn[n] = loss_fn_map[n]

    # L2 penalty (note: your MultiTaskLoss names this l2_penalty)
    l2_penalty = float((method_cfg.get("regularization", {}) or {}).get("l2", 1e-3))

    return MultiTaskLoss(
        model=model,
        loss_fn=loss_fn,
        errors_fn=errors_fn,
        l2_penalty=l2_penalty,
    )

def build_ale(model: MultiTaskModel, dls: Dict[str, MultitaskDataloader], n_tasks: int, method_cfg: Dict[str, Any]) -> MultiTaskALE:
    """
    Instantiate MultiTaskALE consistent with your original bikes script.
    """
    ale_cfg = method_cfg.get("ale", {})
    return MultiTaskALE(
        model=model,
        dataloader=dls["ale"],
        n_tasks=n_tasks,
        shared_input_data=bool(method_cfg.get("shared_input_data", False)),
        n_features_out=int(ale_cfg.get("n_features_out", 1)),
        num_intervals=int(ale_cfg.get("n_intervals", 30)),
        device=str(method_cfg.get("ale_device", "cuda" if torch.cuda.is_available() else "cpu")),
        epsilon=float(ale_cfg.get("epsilon", 1e-4)),
        n_guess=int(ale_cfg.get("n_guess", 1000)),
    )


def build_similarity(ale_obj: MultiTaskALE) -> MultitaskSimilarity:
    """
    Build MultitaskSimilarity from ALE curves (bikes-style).
    """
    return MultitaskSimilarity(ale_curves=ale_obj)


def build_optimizer_and_scheduler(model: MultiTaskModel, method_cfg: Dict[str, Any]):
    """
    Create optimizer and scheduler, matching bikes-style trainer signature.
    """
    optim_cfg = method_cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Default scheduler: ReduceLROnPlateau (as in bikes)
    sched_cfg = optim_cfg.get("scheduler", {"type": "plateau", "mode": "min", "factor": 0.5, "patience": 10})
    t = sched_cfg.get("type", "plateau").lower()
    if t == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "min"),
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 10))
        )
    elif t == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 100)),
            gamma=float(sched_cfg.get("gamma", 0.5)),
        )
    else:
        scheduler = None
    return optimizer, scheduler


def build_trainer(
    model: MultiTaskModel,
    dataloaders: Dict[str, MultitaskDataloader],
    losses: MultiTaskLoss,
    ale_obj: MultiTaskALE,
    similarity_obj: MultitaskSimilarity,
    method_cfg: Dict[str, Any],
    log_dir: Path,
    seed: int,
) -> MultiTaskTrainer:
    """
    Assemble the MultiTaskTrainer using bikes-style argument names.

    Returns
    -------
    MultiTaskTrainer
        Configured trainer with tracking set to `log_dir`.
    """
    optim_cfg = method_cfg.get("optim", {})
    reg_cfg = method_cfg.get("regularization", {})
    trainer_cfg = method_cfg.get("trainer", {})

    optimizer, scheduler = build_optimizer_and_scheduler(model, method_cfg)

    trainer = MultiTaskTrainer(
        model=model,
        train_dataloader=dataloaders["train"],
        validation_dataloader=dataloaders.get("val", None),
        test_dataloader=dataloaders["test"],
        #dataloader_ale=dataloaders["ale"],
        optimizer=optimizer,
        scheduler=scheduler,
        loss=losses,
        ale=ale_obj,
        multitask_similarity=similarity_obj,
        maximize_loss=False,
        early_stopping_epochs=int(trainer_cfg.get("early_stopping", trainer_cfg.get("early_stopping_epochs", 25))),
        print_each_epochs=int(trainer_cfg.get("print_each", 1)),
        ale_each_epochs= None if method_cfg.get("ale_update_every") is None else int(method_cfg.get("ale_update_every")),
        similarity_each_epochs=None if method_cfg.get("similarity", {}).get("update_every", 10) is None else int(method_cfg.get("similarity", {}).get("update_every", 10)),
        keep_similarity_epochs=None if method_cfg.get("similarity", {}).get("keep_epochs", 10) is None else int(method_cfg.get("similarity", {}).get("keep_epochs", 10)),
        track_device=str(trainer_cfg.get("track_device", "cpu")),
        track_epochs=trainer_cfg.get("track_epochs", None),
        save_results_each=trainer_cfg.get("save_results_each", None),
        logging_dir=str(log_dir),
        learning_type=str(method_cfg.get("learning_type", "")),
        dataset_name=str(trainer_cfg.get("dataset_name", "")),
        n_intervals_ale=int(method_cfg.get("ale", {}).get("n_intervals", 30)),
        learning_rate=float(optim_cfg.get("lr", 1e-3)),
        train_batch_size=dataloaders["train"]._dataloader.batch_size,
        test_batch_size=dataloaders["test"]._dataloader.batch_size,
        ale_batch_size=dataloaders["ale"]._dataloader.batch_size,
        l2penalty=float(reg_cfg.get("l2", 1e-2)),
        architecture=str(method_cfg.get("architecture", "Architecture")),
        seed=int(seed),
        print_limit_epochs=int(trainer_cfg.get("print_limit_epochs", 20)),
        config_info=method_cfg,
    )
    trainer.tracking.path = str(log_dir)
    return trainer

def extract_times(tracker: Tracker) -> Dict[str, float]:
    """Extract total elapsed times for train, validation, test, ALE, and similarity."""
    times = {}
    for key in ["train", "validation", "test", "ale", "similarity"]:
        timing = tracker.track.get(key, {}).get("timing", None)
        if timing is not None:
            times[f"time_{key}"] = timing.total_time()
    times["total_time"] = sum(times.values())
    return times

def best_model_metrics(trainer: MultiTaskTrainer) -> dict:
    best_model = trainer.get_best_model()
    metrics_names = best_model["train_metrics"]

    info = {"epoch": best_model["epoch"]} | extract_times(trainer.tracking)

    for metric in metrics_names:
        info[f"{metric}_train"] = best_model["train_metrics"][metric].item()
        info[f"{metric}_val"] = best_model["val_metrics"][metric].item()
        info[f"{metric}_test"] = best_model["test_metrics"][metric].item()

    return info

def flatten_dict(d, parent_key="", sep="_"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def update_results_metrics(metrics_file: Path, run_dir: str, metrics: dict, config: dict) -> None:
    config_aux = dict(config)
    #del config_aux["trainer"]
    del config_aux["loss"]
    del config_aux["errors"]

    results = {'run_dir': str(run_dir)} | flatten_dict(config_aux) | metrics
    pd.DataFrame([results]).to_csv(str(metrics_file), index=False, mode='a', header=not metrics_file.exists())



# =========================================================
# Main execution
# =========================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Train ALE–Fréchet MTL with train/val/test splits.")
    ap.add_argument("--global-config", required=True)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--method-config", required=True)
    ap.add_argument("--tag", default="", help="Optional experiment tag.")
    return ap.parse_args()


def main() -> None:
    """
    Load configs, prepare data, build an arbitrary-depth architecture,
    train the model, and save results.
    """
    args = parse_args()

    gcfg = load_yaml(args.global_config)
    dcfg = load_yaml(args.dataset_config)
    mcfg = load_yaml(args.model_config)
    xcfg = load_yaml(args.method_config)

    resolved_cfg = merge_dicts({"global": gcfg}, {"dataset": dcfg}, {"model": mcfg}, {"method": xcfg})
    seed = int(gcfg.get("seed", 42))
    set_seed(seed, bool(gcfg.get("deterministic", True)))
    device = resolve_device(gcfg.get("device", "auto"))

    dls = build_dataloaders(dcfg)

    # ---------------------------------------------------------
    # Infer input dimension and patch LSTM specs
    # ---------------------------------------------------------
    input_dim = infer_input_dim(dcfg)

    for module in mcfg.get("modules", []):
        for layer in module.get("layers", []):
            if str(layer.get("type", "")).lower() == "lstm":
                layer["input_size"] = input_dim

    # Discover number of tasks from training dataset
    train_ds = dls["train"]._dataloader.dataset  # underlying dataset
    n_tasks = int(getattr(train_ds, "n_tasks"))

    # Build model + ALE/similarity + losses
    model = build_model(mcfg, n_tasks=n_tasks, device=device)
    losses = build_losses(model, xcfg)
    mt_ale = build_ale(model, dls, n_tasks=n_tasks, method_cfg=xcfg)
    mt_sim = build_similarity(mt_ale)

    # Logging dir
    root_log = ensure_dir(gcfg.get("log_dir", "results"))
    dataset_name = dcfg.get("name", Path(dcfg["paths"]["train"]).parent.name)
    run_dir = ensure_dir(root_log / "runs" / f"{dataset_name}__ale_frechet__{now_stamp()}__{args.tag}")
    meta_dir = ensure_dir(root_log / "metrics")

    # Save resolved config
    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, indent=2)

    # Trainer
    trainer = build_trainer(model, dls, losses, mt_ale, mt_sim, xcfg, run_dir, seed)
    trainer.train(epochs=int(xcfg.get("trainer", {}).get("epochs", 1000)))

    # Persist tracking and a compact CSV row
    try:
        trainer.tracking.save()
    except Exception:
        pass


    metrics_dir = meta_dir / f"{dataset_name}__ale_frechet.csv"

    update_results_metrics(metrics_dir, run_dir, best_model_metrics(trainer), xcfg)

    print(f"[OK] Run complete → {run_dir}")


if __name__ == "__main__":
    main()
