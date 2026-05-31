# -*- coding: utf-8 -*-
"""
Architecture factory for MultiTaskModel.

This local factory lets define the architecture-building logic.

Usage
-----
In your model YAML:
-------------------
use_factory: true
factory:
  name: "mlp_soft"
  input_dim: 64
  hard: [128, 128]
  soft: [64, 32]
  head: [32, 1]
  activation: "relu"

Then in `train_ale_frechet.py`, `build_model()` will call:
>>> from scripts.architectures import build_from_yaml
>>> model = build_from_yaml(model_cfg, device=device)
"""

from __future__ import annotations
from typing import Dict, Any, List
import torch
from torch import nn
from MultiTaskDeepLearning.model import MultiTaskModel


def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}.get(name, nn.ReLU())


def _mlp(sizes: List[int], activation: str = "relu", dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:  # no act/dropout on last layer
            layers += [_act(activation)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)


def build_from_yaml(model_cfg: Dict[str, Any], device: torch.device) -> MultiTaskModel:
    """
    Build a MultiTaskModel by constructing submodules from YAML.

    Parameters
    ----------
    model_cfg : dict
        Expected keys:
        - use_factory : bool (must be True to call this)
        - factory.name : str, currently supports {"mlp_soft"}
        - factory.input_dim : int
        - factory.hard / soft / head : list[int]
        - factory.activation : str (relu|tanh|gelu)
        - factory.dropout : float, optional
        - similarity_layers.in / out : str, passed through in config
    device : torch.device
        Target device.

    Returns
    -------
    MultiTaskModel
        Instantiated model compatible with your project.
    """
    fac = model_cfg.get("factory", {})
    name = fac.get("name", "mlp_soft").lower()
    input_dim = int(fac.get("input_dim"))
    act = fac.get("activation", "relu")
    drop = float(fac.get("dropout", 0.0))

    if name != "mlp_soft":
        raise ValueError(f"Unsupported factory.name={name}")

    hard = [input_dim] + list(fac.get("hard", []))
    soft = [hard[-1]] + list(fac.get("soft", []))
    head = [soft[-1]] + list(fac.get("head", []))

    hard_shared = _mlp(hard, activation=act, dropout=drop)
    soft_shared = _mlp(soft, activation=act, dropout=drop)
    task_head   = _mlp(head, activation=act, dropout=0.0)

    # IMPORTANT:
    # If your original MultiTaskModel expected submodules instead of an "architecture"
    # string, expose a constructor that accepts them. If not, we pass these sizes
    # via the config and let the class create the modules internally.
    try:
        model = MultiTaskModel(
            architecture="mlp_soft",
            config={
                **model_cfg,
                "sizes": {"hard": hard, "soft": soft, "head": head},
                "activation": act,
                "dropout": drop,
            },
            device=device,
        )
        return model
    except TypeError:
        # Fallback: try a constructor that accepts modules directly (if your project defines it).
        model = MultiTaskModel(
            hard_layers=hard_shared,
            soft_layers=soft_shared,
            task_independent_layers=task_head,
            device=device,
        )
        return model
