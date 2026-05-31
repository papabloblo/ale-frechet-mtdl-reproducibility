#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import platform
import re
import sys
from pathlib import Path
from typing import Iterable

import yaml

REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "yaml",
    "matplotlib",
    "scipy",
    "sklearn",
    "requests",
    "PIL",
    "torch",
    "torchvision",
]

DEFAULT_CONFIGS = [
    Path("configs/global.yaml"),
    Path("configs/models/baselines.yaml"),
    Path("configs/methods/ale_frechet.yaml"),
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _check_paths(paths: Iterable[Path]) -> int:
    failures = 0
    for path in paths:
        if path.exists():
            print(f"[OK] {path}")
        else:
            print(f"[MISSING] {path}")
            failures += 1
    return failures


def _check_imports() -> int:
    failures = 0
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"[OK] import {module_name}")
        except Exception as exc:
            print(f"[FAIL] import {module_name}: {exc}")
            failures += 1
    return failures


def _check_dataset_files(dataset: str | None) -> int:
    if not dataset:
        return 0

    dataset_cfg_path = Path("configs/datasets") / f"{dataset}.yaml"
    if not dataset_cfg_path.exists():
        print(f"[FAIL] dataset config not found: {dataset_cfg_path}")
        return 1

    cfg = _load_yaml(dataset_cfg_path)
    files = [Path(p) for p in cfg.get("paths", {}).values() if p]
    print(f"Dataset: {dataset}")
    return _check_paths(files)


def _check_torch_runtime() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"[FAIL] torch runtime unavailable: {exc}")
        return 1

    def _parse_cuda_version(version: str | None) -> tuple[int, int] | None:
        if not version:
            return None
        match = re.match(r"^(\d+)\.(\d+)", version)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _parse_sm_arch(arch: str) -> tuple[int, int] | None:
        match = re.match(r"^sm_(\d+)$", arch)
        if not match:
            return None
        digits = match.group(1)
        if len(digits) < 2:
            return None
        return int(digits[:-1]), int(digits[-1])

    failures = 0
    print(f"torch: {torch.__version__}")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device_capability = torch.cuda.get_device_capability(device_index)
        supported_arches = list(getattr(torch.cuda, "get_arch_list", lambda: [])())
        parsed_arches = [arch for arch in (_parse_sm_arch(item) for item in supported_arches) if arch is not None]

        print(f"cuda device count: {torch.cuda.device_count()}")
        print(f"cuda current device: {device_index}")
        print(f"cuda device name: {torch.cuda.get_device_name(device_index)}")
        print(f"cuda device capability: {device_capability[0]}.{device_capability[1]}")
        if supported_arches:
            print(f"torch supported arches: {' '.join(supported_arches)}")

        if parsed_arches and device_capability > max(parsed_arches):
            max_supported = max(parsed_arches)
            print(
                "[FAIL] PyTorch was not built for this GPU architecture: "
                f"device sm_{device_capability[0]}{device_capability[1]}, "
                f"max supported sm_{max_supported[0]}{max_supported[1]}"
            )
            failures += 1

        cuda_version = _parse_cuda_version(torch.version.cuda)
        if device_capability >= (12, 0) and (cuda_version is None or cuda_version < (12, 8)):
            print(
                "[FAIL] Blackwell GPUs require a PyTorch build with CUDA 12.8 or newer. "
                f"Detected torch CUDA build: {torch.version.cuda}"
            )
            failures += 1

        try:
            x = torch.randn((64, 64), device="cuda")
            y = torch.matmul(x, x)
            _ = y.sum().item()
            print("[OK] simple CUDA kernel smoke test")
        except Exception as exc:
            print(f"[FAIL] simple CUDA kernel smoke test: {exc}")
            failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate server environment for this repository.")
    parser.add_argument("--dataset", default=None, help="Optional dataset name to verify data files.")
    args = parser.parse_args()

    failures = 0

    _print_header("Repository files")
    failures += _check_paths(DEFAULT_CONFIGS)

    _print_header("Python imports")
    failures += _check_imports()

    _print_header("Torch runtime")
    failures += _check_torch_runtime()

    if args.dataset:
        _print_header("Dataset files")
        failures += _check_dataset_files(args.dataset)

    if failures:
        print(f"\nPreflight failed with {failures} issue(s).")
        return 1

    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
