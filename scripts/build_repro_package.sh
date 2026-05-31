#!/usr/bin/env bash
# Build a clean reproducibility package for the ALE--Fréchet paper.
#
# Expected location:
#   scripts/build_repro_package.sh
#
# Default output:
#   ../ale-frechet-mtdl-reproducibility relative to the private paper repo.
#
# Usage:
#   bash scripts/build_repro_package.sh
#   bash scripts/build_repro_package.sh --target ../ale-frechet-mtdl-reproducibility
#   bash scripts/build_repro_package.sh --target /path/to/repo --clean --archive
#   bash scripts/build_repro_package.sh --dry-run

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_TARGET="$(cd "${PROJECT_ROOT}" && pwd)/ale-frechet-mtdl-reproducibility"

TARGET_DIR="${DEFAULT_TARGET}"
DO_CLEAN=0
DO_ARCHIVE=0
DRY_RUN=0
VERBOSE=1
PUBLIC_DATASETS=(polynomial multisine electricity exchange metrla nn5)

print_help() {
  cat <<USAGE
Build a clean ALE--Fréchet reproducibility repository from the private paper-dev repository.

Usage:
  ${SCRIPT_NAME} [options]

Options:
  --target DIR    Output reproducibility repository directory.
                  Default: ${DEFAULT_TARGET}
  --clean         Before exporting, remove only the target paths this script manages:
                  src/, MultiTaskDeepLearning/, configs/, scripts/, tests/, docs/,
                  paper_artifacts/, results/, data/sample/, .idea/, and MANIFEST.md.
                  This prevents stale files from older exports from remaining in
                  the reproducibility repository. It preserves .git/ and does not
                  delete unrelated files outside the listed managed paths.
  --archive       Create a .tar.gz archive next to the target directory after export.
  --dry-run       Show what would be copied without modifying the target.
  --quiet         Reduce output.
  -h, --help      Show this help message.

Recommended workflow:
  1. Work only in the private repository.
  2. Run this script to export a clean public reproducibility package.
  3. Inspect the target repository before committing/pushing.
USAGE
}

log() {
  if [[ "${VERBOSE}" -eq 1 ]]; then
    printf '[%s] %s\n' "${SCRIPT_NAME}" "$*"
  fi
}

warn() {
  printf '[%s] WARNING: %s\n' "${SCRIPT_NAME}" "$*" >&2
}

fail() {
  printf '[%s] ERROR: %s\n' "${SCRIPT_NAME}" "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || fail "--target requires a directory argument"
      TARGET_DIR="$2"
      shift 2
      ;;
    --clean)
      DO_CLEAN=1
      shift
      ;;
    --archive)
      DO_ARCHIVE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --quiet)
      VERBOSE=0
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

TARGET_DIR="$(mkdir -p "$(dirname "${TARGET_DIR}")" && cd "$(dirname "${TARGET_DIR}")" && pwd)/$(basename "${TARGET_DIR}")"

[[ -d "${PROJECT_ROOT}" ]] || fail "Project root not found: ${PROJECT_ROOT}"

RSYNC_DRY=""
if [[ "${DRY_RUN}" -eq 1 ]]; then
  RSYNC_DRY="--dry-run"
  log "Dry run enabled: no files will be changed."
fi

RSYNC_FLAGS=(-a)
if [[ "${VERBOSE}" -eq 1 ]]; then
  RSYNC_FLAGS+=(-v)
fi

make_dir() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo mkdir -p "$@"
  else
    mkdir -p "$@"
  fi
}

remove_paths() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo rm -rf "$@"
  else
    rm -rf "$@"
  fi
}

COMMON_EXCLUDES=(
  --exclude='.git/'
  --exclude='.github/'
  --exclude='.idea/'
  --exclude='__pycache__/'
  --exclude='*.py[cod]'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
  --exclude='.ipynb_checkpoints/'
  --exclude='.DS_Store'
  --exclude='*.log'
  --exclude='*.tmp'
  --exclude='*.bak'
  --exclude='*BACKUP*'
  --exclude='*backup*'
  --exclude='*_old.*'
  --exclude='*old.yaml'
  --exclude='wandb/'
  --exclude='mlruns/'
  --exclude='runs/'
  --exclude='checkpoints/'
)

copy_dir() {
  local src="$1"
  local dst="$2"
  shift 2 || true

  if [[ ! -d "${src}" ]]; then
    warn "Skipping missing directory: ${src#${PROJECT_ROOT}/}"
    return 0
  fi

  log "Copying ${src#${PROJECT_ROOT}/}/ -> ${dst#${TARGET_DIR}/}/"
  make_dir "${dst}"
  rsync "${RSYNC_FLAGS[@]}" --delete ${RSYNC_DRY} "${COMMON_EXCLUDES[@]}" "$@" "${src}/" "${dst}/"
}

copy_file() {
  local src="$1"
  local dst="$2"

  if [[ ! -f "${src}" ]]; then
    warn "Skipping missing file: ${src#${PROJECT_ROOT}/}"
    return 0
  fi

  log "Copying ${src#${PROJECT_ROOT}/} -> ${dst#${TARGET_DIR}/}"
  make_dir "$(dirname "${dst}")"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo cp "${src}" "${dst}"
  else
    cp "${src}" "${dst}"
  fi
}

write_file() {
  local dst="$1"
  local content="$2"

  log "Writing ${dst#${TARGET_DIR}/}"
  make_dir "$(dirname "${dst}")"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo cat ">" "${dst}"
  else
    printf '%s\n' "${content}" > "${dst}"
  fi
}

write_default_pyproject() {
  write_file "${TARGET_DIR}/pyproject.toml" '[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ale-frechet-mtdl"
version = "0.1.0"
description = "Reproducibility package for ALE-Frechet multi-task time-series prediction experiments."
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
  "numpy>=1.24,<3",
  "pandas>=2.0,<3",
  "PyYAML>=6.0,<7",
  "matplotlib>=3.7,<4",
  "scipy>=1.10,<2",
  "scikit-learn>=1.3,<2",
  "requests>=2.31,<3",
  "Pillow>=10.0,<12",
]

[tool.setuptools.packages.find]
where = ["src"]'
}

write_public_readme() {
  write_file "${TARGET_DIR}/README.md" '# ALE--Fréchet Multi-Task Learning Reproducibility Package

This repository contains the public reproducibility package for the paper:

> **Explainability-Guided Soft Parameter Sharing for Multi-Task Time-Series Prediction Using ALE--Fréchet Similarity**

The package is prepared for paper review and archival release. It contains the source code, experiment configurations, data-generation scripts, aggregated reference outputs, and manuscript artifacts needed to reproduce the main experiments and compare a local run against the submitted results.

## Repository Contents

```text
.
├── src/MultiTaskDeepLearning/     # Core package: datasets, models, trainer, losses, ALE similarity
├── configs/                       # Global, dataset, model, method, and sweep YAML files
├── scripts/                       # Data generation, training, aggregation, plotting, and server helpers
│   └── server/                    # Bootstrap, preflight, and logged run helpers
├── data/                          # Data README and optional small samples only
├── results/published/             # Aggregated reference CSV outputs used for validation
├── paper_artifacts/               # Figures, LaTeX tables, and CSV summaries used in the submitted paper
├── docs/                          # Dataset, hardware, and reproduction notes
├── Makefile                       # Main reproducibility entry points
├── pyproject.toml                 # Editable package metadata for the src/ layout
└── requirements.txt               # Non-PyTorch Python dependencies
```

Large raw datasets, full run logs, checkpoints, and local backups are intentionally excluded.

## Environment

Recommended environment:

- Linux workstation or server;
- Python 3.10 or newer;
- NVIDIA GPU for full sweeps;
- internet access for Python packages and public real-dataset downloads.

PyTorch and torchvision are not listed in `requirements.txt` because the correct wheels depend on the machine. The server bootstrap installs PyTorch before installing this package.

### Bootstrap

```bash
bash scripts/server/bootstrap_server.sh
source .venv/bin/activate
```

The default bootstrap uses CUDA 12.8 PyTorch wheels. For CPU-only checks:

```bash
CUDA_VARIANT=cpu bash scripts/server/bootstrap_server.sh
source .venv/bin/activate
```

Manual installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

Use the CPU PyTorch index URL instead of `cu128` on machines without CUDA.

## Preflight

```bash
python scripts/server/preflight_check.py
python scripts/server/preflight_check.py --dataset electricity
```

The preflight checks imports, PyTorch/CUDA visibility, GPU architecture compatibility, and expected processed dataset files when a dataset is supplied.

## Data Preparation

Generate synthetic datasets:

```bash
make data-synth
```

This writes:

```text
data/interim/multisine/
data/interim/polynomial/
```

Generate real datasets:

```bash
make generate-real DATASET=electricity
make generate-real DATASET=exchange
make generate-real DATASET=metrla
```

The additional supported generator included in this package is `nn5`. See `docs/datasets.md` for sources, redistribution notes, and expected output paths.

## Smoke Test

A small local smoke test is:

```bash
make data-synth
make sweep-compare-full DATASET=multisine SWEEP_METHODS="ale_frechet hard" SWEEP_COMBOS=1 COMPARE_SEEDS=0 SWEEP_MAX_WORKERS=1
make paper-results
```

This checks dataset generation, training, aggregation, and table-building without running the full benchmark. Small smoke-test outputs are not expected to match the full published benchmark exactly; use `validate-published` after a full reproduction or `compare-to-published` for the bundled reference workflow below.

## Full Paper Experiments

Run one dataset at a time:

```bash
make sweep-compare-full DATASET=polynomial
make sweep-compare-full DATASET=multisine
make sweep-compare-full DATASET=electricity
make sweep-compare-full DATASET=exchange
make sweep-compare-full DATASET=metrla
```

Each run writes:

```text
results/comparisons/<dataset>/results_all.csv
results/comparisons/<dataset>/results_mean_std.csv
results/comparisons/<dataset>/results_task_behavior.csv
```

For logged server execution:

```bash
bash scripts/server/run_experiment.sh sweep-compare-full electricity COMPARE_SEEDS=0,1,2,3,4
```

## Tables and Figures

After comparison outputs exist:

```bash
make paper-results
make ablation-tables
make similarity-update-table
make interpretability-validation
make interpretability-figures
```

Generated paper outputs are written under `paper/results/`, `paper/tables/`, and `paper/figures/`. The submitted reference artifacts are included under `paper_artifacts/`.

In a clean clone, `make paper-results` requires `results/comparisons/<dataset>/results_mean_std.csv` files. Those files are produced by the experiment sweeps. To rebuild paper tables from the bundled published aggregate outputs without rerunning experiments, use:

```bash
make compare-to-published
```

This runs `restore-published-results`, `paper-results`, and `validate-published`.

## Reference Outputs

Reference CSVs used for validation are included under:

```text
results/published/
paper_artifacts/tables/
paper_artifacts/latex_tables/
paper_artifacts/figures/
```

Exact floating-point equality is not expected across GPU architectures, CUDA versions, PyTorch versions, and low-level numerical libraries. Compare reproduced runs using aggregate trends, method rankings, and reported mean/std metrics. Keep `results/comparisons/` with any reproduced paper release.

Useful validation targets:

```bash
make restore-published-results  # Populate results/comparisons from results/published
make validate-published         # Compare paper/results CSVs against results/published
make compare-to-published       # Restore, rebuild paper CSVs/tables, and validate
```

## Reproduction Notes

- `configs/global.yaml` enables deterministic execution where supported.
- Dataset-specific seeds and hyperparameter grids are defined in `configs/sweeps/<dataset>.yaml`.
- The full real-dataset benchmark is GPU-expensive; run under `tmux`, `screen`, Slurm, or another scheduler.
- Reduced seed lists are appropriate for debugging only, not for paper-level reproduction.

## Citation

If you use this package, cite the accompanying paper submitted to *Knowledge-Based Systems*. Replace the placeholder BibTeX below with the final accepted citation when available:

```bibtex
@article{ale_frechet_mtdl_2026,
  title   = {Explainability-Guided Soft Parameter Sharing for Multi-Task Time-Series Prediction Using ALE--Frechet Similarity},
  journal = {Knowledge-Based Systems},
  year    = {2026},
  note    = {Submitted}
}
```

## License

Source code is released under the MIT License. Paper artifacts, documentation, figures, and tables are released under CC BY 4.0 unless a venue or data-source restriction states otherwise.
'
}

write_public_docs() {
  make_dir "${TARGET_DIR}/docs"

  write_file "${TARGET_DIR}/docs/reproduction_instructions.md" '# Reproduction Instructions

These commands are intended for a clean clone of the public reproducibility repository.

## 1. Create the Environment

Recommended GPU setup:

```bash
bash scripts/server/bootstrap_server.sh
source .venv/bin/activate
```

CPU-only setup for small checks:

```bash
CUDA_VARIANT=cpu bash scripts/server/bootstrap_server.sh
source .venv/bin/activate
```

Manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

## 2. Verify the Environment

```bash
python scripts/server/preflight_check.py
make print-datasets
```

## 3. Prepare Data

Synthetic datasets:

```bash
make data-synth
```

Real datasets:

```bash
make generate-real DATASET=electricity
make generate-real DATASET=exchange
make generate-real DATASET=metrla
```

Optional additional generators:

```bash
make generate-real DATASET=nn5
```

## 4. Run a Smoke Test

```bash
make data-synth
make sweep-compare-full DATASET=multisine SWEEP_METHODS="ale_frechet hard" SWEEP_COMBOS=1 COMPARE_SEEDS=0 SWEEP_MAX_WORKERS=1
make paper-results
```

## 5. Run Paper-Level Comparisons

```bash
make sweep-compare-full DATASET=polynomial
make sweep-compare-full DATASET=multisine
make sweep-compare-full DATASET=electricity
make sweep-compare-full DATASET=exchange
make sweep-compare-full DATASET=metrla
```

For logged server execution:

```bash
bash scripts/server/run_experiment.sh sweep-compare-full electricity COMPARE_SEEDS=0,1,2,3,4
```

## 6. Rebuild Paper Artifacts

```bash
make paper-results
make ablation-tables
make similarity-update-table
make interpretability-validation
make interpretability-figures
```

Reference outputs shipped with the submission are under `results/published/` and `paper_artifacts/`. Newly generated outputs are written under `results/comparisons/` and `paper/`.

For a clean-clone check that does not rerun experiments:

```bash
make compare-to-published
```

This restores `results/comparisons/<dataset>/results_mean_std.csv` from `results/published/paper_results_long.csv`, rebuilds the paper CSV/LaTeX outputs, and validates regenerated CSVs against the bundled published reference files.
'

  write_file "${TARGET_DIR}/docs/datasets.md" '# Datasets

The repository ships scripts and configurations for synthetic datasets and public real-world datasets. Large raw data files are not committed.

## Synthetic Datasets

- `multisine`: generated by `scripts/generate_synthetic.py`.
- `polynomial`: generated by `scripts/generate_synthetic.py`.

Command:

```bash
make data-synth
```

Expected outputs:

```text
data/interim/multisine/
data/interim/polynomial/
```

## Real Datasets

The real-data generator writes standardized supervised multi-task CSV files:

```text
data/interim/<dataset>/<dataset>_train.csv
data/interim/<dataset>/<dataset>_val.csv
data/interim/<dataset>/<dataset>_test.csv
data/interim/<dataset>/<dataset>_meta.csv
data/interim/<dataset>/<dataset>_scaler_stats.csv
```

Supported sources:

| Dataset | Source used by script | Command |
| --- | --- | --- |
| electricity | UCI ElectricityLoadDiagrams20112014 | `make generate-real DATASET=electricity` |
| exchange | European Central Bank euro foreign exchange reference rates | `make generate-real DATASET=exchange` |
| metrla | METR-LA CSV mirror, Zenodo record 5146275 | `make generate-real DATASET=metrla` |
| nn5 | Monash NN5 daily dataset, Zenodo record 4656117 | `make generate-real DATASET=nn5` |

The paper-level benchmark uses `polynomial`, `multisine`, `electricity`, `exchange`, and `metrla`. `nn5` is retained as an additional public generator and reference-result dataset.

## Preprocessing Summary

All real datasets are converted into a unified long-form panel with task identifiers, timestamps, target values, lagged target features, rolling statistics, and dataset-specific calendar/context features. Splits are temporal per task. Scaling statistics are estimated from training rows only and written to `<dataset>_scaler_stats.csv`.

## Redistribution

Raw real datasets are not redistributed in this repository. The scripts download from the public upstream sources above. Users should follow the license and citation requirements of each upstream dataset provider.
'

  write_file "${TARGET_DIR}/docs/hardware.md" '# Hardware and Runtime Notes

The full benchmark is intended for a Linux GPU server. Small synthetic smoke tests can run on CPU.

Recommended minimum environment:

- Python 3.10 or newer;
- PyTorch build matching the local CUDA runtime, or CPU-only PyTorch for smoke tests;
- NVIDIA GPU for full real-dataset sweeps;
- enough disk for `data/`, `results/`, logs, and saved ALE--Fréchet trackers.

The provided bootstrap defaults to:

```text
torch==2.7.1
torchvision==0.22.1
CUDA_VARIANT=cu128
```

Use `CUDA_VARIANT=cpu` for CPU-only checks or `CUDA_VARIANT=cu118` for older CUDA 11.8 environments. RTX 50xx / Blackwell GPUs require PyTorch wheels with CUDA 12.8 or newer.

Before reporting reproduction results, record:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '\''cpu'\'')"
nvidia-smi
```

Runtime depends strongly on GPU, dataset, number of seeds, and sweep grid size. The synthetic datasets are suitable for local debugging; electricity and METR-LA are the most expensive real-dataset runs.
'
}

write_public_data_readme() {
  write_file "${TARGET_DIR}/data/README.md" '# Data

Large raw datasets are not included in this repository. Generate or download data with the provided scripts:

```bash
make data-synth
make generate-real DATASET=electricity
make generate-real DATASET=exchange
make generate-real DATASET=metrla
```

Processed datasets are written under `data/interim/<dataset>/`. See `docs/datasets.md` for sources, preprocessing notes, and redistribution constraints.
'
}

write_public_results_readme() {
  write_file "${TARGET_DIR}/results/README.md" '# Results

`results/published/` contains the aggregated reference outputs shipped with the paper submission. Raw logs, checkpoints, and exploratory runs are intentionally excluded.

When reproducing the experiments, new outputs are written under:

```text
results/comparisons/<dataset>/
```

Compare reproduced aggregate CSVs against `results/published/` at the level of method rankings, mean/std metrics, and reported trends rather than exact floating-point equality.

Use:

```bash
make compare-to-published
```

to verify that the bundled published outputs can rebuild the main paper CSVs and tables in a clean clone. Use:

```bash
make validate-published
```

after a full experimental rerun and `make paper-results` to compare regenerated paper CSVs with the bundled reference CSVs.
'
}

write_public_gitignore() {
  write_file "${TARGET_DIR}/.gitignore" '# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/

# Environments
.venv/
venv/
env/
.env
.envrc

# Raw/generated data
data/raw/
data/interim/
data/processed/
!data/README.md
!data/sample/

# Generated experiment outputs
results/comparisons/
results/ale_frechet/
results/component_ablation/
results/logs/
logs/
reports/
paper/
backups/
checkpoints/
runs/
wandb/
mlruns/
*.log

# Keep submitted reference artifacts tracked
!results/
!results/README.md
!results/published/
!results/published/**
!paper_artifacts/
!paper_artifacts/**

# OS/editor
.DS_Store
.idea/
.vscode/
'
}

write_public_docs_license() {
  write_file "${TARGET_DIR}/LICENSE-DOCS.md" '# Documentation and Paper Artifact License

Unless otherwise stated, documentation, paper tables, paper figures, and aggregated CSV artifacts in this repository are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Dataset redistribution remains governed by the upstream dataset providers.
'
}

copy_public_configs() {
  make_dir "${TARGET_DIR}/configs/data_generation"
  make_dir "${TARGET_DIR}/configs/datasets"
  make_dir "${TARGET_DIR}/configs/methods"
  make_dir "${TARGET_DIR}/configs/models"
  make_dir "${TARGET_DIR}/configs/sweeps"

  copy_file "${PROJECT_ROOT}/configs/global.yaml" "${TARGET_DIR}/configs/global.yaml"
  copy_file "${PROJECT_ROOT}/configs/methods/ale_frechet.yaml" "${TARGET_DIR}/configs/methods/ale_frechet.yaml"
  copy_file "${PROJECT_ROOT}/configs/models/baselines.yaml" "${TARGET_DIR}/configs/models/baselines.yaml"
  copy_file "${PROJECT_ROOT}/configs/sweeps/component_ablation.yaml" "${TARGET_DIR}/configs/sweeps/component_ablation.yaml"

  local dataset
  for dataset in "${PUBLIC_DATASETS[@]}"; do
    copy_file "${PROJECT_ROOT}/configs/datasets/${dataset}.yaml" "${TARGET_DIR}/configs/datasets/${dataset}.yaml"
    copy_file "${PROJECT_ROOT}/configs/models/${dataset}.yaml" "${TARGET_DIR}/configs/models/${dataset}.yaml"
    copy_file "${PROJECT_ROOT}/configs/sweeps/${dataset}.yaml" "${TARGET_DIR}/configs/sweeps/${dataset}.yaml"
    if [[ -f "${PROJECT_ROOT}/configs/data_generation/${dataset}.yaml" ]]; then
      copy_file "${PROJECT_ROOT}/configs/data_generation/${dataset}.yaml" "${TARGET_DIR}/configs/data_generation/${dataset}.yaml"
    fi
  done
}

write_public_generate_real_datasets() {
  local src="${PROJECT_ROOT}/scripts/generate_real_datasets.py"
  local dst="${TARGET_DIR}/scripts/generate_real_datasets.py"

  if [[ ! -f "${src}" ]]; then
    warn "Skipping public real-dataset script generation; missing scripts/generate_real_datasets.py"
    return 0
  fi

  log "Writing scripts/generate_real_datasets.py"
  make_dir "$(dirname "${dst}")"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo python-public-generate-real-datasets "${src}" ">" "${dst}"
    return 0
  fi

  {
    sed -n '1,765p' "${src}"
    sed -n '1029,1080p' "${src}"
    cat <<'PYCODE'


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
PYCODE
  } > "${dst}"
}

write_public_run_compare() {
  write_file "${TARGET_DIR}/scripts/run_compare.sh" '#!/usr/bin/env bash
set -euo pipefail

datasets=(
  polynomial
  multisine
  electricity
  exchange
  metrla
  nn5
)

for dataset in "${datasets[@]}"; do
  echo ">>> Running comparison sweep for ${dataset}"
  make sweep-compare DATASET="${dataset}"

  echo ">>> Aggregating comparison results for ${dataset}"
  make aggregate-compare DATASET="${dataset}"
done
'
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    chmod +x "${TARGET_DIR}/scripts/run_compare.sh"
  fi
}

filter_public_reference_tables() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 0

  local keep
  keep="$(IFS=,; echo "${PUBLIC_DATASETS[*]}")"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo filter-public-reference-tables "${dir}"
    return 0
  fi

  while IFS= read -r -d '' file; do
    awk -F',' -v keep="${keep}" '
      BEGIN {
        split(keep, names, ",")
        for (i in names) allowed[names[i]] = 1
      }
      NR == 1 {
        is_dataset_table = ($1 == "dataset")
        print
        next
      }
      !is_dataset_table || allowed[$1]
    ' "${file}" > "${file}.tmp"
    mv "${file}.tmp" "${file}"
  done < <(find "${dir}" -type f -name '*.csv' -print0)

  if [[ -f "${dir}/README.txt" ]]; then
    printf '%s\n' "Reference CSV outputs for the public reproducibility package." \
      "Rows are limited to the datasets distributed through the public configs." > "${dir}/README.txt"
  fi
}

log "Project root: ${PROJECT_ROOT}"
log "Target repo : ${TARGET_DIR}"

if [[ "${DO_CLEAN}" -eq 1 ]]; then
  log "Cleaning managed target paths before export; preserving .git/ and unrelated files"
  remove_paths \
    "${TARGET_DIR}/src" \
    "${TARGET_DIR}/MultiTaskDeepLearning" \
    "${TARGET_DIR}/configs" \
    "${TARGET_DIR}/scripts" \
    "${TARGET_DIR}/tests" \
	    "${TARGET_DIR}/docs" \
	    "${TARGET_DIR}/paper_artifacts" \
	    "${TARGET_DIR}/results" \
	    "${TARGET_DIR}/data/sample" \
	    "${TARGET_DIR}/.idea" \
	    "${TARGET_DIR}/MANIFEST.md"
fi

make_dir "${TARGET_DIR}"

# -----------------------------------------------------------------------------
# 1. Top-level metadata and environment files
# -----------------------------------------------------------------------------
copy_file "${PROJECT_ROOT}/LICENSE" "${TARGET_DIR}/LICENSE"
copy_file "${PROJECT_ROOT}/CITATION.cff" "${TARGET_DIR}/CITATION.cff"
copy_file "${PROJECT_ROOT}/requirements.txt" "${TARGET_DIR}/requirements.txt"
copy_file "${PROJECT_ROOT}/requirements-dev.txt" "${TARGET_DIR}/requirements-dev.txt"
copy_file "${PROJECT_ROOT}/environment.yml" "${TARGET_DIR}/environment.yml"
copy_file "${PROJECT_ROOT}/pyproject.toml" "${TARGET_DIR}/pyproject.toml"
copy_file "${PROJECT_ROOT}/Makefile" "${TARGET_DIR}/Makefile"
copy_file "${PROJECT_ROOT}/Dockerfile" "${TARGET_DIR}/Dockerfile"

write_public_readme
write_public_docs_license
if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
  write_default_pyproject
fi

# -----------------------------------------------------------------------------
# 2. Source code
# Supports both the current layout: MultiTaskDeepLearning/
# and a future refactored layout: code/src/ or src/
# -----------------------------------------------------------------------------
if [[ -d "${PROJECT_ROOT}/code/src" ]]; then
  copy_dir "${PROJECT_ROOT}/code/src" "${TARGET_DIR}/src"
elif [[ -d "${PROJECT_ROOT}/src" ]]; then
  copy_dir "${PROJECT_ROOT}/src" "${TARGET_DIR}/src"
elif [[ -d "${PROJECT_ROOT}/MultiTaskDeepLearning" ]]; then
  copy_dir "${PROJECT_ROOT}/MultiTaskDeepLearning" "${TARGET_DIR}/src/MultiTaskDeepLearning" \
    --exclude='dataset_DEV.py' \
    --exclude='setup/'
else
  warn "No source-code directory found. Expected one of: code/src, src, MultiTaskDeepLearning."
fi

# -----------------------------------------------------------------------------
# 3. Reproducibility scripts
# -----------------------------------------------------------------------------
if [[ -d "${PROJECT_ROOT}/code/scripts" ]]; then
  copy_dir "${PROJECT_ROOT}/code/scripts" "${TARGET_DIR}/scripts"
elif [[ -d "${PROJECT_ROOT}/scripts" ]]; then
  copy_dir "${PROJECT_ROOT}/scripts" "${TARGET_DIR}/scripts" \
    --exclude='latex/' \
    --exclude='remove-bikes-ale-similarity.py'
else
  warn "No scripts directory found."
fi
write_public_generate_real_datasets
write_public_run_compare

# -----------------------------------------------------------------------------
# 4. Configurations
# Copy only the datasets and auxiliary configs used by the public
# reproducibility package.
# -----------------------------------------------------------------------------
copy_public_configs

# -----------------------------------------------------------------------------
# 5. Tests
# -----------------------------------------------------------------------------
if [[ -d "${PROJECT_ROOT}/code/tests" ]]; then
  copy_dir "${PROJECT_ROOT}/code/tests" "${TARGET_DIR}/tests"
elif [[ -d "${PROJECT_ROOT}/tests" ]]; then
  copy_dir "${PROJECT_ROOT}/tests" "${TARGET_DIR}/tests"
else
  warn "No tests directory found. Consider adding smoke tests before publishing."
fi

# -----------------------------------------------------------------------------
# 6. Published/aggregated results
# Avoid raw logs, checkpoints, and intermediate runs.
# -----------------------------------------------------------------------------
make_dir "${TARGET_DIR}/results"

if [[ -d "${PROJECT_ROOT}/results/aggregated" ]]; then
  copy_dir "${PROJECT_ROOT}/results/aggregated" "${TARGET_DIR}/results/published"
elif [[ -d "${PROJECT_ROOT}/results/processed" ]]; then
  copy_dir "${PROJECT_ROOT}/results/processed" "${TARGET_DIR}/results/published"
elif [[ -d "${PROJECT_ROOT}/paper/results" ]]; then
  copy_dir "${PROJECT_ROOT}/paper/results" "${TARGET_DIR}/results/published"
else
  warn "No aggregated/published results found."
fi
filter_public_reference_tables "${TARGET_DIR}/results/published"

# -----------------------------------------------------------------------------
# 7. Paper artifacts: figures and tables/results used by the manuscript
# -----------------------------------------------------------------------------
make_dir "${TARGET_DIR}/paper_artifacts"
copy_dir "${PROJECT_ROOT}/paper/figures" "${TARGET_DIR}/paper_artifacts/figures"
copy_dir "${PROJECT_ROOT}/paper/results" "${TARGET_DIR}/paper_artifacts/tables"
filter_public_reference_tables "${TARGET_DIR}/paper_artifacts/tables"
copy_dir "${PROJECT_ROOT}/paper/tables" "${TARGET_DIR}/paper_artifacts/latex_tables"
copy_file "${PROJECT_ROOT}/paper/paper.pdf" "${TARGET_DIR}/paper_artifacts/paper.pdf"

# -----------------------------------------------------------------------------
# 8. Documentation
# -----------------------------------------------------------------------------
if [[ -d "${PROJECT_ROOT}/docs" ]]; then
  copy_dir "${PROJECT_ROOT}/docs" "${TARGET_DIR}/docs"
else
  make_dir "${TARGET_DIR}/docs"
fi
write_public_docs

# -----------------------------------------------------------------------------
# 9. Data placeholders only. Do not export private or large raw datasets by default.
# -----------------------------------------------------------------------------
make_dir "${TARGET_DIR}/data/sample"
write_public_data_readme

# -----------------------------------------------------------------------------
# 10. Public .gitignore
# -----------------------------------------------------------------------------
write_public_results_readme
write_public_gitignore

# -----------------------------------------------------------------------------
# 11. Manifest
# -----------------------------------------------------------------------------
MANIFEST_PATH="${TARGET_DIR}/MANIFEST.md"
log "Writing MANIFEST.md"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo cat ">" "${MANIFEST_PATH}"
else
  {
    echo "# Reproducibility package manifest"
    echo
    echo "Source repository: generated from the private development repository associated with the manuscript."
    echo
    echo "## Included top-level paths"
    find "${TARGET_DIR}" -maxdepth 2 -mindepth 1 \
      -not -path "${TARGET_DIR}/.git" \
      -not -path "${TARGET_DIR}/.git/*" \
      | sed "s#${TARGET_DIR}/#- #" \
      | sort
  } > "${MANIFEST_PATH}"
fi

# -----------------------------------------------------------------------------
# 12. Optional archive
# -----------------------------------------------------------------------------
if [[ "${DO_ARCHIVE}" -eq 1 ]]; then
  ARCHIVE_PATH="$(dirname "${TARGET_DIR}")/$(basename "${TARGET_DIR}")_$(date +%Y%m%d_%H%M%S).tar.gz"
  log "Creating archive: ${ARCHIVE_PATH}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${TARGET_DIR}")" "$(basename "${TARGET_DIR}")"
  else
    tar --exclude='.git' -czf "${ARCHIVE_PATH}" -C "$(dirname "${TARGET_DIR}")" "$(basename "${TARGET_DIR}")"
  fi
fi

log "Export complete. Inspect the target before committing: ${TARGET_DIR}"
