[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20688787.svg)](https://doi.org/10.5281/zenodo.20688787)

# ALE--Fréchet Multi-Task Learning Reproducibility Package

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

If you use this repository, please cite the accompanying manuscript and the archived software release.

### Manuscript

The accompanying manuscript is currently under review. Until the final bibliographic information is available, please cite it as:

```bibtex
@unpublished{hidalgo_ale_frechet_mtdl_2026,
  title  = {Explainability-Guided Soft Parameter Sharing for Multi-Task Time-Series Prediction Using ALE--Fréchet Similarity},
  author = {Hidalgo, Pablo and Rodriguez, Daniel and Domínguez-Díaz, Adrián},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

### Reproducibility package

Please also cite the archived version of this repository:

```bibtex
@software{hidalgo_ale_frechet_reproducibility_2026,
  title   = {ALE--Fréchet Multi-Task Learning Reproducibility Package},
  author  = {Hidalgo, Pablo and Rodriguez, Daniel and Domínguez-Díaz, Adrián},
  year    = {2026},
  version = {v1.0},
  doi = {https://doi.org/10.5281/zenodo.20688787},
  url     = {https://github.com/papabloblo/ale-frechet-mtdl-reproducibility}
}
```

Once the paper is accepted or publicly available, this section will be updated with the final journal citation and DOI.

## License

Source code is released under the MIT License. Paper artifacts, documentation, figures, and tables are released under CC BY 4.0 unless a venue or data-source restriction states otherwise.

