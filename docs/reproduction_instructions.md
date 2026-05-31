# Reproduction Instructions

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

