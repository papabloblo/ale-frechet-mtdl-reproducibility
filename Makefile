# ============================================================================
# Makefile for ALE–Fréchet Multitask Learning (Results Repository)
# ============================================================================
# This Makefile keeps the original project layout and paths:
#   - configs/      : dataset, model, method, and sweep YAML files
#   - scripts/      : training, sweep, aggregation, and reporting scripts
#   - results/      : experiment outputs and comparison summaries
#   - paper/        : paper-ready tables and figures
#   - backups/      : tar.gz archives for reproducibility, results, and data
#
# Common usage:
#   make help
#   make sweep-compare DATASET=electricity
#   make sweep-compare DATASET=electricity GPU=1
#   make sweep-compare-full DATASET=electricity
#   make rank-methods DATASET=electricity
#   make latex-ranking DATASET=electricity
#   make paper-results
#   make backup
# ============================================================================

# ---------------------------------------------------------------------------
# Core paths and executables
# ---------------------------------------------------------------------------
PY                  ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHON              ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
CONFIGS             := configs
DATASETS            := $(CONFIGS)/datasets
GLOBAL_CFG          := $(CONFIGS)/global.yaml
COMPARE_OUT_ROOT    ?= results/comparisons
FIGS                := reports/figures
GPU                 ?=
CUDA_ENV            := $(if $(GPU),CUDA_VISIBLE_DEVICES=$(GPU),)

# ---------------------------------------------------------------------------
# Comparison defaults
# ---------------------------------------------------------------------------
COMPARE_ALE_MODEL_CFG       := $(CONFIGS)/models/$(DATASET).yaml
COMPARE_BASELINE_MODEL_CFG  := $(CONFIGS)/models/baselines.yaml
COMPARE_METHOD_CFG          := $(CONFIGS)/methods/ale_frechet.yaml
COMPARE_BASELINES           ?= ale_frechet single_task_mlp hard soft mmoe crossstitch ple mtan
COMPARE_SEEDS               ?= 0,1,2,3,4
SWEEP_METHODS               ?=
SWEEP_COMBOS                ?=
SWEEP_COMBO_TAGS            ?=
SWEEP_LIST_COMBOS           ?=
SWEEP_MAX_WORKERS           ?=

# ---------------------------------------------------------------------------
# Ranking / LaTeX export defaults
# ---------------------------------------------------------------------------
RANK_SCRIPT                 := scripts/rank_methods_from_task_behavior.py
RANK_LATEX_SCRIPT           := scripts/ranking_to_latex.py
RANK_METRIC                 ?= loss
RANK_SPLIT                  ?= test
RANK_PERF_WEIGHT            ?= 0.7
RANK_STAB_WEIGHT            ?= 0.3
RANK_TOPK                   ?= 10

# ---------------------------------------------------------------------------
# Paper-ready results defaults
# ---------------------------------------------------------------------------
PAPER_RESULTS_DIR           := paper/results
PAPER_TABLES_DIR            := paper/tables
COMPARISONS_DIR             := results/comparisons
PUBLISHED_RESULTS_DIR       := results/published
PAPER_METRICS               := test_rmse test_mae total_time
PAPER_SELECTION_METRIC      := val_rmse_mean
PAPER_EXCLUDE_DATASETS      ?=
PAPER_EXCLUDE_METHODS       ?=
INTERPRETABILITY_FIGURES_DIR := paper/figures/interpretability
INTERPRETABILITY_VALIDATION_RESULTS_DIR := paper/results/interpretability_validation
INTERPRETABILITY_VALIDATION_DATASETS ?= multisine polynomial
ABLATION_RESULTS_DIR        := paper/results/ablation
ABLATION_TABLES_DIR         := paper/tables
ABLATION_DATASETS           := polynomial multisine
ABLATION_METRICS            := test_rmse test_mae total_time
ABLATION_SELECTION_METRIC   := val_rmse_mean
ABLATION_TOPK               := 10
SIMILARITY_UPDATE_RESULTS_DIR      := paper/results/ablation
SIMILARITY_UPDATE_TABLES_DIR       := paper/tables
SIMILARITY_UPDATE_DATASETS         := polynomial multisine
SIMILARITY_UPDATE_METRICS          := test_rmse test_mae total_time
SIMILARITY_UPDATE_SELECTION_METRIC := test_rmse_mean
COMPONENT_ABLATION_SWEEP_CFG       := configs/sweeps/component_ablation.yaml
COMPONENT_ABLATION_OUT_ROOT        := results/component_ablation
COMPONENT_ABLATION_RESULTS_DIR     := paper/results/component_ablation
COMPONENT_ABLATION_DATASETS        ?= multisine polynomial

# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
REAL_DATASETS := electricity exchange metrla nn5
DATA_GENERATION_CONFIGS := $(CONFIGS)/data_generation
# Use the union of generated data folders and dataset config filenames.
INTERIM_DIR := data/interim
ALLOWED_DATASETS := $(strip $(shell \
	{ \
		for d in "$(INTERIM_DIR)"/*/; do [ -d "$$d" ] && basename "$${d%/}"; done | sort; \
		for f in "$(DATASETS)"/*.yaml; do [ -f "$$f" ] && basename "$$f" .yaml; done | sort; \
	} | sort -u))


# ---------------------------------------------------------------------------
# Helper macros
# ---------------------------------------------------------------------------
define assert_allowed_datasets
	@if [ -z "$(ALLOWED_DATASETS)" ]; then \
		echo "ERROR: No datasets were detected."; \
		echo "       Checked $(INTERIM_DIR)/ and $(DATASETS)/*.yaml"; \
		exit 1; \
	fi
endef

define require_dataset
	@if [ -z "$(DATASET)" ]; then 		echo "ERROR: Missing DATASET. Usage: make $(1) DATASET=<$(ALLOWED_DATASETS)>"; 		exit 2; 	fi
	$(call assert_allowed_datasets)
	@if ! echo " $(ALLOWED_DATASETS) " | grep -F -q " $(DATASET) "; then 		echo "ERROR: Dataset '''$(DATASET)''' is not allowed. Allowed: $(ALLOWED_DATASETS)"; 		exit 2; 	fi
endef

print-datasets:
	@echo "$(ALLOWED_DATASETS)"

help:
	@echo "Usage: make <target> [DATASET=<name>] [GPU=<cuda-index>]"
	@echo ""
	@echo "Setup and discovery"
	@echo "  setup                               Install project dependencies"
	@echo "  print-datasets                      List detected datasets"
	@echo ""
	@echo "Data generation and visualization"
	@echo "  data-synth                          Generate synthetic datasets"
	@echo "  vis-synth                           Create synthetic dataset figures"
	@echo "  generate-real DATASET=<name>        Build one real dataset"
	@echo ""
	@echo "Training and sweeps"
	@echo "  train-compare DATASET=<name>        Train the comparison methods once"
	@echo "  train-compare-all                   Train once for every detected dataset"
	@echo "  sweep-compare DATASET=<name>        Run the comparison sweep"
	@echo "      SWEEP_METHODS='m1 m2' SWEEP_COMBOS='1 3' SWEEP_COMBO_TAGS='tag'"
	@echo "      SWEEP_LIST_COMBOS=1             List matching combos without training"
	@echo "      SWEEP_MAX_WORKERS=1             Cap concurrent training subprocesses"
	@echo "  aggregate-compare DATASET=<name>    Aggregate one sweep output"
	@echo "  sweep-compare-full DATASET=<name>   Run sweep + aggregation"
	@echo "  sweep-compare-all                   Run sweeps for all datasets"
	@echo "  aggregate-compare-all               Aggregate all dataset sweeps"
	@echo "  sweep-compare-full-all              Run sweep + aggregation for all datasets"
	@echo "  sweep-synthetic                     Run sweep + aggregation for synthetic datasets"
	@echo ""
	@echo "Ranking and paper artifacts"
	@echo "  rank-methods DATASET=<name>         Build method ranking CSV"
	@echo "  rank-methods-all                    Build rankings for all datasets"
	@echo "  latex-ranking DATASET=<name>        Export ranking table to LaTeX"
	@echo "  latex-ranking-all                   Export all ranking tables to LaTeX"
	@echo "  paper-results                       Build paper-ready result tables"
	@echo "  restore-published-results           Restore bundled published CSVs into results/comparisons"
	@echo "  validate-published                  Compare paper/results CSVs against results/published"
	@echo "  compare-to-published                Restore bundled results, rebuild tables, and validate"
	@echo "      PAPER_EXCLUDE_DATASETS='d1 d2' PAPER_EXCLUDE_METHODS='m1 m2'"
	@echo "  ablation-tables                     Build ALE--Frechet ablation tables for polynomial and multisine"
	@echo "  component-ablation                  Run/summarize ALE--Frechet component ablation"
	@echo "  similarity-update-table            Build the table answering whether similarity should be recomputed every step"
	@echo "  interpretability-figures           Build figures for the paper interpretability subsection"
	@echo "  interpretability-validation        Validate learned synthetic task relationships"
	@echo "  clean-paper-results                 Remove generated paper result files"
	@echo "  learning-curves DATASET=<name>      Build comparison learning-curves figure"
	@echo "  ale-frechet-ablation-curves DATASET=<name>  Build ALE--Frechet ablation curves"
	@echo ""
	@echo "Backups"
	@echo "  build-repro-package                 Export a clean public reproducibility package"
	@echo "      REPRO_PACKAGE_TARGET=../repo    Optional output directory"
	@echo "      REPRO_PACKAGE_CLEAN=1           Clean managed target paths first"
	@echo "      REPRO_PACKAGE_ARCHIVE=1         Also create a tar.gz archive"
	@echo "      REPRO_PACKAGE_DRY_RUN=1         Show export actions without writing"
	@echo "  backup                              Create reproducibility, results, and data backups"
	@echo "  backup-repro                        Archive code and configs"
	@echo "  backup-results                      Archive reports and results"
	@echo "  backup-paper                        Archive paper/"
	@echo "  backup-data                         Archive data"
	@echo "  backup-data-minimal                 Archive reproducibility-minimal interim data (exclude by_task/)"
	@echo "      BACKUP_DATASETS='d1 d2'         Optional dataset subset; defaults to all"

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
setup:
	$(PY) -m pip install -U pip
	@if [ -f pyproject.toml ]; then $(PY) -m pip install -e . ; \
	elif [ -f requirements.txt ]; then $(PY) -m pip install -r requirements.txt ; fi

# ---------------------------------------------------------------------------
# Data generation and visualization
# ---------------------------------------------------------------------------
data-synth:
	@echo ">>> Generating synthetic datasets..."
	$(PY) scripts/generate_synthetic.py

vis-synth:
	@echo ">>> Visualizing synthetic datasets..."
	$(PY) scripts/visualize_synthetic.py --dataset all --outdir $(FIGS)
	@echo ">>> Overlay figures saved under $(FIGS)"
	@echo ">>> Generating per-task grids (limited to 6 tasks)..."
	$(PY) scripts/visualize_synthetic.py --dataset all --per-task --max-tasks 6 --outdir $(FIGS)
	@echo ">>> All visualization completed."


generate-real:
ifndef DATASET
	$(error DATASET is not set. Use one of: $(REAL_DATASETS), or a config file name under $(DATA_GENERATION_CONFIGS))
endif
	@dataset_arg="$(DATASET)"; \
	config_path="$(DATA_GENERATION_CONFIGS)/$(DATASET).yaml"; \
	if [ -f "$(DATA_GENERATION_CONFIGS)/$(DATASET)" ]; then \
		config_path="$(DATA_GENERATION_CONFIGS)/$(DATASET)"; \
	fi; \
	if echo " $(REAL_DATASETS) " | grep -F -q " $(DATASET) "; then \
		:; \
	elif [ -f "$$config_path" ]; then \
		config_dataset=$$(sed -n 's/^[[:space:]]*dataset:[[:space:]]*//p' "$$config_path" | sed -n '1p'); \
		if [ -n "$$config_dataset" ]; then \
			dataset_arg="$$config_dataset"; \
		else \
			dataset_arg=""; \
			for d in $(REAL_DATASETS); do \
				case "$(DATASET)" in $$d*) dataset_arg="$$d"; break ;; esac; \
			done; \
		fi; \
		if ! echo " $(REAL_DATASETS) " | grep -F -q " $$dataset_arg "; then \
			echo "ERROR: Could not infer a real dataset for config '$$config_path'."; \
			echo "       Add 'dataset: <one of $(REAL_DATASETS)>' to the config."; \
			exit 1; \
		fi; \
	else \
		echo "ERROR: Invalid DATASET '$(DATASET)'"; \
		echo "Allowed real datasets: $(REAL_DATASETS)"; \
		echo "Allowed config names: $$(find "$(DATA_GENERATION_CONFIGS)" -maxdepth 1 -type f -name '*.yaml' -exec basename {} .yaml \; 2>/dev/null | sort | xargs)"; \
		exit 1; \
	fi; \
	echo "Generating real dataset: $(DATASET)"; \
	$(PY) scripts/generate_real_datasets.py \
		--dataset "$$dataset_arg" \
		--config "$$config_path" \
		--outdir "$(INTERIM_DIR)/$(DATASET)"

generate-electricity:
	$(MAKE) generate-real DATASET=electricity

generate-exchange:
	$(MAKE) generate-real DATASET=exchange

generate-metrla:
	$(MAKE) generate-real DATASET=metrla

generate-nn5:
	$(MAKE) generate-real DATASET=nn5

generate-all-reals: generate-electricity generate-exchange generate-metrla generate-nn5

sweep-ale-frechet:
	$(call require_dataset,sweep-ale-frechet)
	@echo ">>> ALE–Fréchet sweep on dataset: $(DATASET)"
	$(PY) scripts/sweep_ale_frechet.py \
		--global-config $(GLOBAL_CFG) \
		--dataset-config $(DATASETS)/$(DATASET).yaml \
		--model-config $(CONFIGS)/models/$(DATASET).yaml \
		--base-method-config $(ALE_BASE_METHOD) \
		--sweep-config $(CONFIGS)/sweeps/$(DATASET).yaml \
		--outdir $(CONFIGS)/methods/sweeps/ale_frechet/$(DATASET)

sweep-ale-frechet-all:
	$(call assert_allowed_datasets)
	@echo ">>> ALE–Fréchet sweep on ALL allowed datasets: $(ALLOWED_DATASETS)"
	@for d in $(ALLOWED_DATASETS); do \
		echo ">>> ------------------------------------------"; \
		echo ">>> Dataset: $$d"; \
		$(PY) scripts/sweep_ale_frechet.py \
			--global-config $(GLOBAL_CFG) \
			--dataset-config $(DATASETS)/$$d.yaml \
			--model-config $(CONFIGS)/models/$$d.yaml \
			--base-method-config $(ALE_BASE_METHOD) \
			--sweep-config $(CONFIGS)/sweeps/$$d.yaml \
			--outdir $(CONFIGS)/methods/sweeps/ale_frechet/$$d || exit 1; \
	done

# ---------------------------------------------------------------------------
# ALE–Fréchet sweep study
# ---------------------------------------------------------------------------
ALE_BASE_METHOD := $(CONFIGS)/methods/ale_frechet.yaml

# ---------------------------------------------------------------------------
# Comparison experiments
# ---------------------------------------------------------------------------
train-compare:
	$(call require_dataset,train-compare)
	@echo ">>> Running comparison for dataset: $(DATASET)"
	$(CUDA_ENV) $(PYTHON) -m scripts.train_compare_methods \
		--global-config $(GLOBAL_CFG) \
		--dataset-config $(DATASETS)/$(DATASET).yaml \
		--ale-model-config $(CONFIGS)/models/$(DATASET).yaml \
		--baseline-model-config $(COMPARE_BASELINE_MODEL_CFG) \
		--method-config $(COMPARE_METHOD_CFG) \
		$(foreach b,$(COMPARE_BASELINES),--baseline $(b))

train-compare-all:
	$(call assert_allowed_datasets)
	@for d in $(ALLOWED_DATASETS); do \
		echo ">>> Running comparison for dataset: $$d"; \
		$(MAKE) train-compare DATASET=$$d || exit 1; \
	done

sweep-compare:
	$(call require_dataset,sweep-compare)
	@echo ">>> Running comparison seed sweep for dataset: $(DATASET)"
	@if [ -n "$(GPU)" ]; then echo ">>> Using CUDA_VISIBLE_DEVICES=$(GPU)"; fi
	@mkdir -p $(COMPARE_OUT_ROOT)/$(DATASET)
	$(CUDA_ENV) $(PYTHON) -m scripts.sweep_compare_methods \
		--global-config $(GLOBAL_CFG) \
		--dataset-config $(DATASETS)/$(DATASET).yaml \
		--ale-model-config $(CONFIGS)/models/$(DATASET).yaml \
		--baseline-model-config $(COMPARE_BASELINE_MODEL_CFG) \
		--sweep-config $(CONFIGS)/sweeps/$(DATASET).yaml \
		--method-config $(COMPARE_METHOD_CFG) \
		$(if $(strip $(SWEEP_METHODS)),--methods $(SWEEP_METHODS),) \
		$(if $(strip $(SWEEP_COMBOS)),--combo-indices $(SWEEP_COMBOS),) \
		$(if $(strip $(SWEEP_COMBO_TAGS)),--combo-tags $(SWEEP_COMBO_TAGS),) \
		$(if $(strip $(SWEEP_LIST_COMBOS)),--list-combos,) \
		$(if $(strip $(SWEEP_MAX_WORKERS)),--max-workers $(SWEEP_MAX_WORKERS),) \
		--out-dir $(COMPARE_OUT_ROOT)/$(DATASET)

aggregate-compare:
	$(call require_dataset,aggregate-compare)
	@echo ">>> Aggregating comparison results for dataset: $(DATASET)"
	$(PYTHON) -m scripts.aggregate_compare_results \
		--run-dir $(COMPARE_OUT_ROOT)/$(DATASET)/sweep_runs \
		--out-dir $(COMPARE_OUT_ROOT)/$(DATASET)

sweep-compare-full: sweep-compare aggregate-compare

sweep-compare-all:
	$(call assert_allowed_datasets)
	@for d in $(ALLOWED_DATASETS); do \
		echo ">>> Running sweep for dataset: $$d"; \
		$(MAKE) sweep-compare DATASET=$$d || exit 1; \
	done

aggregate-compare-all:
	$(call assert_allowed_datasets)
	@for d in $(ALLOWED_DATASETS); do \
		echo ">>> Aggregating comparison results for dataset: $$d"; \
		$(MAKE) aggregate-compare DATASET=$$d || exit 1; \
	done

rank-methods:
	$(call require_dataset,rank-methods)
	@echo ">>> Ranking methods for dataset: $(DATASET)"
	$(PYTHON) $(RANK_SCRIPT) \
		--input $(COMPARE_OUT_ROOT)/$(DATASET)/results_task_behavior.csv \
		--metric $(RANK_METRIC) \
		--split $(RANK_SPLIT) \
		--performance-weight $(RANK_PERF_WEIGHT) \
		--stability-weight $(RANK_STAB_WEIGHT)

rank-methods-all:
	$(call assert_allowed_datasets)
	@for d in $(ALLOWED_DATASETS); do \
		echo ">>> Ranking methods for dataset: $$d"; \
		$(MAKE) rank-methods DATASET=$$d || exit 1; \
	done

latex-ranking: rank-methods
	$(call require_dataset,latex-ranking)
	@echo ">>> Generating LaTeX ranking table for dataset: $(DATASET)"
	$(PYTHON) $(RANK_LATEX_SCRIPT) \
		--input $(COMPARE_OUT_ROOT)/$(DATASET)/method_ranking.csv \
		--output $(COMPARE_OUT_ROOT)/$(DATASET)/method_ranking.tex \
		--top-k $(RANK_TOPK) \
		--caption "Method ranking for $(DATASET) ($(RANK_SPLIT), $(RANK_METRIC)) balancing average performance and cross-task stability." \
		--label tab:ranking_$(DATASET)_$(RANK_SPLIT)_$(RANK_METRIC) \
		--use-booktabs \
		--bold-best

latex-ranking-all:
	$(call assert_allowed_datasets)
	@for d in $(ALLOWED_DATASETS); do \
		echo ">>> Generating LaTeX ranking table for dataset: $$d"; \
		$(MAKE) latex-ranking DATASET=$$d || exit 1; \
	done

sweep-compare-full-all:
	$(call assert_allowed_datasets)
	@for d in $(ALLOWED_DATASETS); do \
		echo ">>> Running sweep+aggregation for dataset: $$d"; \
		$(MAKE) sweep-compare-full DATASET=$$d || exit 1; \
	done

# ---------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------
BACKUP_PATH   := backups
STAMP         := $(shell date +%Y%m%d_%H%M%S)
REPRO_PACKAGE_SCRIPT := scripts/build_repro_package.sh
REPRO_PACKAGE_TARGET ?=
REPRO_PACKAGE_CLEAN ?=
REPRO_PACKAGE_ARCHIVE ?=
REPRO_PACKAGE_DRY_RUN ?=
REPRO_PACKAGE_QUIET ?=
REPRO_PACKAGE_ARGS = \
	$(if $(strip $(REPRO_PACKAGE_TARGET)),--target "$(REPRO_PACKAGE_TARGET)",) \
	$(if $(filter 1 true yes,$(REPRO_PACKAGE_CLEAN)),--clean,) \
	$(if $(filter 1 true yes,$(REPRO_PACKAGE_ARCHIVE)),--archive,) \
	$(if $(filter 1 true yes,$(REPRO_PACKAGE_DRY_RUN)),--dry-run,) \
	$(if $(filter 1 true yes,$(REPRO_PACKAGE_QUIET)),--quiet,)
REPRO_ITEMS   := configs MultiTaskDeepLearning scripts Makefile README.md DEPLOYMENT.md Dockerfile requirements.txt requirements-dev.txt
RESULTS_ITEMS := reports results
PAPER_ITEMS   := paper
DATA_ITEMS    := data

# Directories for each backup type (so archives live under these folders)
REPRO_DIR     := $(BACKUP_PATH)/repro
RESULTS_DIR   := $(BACKUP_PATH)/results
PAPER_DIR     := $(BACKUP_PATH)/paper
DATA_DIR      := $(BACKUP_PATH)/data

REPRO_BACKUP   := $(REPRO_DIR)/repro_$(STAMP).tar.gz
RESULTS_BACKUP := $(RESULTS_DIR)/results_$(STAMP).tar.gz
PAPER_BACKUP   := $(PAPER_DIR)/paper_$(STAMP).tar.gz
DATA_BACKUP    := $(DATA_DIR)/data_$(STAMP).tar.gz
DATA_MIN_BACKUP := $(DATA_DIR)/data_interim_minimal_$(STAMP).tar.gz
BACKUP_DATASETS ?=

build-repro-package:
	@echo ">>> Building reproducibility package"
	bash $(REPRO_PACKAGE_SCRIPT) $(REPRO_PACKAGE_ARGS)

backup: backup-repro backup-results backup-data

backup-repro:
	@mkdir -p $(REPRO_DIR)
	@items=""; \
	for f in $(REPRO_ITEMS); do \
		if [ -e $$f ]; then items="$$items $$f"; fi; \
	done; \
	echo "Creating reproducibility backup: $(REPRO_BACKUP)"; \
	tar -czf $(REPRO_BACKUP) $$items

backup-results:
	@mkdir -p $(RESULTS_DIR)
	@items=""; \
	for f in $(RESULTS_ITEMS); do \
		if [ -e $$f ]; then items="$$items $$f"; fi; \
	done; \
	echo "Creating results backup: $(RESULTS_BACKUP)"; \
	tar -czf $(RESULTS_BACKUP) $$items

backup-paper:
	@mkdir -p $(PAPER_DIR)
	@items=""; \
	for f in $(PAPER_ITEMS); do \
		if [ -e $$f ]; then items="$$items $$f"; fi; \
	done; \
	if [ -z "$$items" ]; then \
		echo "ERROR: no paper items found to archive."; \
		exit 1; \
	fi; \
	echo "Creating paper backup: $(PAPER_BACKUP)"; \
	tar -czf $(PAPER_BACKUP) $$items

backup-data:
	@mkdir -p $(DATA_DIR)
	@echo "Creating data backup: $(DATA_BACKUP)"
	@tar -czf $(DATA_BACKUP) $(DATA_ITEMS)

backup-data-minimal:
	@mkdir -p $(DATA_DIR)
	@if [ ! -d $(INTERIM_DIR) ]; then \
		echo "ERROR: $(INTERIM_DIR) does not exist."; \
		exit 1; \
	fi
	@echo "Creating minimal interim data backup: $(DATA_MIN_BACKUP)"
	@if [ -z "$(strip $(BACKUP_DATASETS))" ]; then \
		echo "Datasets: all under $(INTERIM_DIR)"; \
		tar -czf $(DATA_MIN_BACKUP) \
			--exclude='$(INTERIM_DIR)/*/by_task' \
			--exclude='$(INTERIM_DIR)/*/by_task/*' \
			$(INTERIM_DIR); \
	else \
		items=""; \
		for ds in $(BACKUP_DATASETS); do \
			path="$(INTERIM_DIR)/$$ds"; \
			if [ ! -d "$$path" ]; then \
				echo "ERROR: requested dataset '$$ds' does not exist at $$path"; \
				exit 1; \
			fi; \
			items="$$items $$path"; \
		done; \
		echo "Datasets: $(BACKUP_DATASETS)"; \
		tar -czf $(DATA_MIN_BACKUP) \
			--exclude='*/by_task' \
			--exclude='*/by_task/*' \
			$$items; \
	fi

# ---------------------------------------------------------------------------
# Synthetic sweeps
# ---------------------------------------------------------------------------
synthetic-datasets := polynomial multisine

sweep-synthetic:
	@echo "Running synthetic sweeps..."
	@for ds in $(synthetic-datasets); do \
		echo "----------------------------------------"; \
		echo "Dataset: $$ds"; \
		$(MAKE) sweep-compare DATASET=$$ds || exit 1; \
		$(MAKE) aggregate-compare DATASET=$$ds || exit 1; \
	done

# ---------------------------------------------------------------------------
# Build paper-ready results tables
# ---------------------------------------------------------------------------
paper-results:
	@echo ">>> Building paper-ready results tables"
	$(PYTHON) -m scripts.build_paper_results \
		--comparisons-dir $(COMPARISONS_DIR) \
		--out-dir $(PAPER_RESULTS_DIR) \
		--tables-dir $(PAPER_TABLES_DIR) \
		--selection-metric $(PAPER_SELECTION_METRIC) \
		--exclude-datasets $(PAPER_EXCLUDE_DATASETS) \
		--exclude-methods $(PAPER_EXCLUDE_METHODS) \
		--metrics $(PAPER_METRICS)

restore-published-results:
	@echo ">>> Restoring bundled published results into $(COMPARISONS_DIR)"
	$(PYTHON) -m scripts.restore_published_results \
		--published-dir $(PUBLISHED_RESULTS_DIR) \
		--comparisons-dir $(COMPARISONS_DIR) \
		--overwrite

validate-published:
	@echo ">>> Validating regenerated paper CSVs against $(PUBLISHED_RESULTS_DIR)"
	$(PYTHON) -m scripts.validate_against_published \
		--actual-dir $(PAPER_RESULTS_DIR) \
		--published-dir $(PUBLISHED_RESULTS_DIR)

compare-to-published: restore-published-results paper-results validate-published

ablation-tables:
	@echo ">>> Building ALE--Frechet ablation tables for: $(ABLATION_DATASETS)"
	$(PYTHON) scripts/build_ale_frechet_ablation_tables.py \
		--comparisons-root $(COMPARISONS_DIR) \
		--out-dir $(ABLATION_RESULTS_DIR) \
		--tables-dir $(ABLATION_TABLES_DIR) \
		--datasets $(ABLATION_DATASETS) \
		--selection-metric $(ABLATION_SELECTION_METRIC) \
		--metrics $(ABLATION_METRICS) \
		--top-k $(ABLATION_TOPK)

similarity-update-table:
	@echo ">>> Building similarity update-frequency table for: $(SIMILARITY_UPDATE_DATASETS)"
	$(PYTHON) scripts/build_similarity_update_relative_table.py \
		--comparisons-root $(COMPARISONS_DIR) \
		--out-dir $(SIMILARITY_UPDATE_RESULTS_DIR) \
		--tables-dir $(SIMILARITY_UPDATE_TABLES_DIR) \
		--datasets $(SIMILARITY_UPDATE_DATASETS) \
		--selection-metric $(SIMILARITY_UPDATE_SELECTION_METRIC) \
		--metrics $(SIMILARITY_UPDATE_METRICS)

component-ablation-sweep:
	$(call require_dataset,component-ablation-sweep)
	@echo ">>> Running ALE--Frechet component ablation for dataset: $(DATASET)"
	$(CUDA_ENV) $(PYTHON) -m scripts.sweep_compare_methods \
		--global-config $(GLOBAL_CFG) \
		--dataset-config $(DATASETS)/$(DATASET).yaml \
		--ale-model-config $(CONFIGS)/models/$(DATASET).yaml \
		--baseline-model-config $(COMPARE_BASELINE_MODEL_CFG) \
		--sweep-config $(COMPONENT_ABLATION_SWEEP_CFG) \
		--method-config $(COMPARE_METHOD_CFG) \
		--out-dir $(COMPONENT_ABLATION_OUT_ROOT)/$(DATASET)

component-ablation-aggregate:
	$(call require_dataset,component-ablation-aggregate)
	@echo ">>> Aggregating ALE--Frechet component ablation for dataset: $(DATASET)"
	$(PYTHON) -m scripts.aggregate_compare_results \
		--run-dir $(COMPONENT_ABLATION_OUT_ROOT)/$(DATASET)/sweep_runs \
		--out-dir $(COMPONENT_ABLATION_OUT_ROOT)/$(DATASET)

component-ablation-table:
	@echo ">>> Building ALE--Frechet component ablation table"
	$(PYTHON) scripts/build_component_ablation_table.py \
		--results-root $(COMPONENT_ABLATION_OUT_ROOT) \
		--out-dir $(COMPONENT_ABLATION_RESULTS_DIR) \
		--tables-dir $(PAPER_TABLES_DIR) \
		--datasets $(COMPONENT_ABLATION_DATASETS)

component-ablation:
	@for ds in $(COMPONENT_ABLATION_DATASETS); do \
		$(MAKE) component-ablation-sweep DATASET=$$ds || exit 1; \
		$(MAKE) component-ablation-aggregate DATASET=$$ds || exit 1; \
	done
	$(MAKE) component-ablation-table

interpretability-figures:
	@echo ">>> Building interpretability figures"
	$(PYTHON) scripts/build_interpretability_figures.py \
		--comparisons-root $(COMPARISONS_DIR) \
		--logging-root results \
		--out-dir $(INTERPRETABILITY_FIGURES_DIR)

interpretability-validation:
	@echo ">>> Building quantitative interpretability validation"
	$(PYTHON) scripts/build_interpretability_validation.py \
		--datasets $(INTERPRETABILITY_VALIDATION_DATASETS) \
		--comparisons-root $(COMPARISONS_DIR) \
		--logging-root results \
		--out-dir $(INTERPRETABILITY_VALIDATION_RESULTS_DIR) \
		--tables-dir $(PAPER_TABLES_DIR)

clean-paper-results:
	rm -rf $(PAPER_RESULTS_DIR)

learning-curves:
	$(call require_dataset,learning-curves)
	@echo ">>> Building learning-curves figure for dataset: $(DATASET)"
	$(PYTHON) scripts/build_learning_curves.py \
		--dataset $(DATASET) \
		--comparisons-root $(COMPARE_OUT_ROOT) \
		--logging-root results \
		--out-dir paper/figures/learning_curves

ale-frechet-ablation-curves:
	$(call require_dataset,ale-frechet-ablation-curves)
	@echo ">>> Building ALE–Fréchet ablation learning curves for dataset: $(DATASET)"
	$(PYTHON) scripts/build_ale_frechet_ablation_curves.py \
		--dataset $(DATASET) \
		--comparisons-root $(COMPARE_OUT_ROOT) \
		--logging-root results \
		--out-dir paper/figures/ablation

.PHONY: \
	help print-datasets setup \
	data-synth vis-synth generate-real \
	train-compare train-compare-all \
	sweep-compare aggregate-compare sweep-compare-full \
	sweep-compare-all aggregate-compare-all sweep-compare-full-all \
	rank-methods rank-methods-all latex-ranking latex-ranking-all \
	build-repro-package backup backup-repro backup-results backup-paper backup-data \
	sweep-synthetic paper-results restore-published-results validate-published compare-to-published ablation-tables component-ablation component-ablation-sweep component-ablation-aggregate component-ablation-table similarity-update-table interpretability-figures interpretability-validation clean-paper-results \
	learning-curves ale-frechet-ablation-curves
