#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <target> <dataset> [extra make args...]"
  echo "Example: $0 sweep-compare-full electricity COMPARE_SEEDS=0,1,2"
  exit 2
fi

TARGET="$1"
DATASET="$2"
shift 2

LOG_DIR="${LOG_DIR:-logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${TARGET}__${DATASET}__${STAMP}.log"

echo ">>> Logging to ${LOG_FILE}"
make "${TARGET}" DATASET="${DATASET}" "$@" 2>&1 | tee "${LOG_FILE}"
