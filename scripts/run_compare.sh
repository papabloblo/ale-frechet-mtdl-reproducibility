#!/usr/bin/env bash
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

