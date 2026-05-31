#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CUDA_VARIANT="${CUDA_VARIANT:-cu128}"
INSTALL_DEV="${INSTALL_DEV:-0}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"

version_lt() {
  [[ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" != "$2" ]]
}

detect_gpu_compute_capability() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]'
}

case "${CUDA_VARIANT}" in
  cu118) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118" ;;
  cu126) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126" ;;
  cu128) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" ;;
  cpu)   TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" ;;
  *)
    echo "Unsupported CUDA_VARIANT='${CUDA_VARIANT}'. Use one of: cu118, cu126, cu128, cpu"
    exit 2
    ;;
esac

GPU_COMPUTE_CAPABILITY="$(detect_gpu_compute_capability || true)"
if [[ -n "${GPU_COMPUTE_CAPABILITY}" ]]; then
  echo ">>> Detected GPU compute capability: ${GPU_COMPUTE_CAPABILITY}"
  if [[ "${GPU_COMPUTE_CAPABILITY}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    GPU_CC_MAJOR="${BASH_REMATCH[1]}"
    if [[ "${GPU_CC_MAJOR}" -ge 12 ]]; then
      if version_lt "${TORCH_VERSION}" "2.7.0"; then
        echo "PyTorch ${TORCH_VERSION} is too old for Blackwell GPUs (compute capability ${GPU_COMPUTE_CAPABILITY})."
        echo "Use torch>=2.7.0 with CUDA 12.8+ wheels."
        exit 2
      fi
      if [[ "${CUDA_VARIANT}" != "cu128" ]]; then
        echo "CUDA_VARIANT='${CUDA_VARIANT}' is incompatible with Blackwell GPUs."
        echo "Use CUDA_VARIANT=cu128."
        exit 2
      fi
    fi
  fi
fi

echo ">>> Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo ">>> Upgrading pip"
pip install --upgrade pip setuptools wheel

echo ">>> Installing PyTorch ${TORCH_VERSION} / torchvision ${TORCHVISION_VERSION} (${CUDA_VARIANT})"
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

echo ">>> Installing repository dependencies"
pip install -r requirements.txt

if [[ "${INSTALL_DEV}" == "1" ]]; then
  echo ">>> Installing development dependencies"
  pip install -r requirements-dev.txt
fi

echo ">>> Running environment preflight"
python scripts/server/preflight_check.py

echo ">>> Bootstrap completed successfully"
