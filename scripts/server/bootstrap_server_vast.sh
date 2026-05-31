#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

INSTALL_DEV="${INSTALL_DEV:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"
VAST_VENV_DIR="${VAST_VENV_DIR:-/venv/main}"

version_lt() {
  [[ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" != "$2" ]]
}

detect_gpu_compute_capability() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]'
}

backend_supports_blackwell() {
  local backend="$1"
  if [[ "${backend}" == "cpu" ]]; then
    return 1
  fi
  if [[ "${backend}" =~ ^cu([0-9]+)$ ]]; then
    [[ "${BASH_REMATCH[1]}" -ge 128 ]]
    return
  fi
  return 1
}

if [[ ! -f "${VAST_VENV_DIR}/bin/activate" ]]; then
  echo "Expected Vast main environment at ${VAST_VENV_DIR}, but it was not found."
  echo "Use scripts/server/bootstrap_server.sh for non-Vast images."
  exit 2
fi

echo ">>> Activating Vast Python environment at ${VAST_VENV_DIR}"
# shellcheck disable=SC1090
source "${VAST_VENV_DIR}/bin/activate"

GPU_COMPUTE_CAPABILITY="$(detect_gpu_compute_capability || true)"
EFFECTIVE_TORCH_BACKEND="${TORCH_BACKEND}"
if [[ -n "${GPU_COMPUTE_CAPABILITY}" ]]; then
  echo ">>> Detected GPU compute capability: ${GPU_COMPUTE_CAPABILITY}"
  if [[ "${GPU_COMPUTE_CAPABILITY}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    GPU_CC_MAJOR="${BASH_REMATCH[1]}"
    if [[ "${GPU_CC_MAJOR}" -ge 12 ]]; then
      if version_lt "${TORCH_VERSION}" "2.7.0"; then
        echo "PyTorch ${TORCH_VERSION} is too old for Blackwell GPUs (compute capability ${GPU_COMPUTE_CAPABILITY})."
        echo "Use torch>=2.7.0 with a CUDA 12.8+ backend."
        exit 2
      fi
      if [[ "${EFFECTIVE_TORCH_BACKEND}" == "auto" ]]; then
        EFFECTIVE_TORCH_BACKEND="cu128"
      elif ! backend_supports_blackwell "${EFFECTIVE_TORCH_BACKEND}"; then
        echo "TORCH_BACKEND='${EFFECTIVE_TORCH_BACKEND}' is incompatible with Blackwell GPUs."
        echo "Use TORCH_BACKEND=cu128 (or newer, e.g. cu130)."
        exit 2
      fi
    fi
  fi
fi

INSTALLER="pip"
if command -v uv >/dev/null 2>&1; then
  INSTALLER="uv pip"
fi

echo ">>> Upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

echo ">>> Installing PyTorch ${TORCH_VERSION} / torchvision ${TORCHVISION_VERSION} with backend '${EFFECTIVE_TORCH_BACKEND}'"
if [[ "${INSTALLER}" == "uv pip" ]]; then
  uv pip install \
    --python "${VAST_VENV_DIR}/bin/python" \
    --torch-backend "${EFFECTIVE_TORCH_BACKEND}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"
else
  python -m pip install \
    --index-url "https://download.pytorch.org/whl/${EFFECTIVE_TORCH_BACKEND}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"
fi

echo ">>> Installing repository dependencies"
python -m pip install -r requirements.txt

if [[ "${INSTALL_DEV}" == "1" ]]; then
  echo ">>> Installing development dependencies"
  python -m pip install -r requirements-dev.txt
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  echo ">>> Running environment preflight"
  python scripts/server/preflight_check.py
fi

echo ">>> Vast bootstrap completed successfully"
