#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

INSTALL_DEV="${INSTALL_DEV:-0}"


echo ">>> Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo ">>> Upgrading pip"
pip install --upgrade pip setuptools wheel

echo ">>> Installing repository dependencies"
pip install -r requirements.txt

if [[ "${INSTALL_DEV}" == "1" ]]; then
  echo ">>> Installing development dependencies"
  pip install -r requirements-dev.txt
fi


echo ">>> Bootstrap completed successfully"
