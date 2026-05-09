#!/bin/bash -l
# One-time venv setup on a compute node (aarch64/GH200).
# Must run on a compute node — login node has no PyPI access and is x86_64.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:0
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=logs/setup-venv-%j.log
#SBATCH --error=logs/setup-venv-%j.err

ml ML-bundle/25.10

set -euo pipefail

# Helios sets PIP_EXTRA_INDEX_URL to an internal mirror with a limited package set.
# Unset it so pip uses the real PyPI.
unset PIP_EXTRA_INDEX_URL

SCRATCH_ROOT=${SCRATCH}/cpt_experiments
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv

mkdir -p logs

echo "Setting up Python venv at ${VENV_DIR}..."

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
    echo "Venv created."
fi

source "${VENV_DIR}/bin/activate"

VENV_MARKER="${VENV_DIR}/.cpt_deps_installed"
if [ ! -f "${VENV_MARKER}" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip -q
    pip install -r "${REPO_DIR}/requirements.txt" -q
    touch "${VENV_MARKER}"
    echo "Dependencies installed."
else
    echo "Dependencies already installed."
fi

echo "Venv ready."
