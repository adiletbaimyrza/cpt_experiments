#!/bin/bash -l
# Upload all CPT adapters to the HuggingFace Hub.
# Run on a compute node so the upload doesn't fight the login-node bandwidth cap.
#
# Usage:
#   sbatch --job-name=upload-adapters --output=logs/%x-%j.log --error=logs/%x-%j.err \
#       jobs/upload_adapters.sh <hf_org_or_user> [--private] [--experiment words|tokens|both] [--dry-run]
#
# HF_TOKEN must be in $SCRATCH/cpt_experiments/.env with WRITE scope.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err

ml ML-bundle/25.10

set -euo pipefail

ORG=${1:?"first arg must be the HF org/username"}
shift || true  # remaining args (--private, --experiment, --dry-run) forwarded to Python

SCRATCH_ROOT=${SCRATCH}/cpt_experiments
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

ENV_FILE="${SCRATCH_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a; source "${ENV_FILE}"; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Add it (with write scope) to ${ENV_FILE}."
    exit 1
fi
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

mkdir -p logs

echo "=========================================="
echo "Uploading adapters to HF org: ${ORG}"
echo "Args: $*"
echo "Source: ${SCRATCH_ROOT}/checkpoints"
echo "=========================================="

python scripts/upload_adapters.py --org "${ORG}" "$@"

echo ""
echo "Upload job complete."
