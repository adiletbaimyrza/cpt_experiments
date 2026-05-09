#!/bin/bash -l
# One-shot setup and submission script for CPT experiments on Helios.
# Safe to run multiple times — all setup steps are idempotent.
#
# Submits the full automated pipeline for each model:
#   FT-KY data prep → grid search → pick winner → patch config → train all 3 variants
#
# Usage:
#   bash setup_and_submit.sh [words|tokens]
#
# Fill in dataset IDs in submit_cpt_matrix.sh before running.

ml ML-bundle/25.10

set -euo pipefail

EXPERIMENT=${1:-words}

if [ "${EXPERIMENT}" != "words" ] && [ "${EXPERIMENT}" != "tokens" ]; then
    echo "ERROR: EXPERIMENT must be 'words' or 'tokens'"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH_ROOT="${SCRATCH}/cpt_experiments"
REPO_DIR="${SCRATCH_ROOT}"
VENV_DIR="${SCRATCH_ROOT}/venv"
HF_HOME="${SCRATCH_ROOT}/cache"

RUN_DATE=$(date +%Y-%m-%d)
LOG_DIR="${REPO_DIR}/logs/${RUN_DATE}"
export CPT_LOG_DIR="${LOG_DIR}"

echo "=========================================="
echo "CPT Setup and Submit"
echo "=========================================="
echo "Experiment:       ${EXPERIMENT}"
echo "Scratch root:     ${SCRATCH_ROOT}"
echo "Log dir:          ${LOG_DIR}"
echo "=========================================="
echo ""

# ── [1/5] Directories (idempotent) ────────────────────────────────────────────
echo "[1/5] Creating directories..."
mkdir -p "${LOG_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${REPO_DIR}/data/cpt_processed"
mkdir -p "${REPO_DIR}/checkpoints"
echo "  logs/${RUN_DATE}/      OK"
echo "  cache/                 OK"
echo "  data/cpt_processed/    OK"
echo "  checkpoints/           OK"
echo ""

# ── [2/5] Python venv ─────────────────────────────────────────────────────────
# pip install must run on a GH200 compute node (login node is x86_64 with no PyPI access).
echo "[2/5] Checking Python venv..."
VENV_MARKER="${VENV_DIR}/.cpt_deps_installed"
if [ ! -f "${VENV_MARKER}" ]; then
    echo "  Venv not ready — submitting setup job on compute node..."
    export CPT_SETUP_JOB_ID=$(sbatch \
        --parsable \
        --output="${LOG_DIR}/setup-venv-%j.log" \
        --error="${LOG_DIR}/setup-venv-%j.err" \
        jobs/setup_venv.sh)
    echo "  Setup job: ${CPT_SETUP_JOB_ID}"
    echo "  Pipeline jobs will wait for it via afterok dependency."
else
    echo "  Venv already ready."
    unset CPT_SETUP_JOB_ID
fi
echo ""

# ── [3/5] HF_TOKEN ────────────────────────────────────────────────────────────
echo "[3/5] Checking HF_TOKEN..."
ENV_FILE="${SCRATCH_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a; source "${ENV_FILE}"; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo ""
    echo "Create ${ENV_FILE} with one line:"
    echo "  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"
    exit 1
fi
echo "  HF_TOKEN found."
echo ""

# ── [4/5] Dataset ID check ────────────────────────────────────────────────────
echo "[4/5] Checking dataset IDs for experiment '${EXPERIMENT}'..."
if [ "${EXPERIMENT}" = "words" ]; then
    _REQUIRED_VARS=(CPT_DATASET_FT_KY_WORDS CPT_DATASET_FT_KZ_WORDS CPT_DATASET_FT_PL_WORDS CPT_DATASET_ENGLISH)
else
    _REQUIRED_VARS=(CPT_DATASET_FT_KY_TOKENS CPT_DATASET_FT_KZ_TOKENS CPT_DATASET_FT_PL_TOKENS CPT_DATASET_ENGLISH)
fi
_MISSING=0
for _VAR in "${_REQUIRED_VARS[@]}"; do
    if [ -z "${!_VAR:-}" ]; then
        echo "  ERROR: ${_VAR} is not set in ${ENV_FILE}"
        _MISSING=$((_MISSING + 1))
    fi
done
if [ "${_MISSING}" -gt 0 ]; then
    echo ""
    echo "Add the missing variables to ${ENV_FILE} and rerun."
    exit 1
fi
echo "  All dataset IDs found."
echo ""

# ── [5/5] Submit ──────────────────────────────────────────────────────────────
echo "[5/5] Submitting pipeline..."
echo ""
cd "${REPO_DIR}"
bash submit_cpt_matrix.sh "${EXPERIMENT}"
