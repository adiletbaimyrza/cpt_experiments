#!/bin/bash -l
# CPU data preparation job for CPT experiments.
#
# Arguments:
#   $1  DATASET_ID          HF dataset ID (target language)
#   $2  TOKENIZER_ID        HF model ID for token counting
#   $3  LANG_VARIANT        FT-KY | FT-KZ | FT-PL
#   $4  EXPERIMENT          words | tokens
#   $5  BUDGET              word or token budget (default: 100000000)
#   $6  OUTPUT_NAME         name under data/cpt_processed/
#   $7  ENGLISH_DATASET_ID  HF dataset ID for English mix

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=logs/prepare-cpt-%j.log
#SBATCH --error=logs/prepare-cpt-%j.err

ml ML-bundle/25.10

set -euo pipefail

DATASET_ID=${1:?"DATASET_ID required"}
TOKENIZER_ID=${2:?"TOKENIZER_ID required"}
LANG_VARIANT=${3:?"LANG_VARIANT required"}
EXPERIMENT=${4:?"EXPERIMENT required (words|tokens)"}
BUDGET=${5:-100000000}
OUTPUT_NAME=${6:?"OUTPUT_NAME required"}
ENGLISH_DATASET_ID=${7:?"ENGLISH_DATASET_ID required"}

SCRATCH_ROOT=${SCRATCH}/cpt_experiments
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv
HF_HOME=${SCRATCH_ROOT}/cache

echo "=========================================="
echo "CPT: Data Preparation"
echo "=========================================="
echo "Dataset:         ${DATASET_ID}"
echo "Tokenizer:       ${TOKENIZER_ID}"
echo "Lang variant:    ${LANG_VARIANT}"
echo "Experiment:      ${EXPERIMENT}"
echo "Budget:          ${BUDGET}"
echo "Output name:     ${OUTPUT_NAME}"
echo "English dataset: ${ENGLISH_DATASET_ID}"
echo "=========================================="
echo ""

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

export HF_HOME
export TRANSFORMERS_CACHE=${HF_HOME}

ENV_FILE="${SCRATCH_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a; source "${ENV_FILE}"; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Add it to ${ENV_FILE}."
    exit 1
fi
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

mkdir -p logs

OUTPUT_DIR="${REPO_DIR}/data/cpt_processed/${OUTPUT_NAME}"

if [ -f "${OUTPUT_DIR}/data_stats.json" ] && [ "${FORCE_PREP:-false}" != "true" ]; then
    echo "Existing processed CPT dataset found, skipping data preparation:"
    echo "  ${OUTPUT_DIR}"
    echo "Set FORCE_PREP=true to rebuild it."
    exit 0
fi

BUDGET_ARG=""
if [ "${EXPERIMENT}" = "words" ]; then
    BUDGET_ARG="--word_budget ${BUDGET}"
else
    BUDGET_ARG="--token_budget ${BUDGET}"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting data preparation..."
echo ""

python scripts/prepare_cpt_data.py \
    --dataset_id "${DATASET_ID}" \
    --tokenizer_id "${TOKENIZER_ID}" \
    --lang_variant "${LANG_VARIANT}" \
    --experiment "${EXPERIMENT}" \
    ${BUDGET_ARG} \
    --english_dataset_id "${ENGLISH_DATASET_ID}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Data preparation complete."
echo "Output: ${OUTPUT_DIR}"
