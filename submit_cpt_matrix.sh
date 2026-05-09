#!/bin/bash
# Submit the full CPT matrix: 3 models × (grid search → 3 language variants).
# Dataset IDs are read from $SCRATCH/cpt_experiments/.env — never hardcoded here.
#
# Usage:
#   bash submit_cpt_matrix.sh [words|tokens]

set -euo pipefail

EXPERIMENT=${1:-words}

if [ "${EXPERIMENT}" != "words" ] && [ "${EXPERIMENT}" != "tokens" ]; then
    echo "ERROR: EXPERIMENT must be words or tokens"
    exit 1
fi

# Load .env (fallback for direct calls; setup_and_submit.sh exports these already)
_ENV_FILE="${SCRATCH}/cpt_experiments/.env"
if [ -f "${_ENV_FILE}" ]; then
    set -a; source "${_ENV_FILE}"; set +a
fi

MODELS=(
    "meta-llama/Llama-3.1-8B"
    "Qwen/Qwen3-8B-Base"
    "google/gemma-4-e4b"
)

CONFIGS=(
    "configs/llama_cpt.yaml"
    "configs/qwen_cpt.yaml"
    "configs/gemma_cpt.yaml"
)

if [ "${EXPERIMENT}" = "words" ]; then
    declare -A DATASET_IDS=(
        [FT-KY]="${CPT_DATASET_FT_KY_WORDS:-}"
        [FT-KZ]="${CPT_DATASET_FT_KZ_WORDS:-}"
        [FT-PL]="${CPT_DATASET_FT_PL_WORDS:-}"
    )
else
    declare -A DATASET_IDS=(
        [FT-KY]="${CPT_DATASET_FT_KY_TOKENS:-}"
        [FT-KZ]="${CPT_DATASET_FT_KZ_TOKENS:-}"
        [FT-PL]="${CPT_DATASET_FT_PL_TOKENS:-}"
    )
fi

ENGLISH_DATASET_ID="${CPT_DATASET_ENGLISH:-}"

# Validate all required dataset IDs are set
_MISSING=0
if [ -z "${ENGLISH_DATASET_ID}" ]; then
    echo "ERROR: CPT_DATASET_ENGLISH not set in .env"
    _MISSING=1
fi
for _VARIANT in FT-KY FT-KZ FT-PL; do
    if [ -z "${DATASET_IDS[$_VARIANT]}" ]; then
        _VARNAME="CPT_DATASET_${_VARIANT//-/_}_${EXPERIMENT^^}"
        echo "ERROR: ${_VARNAME} not set in .env"
        _MISSING=$((_MISSING + 1))
    fi
done
if [ "${_MISSING}" -gt 0 ]; then
    echo ""
    echo "Add the missing variables to ${_ENV_FILE} and rerun."
    exit 1
fi

echo "=========================================="
echo "CPT Matrix Submission"
echo "=========================================="
echo "Experiment:      ${EXPERIMENT}"
echo "English dataset: ${ENGLISH_DATASET_ID}"
echo "Epochs:          3 (max_steps auto-computed per language at runtime)"
echo "=========================================="
echo ""

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    CONFIG="${CONFIGS[$i]}"
    echo "Submitting pipeline for ${MODEL}"
    bash submit_cpt_pipeline.sh \
        "${MODEL}" \
        "${EXPERIMENT}" \
        "${CONFIG}" \
        "${ENGLISH_DATASET_ID}" \
        "${DATASET_IDS[FT-KY]}" \
        "${DATASET_IDS[FT-KZ]}" \
        "${DATASET_IDS[FT-PL]}"
    echo ""
done

echo "All pipelines submitted. Monitor: squeue -u \$(whoami)"
