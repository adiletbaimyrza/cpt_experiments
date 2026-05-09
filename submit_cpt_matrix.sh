#!/bin/bash
# Submit the full CPT matrix: 3 models × (grid search → 3 language variants).
# Each model runs an independent automated pipeline.
#
# Usage:
#   bash submit_cpt_matrix.sh [words|tokens]

set -euo pipefail

EXPERIMENT=${1:-words}

if [ "${EXPERIMENT}" != "words" ] && [ "${EXPERIMENT}" != "tokens" ]; then
    echo "ERROR: EXPERIMENT must be words or tokens"
    exit 1
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

declare -A DATASET_IDS_WORDS=(
    [FT-KY]="TBD/kyrgyz-100m-words"
    [FT-KZ]="TBD/kazakh-100m-words"
    [FT-PL]="TBD/polish-100m-words"
)

declare -A DATASET_IDS_TOKENS=(
    [FT-KY]="TBD/kyrgyz-100m-tokens"
    [FT-KZ]="TBD/kazakh-100m-tokens"
    [FT-PL]="TBD/polish-100m-tokens"
)

ENGLISH_DATASET_ID="TBD/english-100m-words"

if [ "${EXPERIMENT}" = "words" ]; then
    declare -n DATASET_IDS=DATASET_IDS_WORDS
else
    declare -n DATASET_IDS=DATASET_IDS_TOKENS
fi

echo "=========================================="
echo "CPT Matrix Submission"
echo "=========================================="
echo "Experiment:      ${EXPERIMENT}"
echo "English dataset: ${ENGLISH_DATASET_ID}"
echo "Epochs:          3 (max_steps auto-computed per language at runtime)"
echo "=========================================="
echo ""

# Validate placeholders
if [[ "${ENGLISH_DATASET_ID}" == TBD/* ]]; then
    echo "ERROR: Fill ENGLISH_DATASET_ID in submit_cpt_matrix.sh before running."
    exit 1
fi
for variant in FT-KY FT-KZ FT-PL; do
    if [[ "${DATASET_IDS[$variant]}" == TBD/* ]]; then
        echo "ERROR: Fill DATASET_IDS_${EXPERIMENT^^}[${variant}] in submit_cpt_matrix.sh before running."
        exit 1
    fi
done

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
