#!/bin/bash
# Submit the 3-model x 3-variant CPT matrix.
#
# Usage:
#   bash cpt/submit_cpt_matrix.sh [words|tokens] [skip_grid_search]
#
# max_steps is auto-computed per language at runtime from dataset size (3 epochs).
# With skip_grid_search=false, this submits FT-KY grid searches only and exits.
# Update configs from winner files, then rerun with skip_grid_search=true.

set -euo pipefail

EXPERIMENT=${1:-words}
SKIP_GRID_SEARCH=${2:-true}

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
    "cpt/configs/llama_cpt.yaml"
    "cpt/configs/qwen_cpt.yaml"
    "cpt/configs/gemma_cpt.yaml"
)

LANG_VARIANTS=("FT-KY" "FT-KZ" "FT-PL")

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
echo "Experiment:       ${EXPERIMENT}"
echo "Epochs:           3 (max_steps auto-computed per language at runtime)"
echo "Default grid:     $([ "${SKIP_GRID_SEARCH}" = "true" ] && echo skip || echo run-ft-ky-only)"
echo "English dataset:  ${ENGLISH_DATASET_ID}"
echo "=========================================="
echo ""

if [[ "${ENGLISH_DATASET_ID}" == TBD/* ]]; then
    echo "ERROR: Fill ENGLISH_DATASET_ID in cpt/submit_cpt_matrix.sh before running."
    exit 1
fi

for variant in "${LANG_VARIANTS[@]}"; do
    if [[ "${DATASET_IDS[$variant]}" == TBD/* ]]; then
        echo "ERROR: Fill DATASET_IDS_${EXPERIMENT^^}[${variant}] in cpt/submit_cpt_matrix.sh before running."
        exit 1
    fi
done

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    CONFIG="${CONFIGS[$i]}"

    if [ "${SKIP_GRID_SEARCH}" != "true" ]; then
        echo "Submitting FT-KY grid search for ${MODEL}"
        bash cpt/submit_cpt_pipeline.sh \
            "${MODEL}" \
            "${DATASET_IDS[FT-KY]}" \
            "FT-KY" \
            "${EXPERIMENT}" \
            "${CONFIG}" \
            "false" \
            "${ENGLISH_DATASET_ID}"
        echo ""
        continue
    fi

    for VARIANT in "${LANG_VARIANTS[@]}"; do
        echo "Submitting ${MODEL} / ${VARIANT}"
        bash cpt/submit_cpt_pipeline.sh \
            "${MODEL}" \
            "${DATASET_IDS[$VARIANT]}" \
            "${VARIANT}" \
            "${EXPERIMENT}" \
            "${CONFIG}" \
            "true" \
            "${ENGLISH_DATASET_ID}"
        echo ""
    done
done

if [ "${SKIP_GRID_SEARCH}" != "true" ]; then
    echo "All FT-KY grid-search chains submitted."
    echo "Update cpt/configs/*.yaml from cpt/logs/grid_winner_*.txt, then rerun with skip_grid_search=true."
else
    echo "All CPT pipelines submitted."
fi
echo "Monitor: squeue -u \$(whoami)"
