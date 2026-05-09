#!/bin/bash
# Submit one CPT pipeline.
#
# With SKIP_GRID_SEARCH=true:
#   data prep -> train
#
# With SKIP_GRID_SEARCH=false:
#   data prep -> FT-KY grid -> pick winner, then stop so configs can be updated.
#
# Usage:
#   bash cpt/submit_cpt_pipeline.sh MODEL DATASET_ID LANG_VARIANT EXPERIMENT CONFIG_FILE MAX_STEPS SKIP_GRID_SEARCH ENGLISH_DATASET_ID [RUN_ID]

set -euo pipefail

MODEL=${1:?"MODEL required"}
DATASET_ID=${2:?"DATASET_ID required"}
LANG_VARIANT=${3:?"LANG_VARIANT required"}
EXPERIMENT=${4:?"EXPERIMENT required: words|tokens"}
CONFIG_FILE=${5:?"CONFIG_FILE required"}
MAX_STEPS=${6:-20000}
SKIP_GRID_SEARCH=${7:-true}
ENGLISH_DATASET_ID=${8:?"ENGLISH_DATASET_ID required"}
RUN_ID=${9:-${CPT_RUN_ID:-resume}}

if [ "${EXPERIMENT}" != "words" ] && [ "${EXPERIMENT}" != "tokens" ]; then
    echo "ERROR: EXPERIMENT must be words or tokens"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}"
    exit 1
fi

for required in cpt/jobs/prepare_cpt_data.sh cpt/jobs/grid_search.sh cpt/jobs/pick_best_grid.sh cpt/jobs/train_cpt.sh; do
    if [ ! -f "${required}" ]; then
        echo "ERROR: missing ${required}"
        exit 1
    fi
done

mkdir -p cpt/logs

MODEL_SHORT="${MODEL##*/}"
DATASET_ID_SAFE=$(printf '%s' "${DATASET_ID}" | tr '/:' '__')
DATASET_SAFE=$(printf '%s' "${LANG_VARIANT}_${EXPERIMENT}_${MODEL_SHORT}_${DATASET_ID_SAFE}" | tr '/:' '__')
BUDGET=100000000

read_config_field() {
    local path=$1
    python3 - "${CONFIG_FILE}" "${path}" <<'PY'
import sys, yaml
config_path, dotted = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    data = yaml.safe_load(f)
cur = data
for part in dotted.split("."):
    cur = cur[part]
print(cur)
PY
}

LORA_R=$(read_config_field "lora.r")
LR=$(read_config_field "training.learning_rate")
GRID_MAX_STEPS=$(read_config_field "grid_search.grid_max_steps")

ADAPTER_REL="checkpoints/cpt_${MODEL_SHORT}_${LANG_VARIANT}_${DATASET_SAFE}_${RUN_ID}/final"

echo "=========================================="
echo "CPT Pipeline Submission"
echo "=========================================="
echo "Model:            ${MODEL}"
echo "Dataset ID:       ${DATASET_ID}"
echo "Prepared dataset: ${DATASET_SAFE}"
echo "Lang variant:     ${LANG_VARIANT}"
echo "Experiment:       ${EXPERIMENT}"
echo "Config:           ${CONFIG_FILE}"
echo "LoRA rank:        ${LORA_R}"
echo "LR:               ${LR}"
echo "Max steps:        ${MAX_STEPS}"
echo "Grid search:      $([ "${SKIP_GRID_SEARCH}" = "true" ] && echo skip || echo run-and-stop)"
echo "Run ID:           ${RUN_ID}"
echo "Final adapter:    ${ADAPTER_REL}"
echo "=========================================="
echo ""

PREP_JOB_ID=$(sbatch \
    --parsable \
    cpt/jobs/prepare_cpt_data.sh \
    "${DATASET_ID}" "${MODEL}" "${LANG_VARIANT}" "${EXPERIMENT}" "${BUDGET}" "${DATASET_SAFE}" "${ENGLISH_DATASET_ID}")

echo "Data prep job: ${PREP_JOB_ID}"

TRAIN_DEP="${PREP_JOB_ID}"

if [ "${SKIP_GRID_SEARCH}" != "true" ]; then
    if [ "${LANG_VARIANT}" != "FT-KY" ]; then
        echo "ERROR: grid search is only defined for FT-KY. Use SKIP_GRID_SEARCH=true for ${LANG_VARIANT}."
        exit 1
    fi

    GRID_JOB_ID=$(sbatch \
        --parsable \
        --dependency=afterok:${PREP_JOB_ID} \
        cpt/jobs/grid_search.sh \
        "${MODEL}" "${DATASET_SAFE}" "${GRID_MAX_STEPS}")
    echo "Grid search array job: ${GRID_JOB_ID}"

    PICK_JOB_ID=$(sbatch \
        --parsable \
        --dependency=afterany:${GRID_JOB_ID} \
        cpt/jobs/pick_best_grid.sh \
        "${MODEL_SHORT}" "${GRID_JOB_ID}")
    echo "Grid winner job: ${PICK_JOB_ID}"
    echo ""
    echo "Submitted grid-search chain:"
    echo "  prep ${PREP_JOB_ID} -> grid ${GRID_JOB_ID} -> pick ${PICK_JOB_ID}"
    echo "Winner file:"
    echo "  cpt/logs/grid_winner_${MODEL_SHORT}.txt"
    echo ""
    echo "Update ${CONFIG_FILE} with the winning lora.r and training.learning_rate,"
    echo "then rerun this script with SKIP_GRID_SEARCH=true for full training."
    exit 0
fi

TRAIN_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:${TRAIN_DEP} \
    cpt/jobs/train_cpt.sh \
    "${MODEL}" "${DATASET_SAFE}" "${LANG_VARIANT}" "${LORA_R}" "${LR}" "${MAX_STEPS}" "${RUN_ID}")

echo "Training job: ${TRAIN_JOB_ID}"

echo ""
echo "Submitted chain:"
echo "  prep ${PREP_JOB_ID} -> train ${TRAIN_JOB_ID}"
echo "Final adapter will be saved at:"
echo "  ${ADAPTER_REL}"
