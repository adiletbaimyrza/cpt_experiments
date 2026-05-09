#!/bin/bash
# Submit one full CPT pipeline for a single model.
#
# Chain:
#   FT-KY data prep
#     → grid search (4-run array on FT-KY)
#     → pick winner
#     → apply winner + train all 3 variants (FT-KY, FT-KZ, FT-PL)
#
# Usage:
#   bash submit_cpt_pipeline.sh MODEL EXPERIMENT CONFIG_FILE ENGLISH_DATASET_ID \
#                               DATASET_FT_KY DATASET_FT_KZ DATASET_FT_PL [RUN_ID]

set -euo pipefail

MODEL=${1:?"MODEL required"}
EXPERIMENT=${2:?"EXPERIMENT required: words|tokens"}
CONFIG_FILE=${3:?"CONFIG_FILE required"}
ENGLISH_DATASET_ID=${4:?"ENGLISH_DATASET_ID required"}
DATASET_FT_KY=${5:?"DATASET_FT_KY required"}
DATASET_FT_KZ=${6:?"DATASET_FT_KZ required"}
DATASET_FT_PL=${7:?"DATASET_FT_PL required"}
RUN_ID=${8:-${CPT_RUN_ID:-resume}}

if [ "${EXPERIMENT}" != "words" ] && [ "${EXPERIMENT}" != "tokens" ]; then
    echo "ERROR: EXPERIMENT must be words or tokens"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}"
    exit 1
fi

for required in jobs/prepare_cpt_data.sh jobs/grid_search.sh jobs/pick_best_grid.sh \
                jobs/apply_winner_and_train.sh jobs/train_cpt.sh; do
    if [ ! -f "${required}" ]; then
        echo "ERROR: missing ${required}"
        exit 1
    fi
done

mkdir -p logs

# Route SLURM logs into date-stamped subdir when set by setup_and_submit.sh
_LOG="${CPT_LOG_DIR:-logs}"
mkdir -p "${_LOG}"

MODEL_SHORT="${MODEL##*/}"
DATASET_ID_SAFE=$(printf '%s' "${DATASET_FT_KY}" | tr '/:' '__')
DATASET_SAFE_FT_KY=$(printf '%s' "FT-KY_${EXPERIMENT}_${MODEL_SHORT}_${DATASET_ID_SAFE}" | tr '/:' '__')
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

GRID_MAX_STEPS=$(read_config_field "grid_search.grid_max_steps")

echo "=========================================="
echo "CPT Pipeline Submission"
echo "=========================================="
echo "Model:        ${MODEL}"
echo "Experiment:   ${EXPERIMENT}"
echo "Config:       ${CONFIG_FILE}"
echo "Grid steps:   ${GRID_MAX_STEPS}"
echo "Run ID:       ${RUN_ID}"
echo "Log dir:      ${_LOG}"
echo "=========================================="
echo ""

# Step 1: FT-KY data prep (optionally wait for venv setup job)
_PREP_DEPS=()
if [ -n "${CPT_SETUP_JOB_ID:-}" ]; then
    _PREP_DEPS=(--dependency=afterok:${CPT_SETUP_JOB_ID})
fi
PREP_JOB_ID=$(sbatch \
    --parsable \
    "${_PREP_DEPS[@]}" \
    --output="${_LOG}/prepare-cpt-%j.log" \
    --error="${_LOG}/prepare-cpt-%j.err" \
    jobs/prepare_cpt_data.sh \
    "${DATASET_FT_KY}" "${MODEL}" "FT-KY" "${EXPERIMENT}" "${BUDGET}" "${DATASET_SAFE_FT_KY}" "${ENGLISH_DATASET_ID}")
echo "Data prep job:    ${PREP_JOB_ID}"

# Step 2: Grid search on FT-KY (4-run array)
GRID_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:${PREP_JOB_ID} \
    --output="${_LOG}/grid-search-%A-%a.log" \
    --error="${_LOG}/grid-search-%A-%a.err" \
    jobs/grid_search.sh \
    "${MODEL}" "${DATASET_SAFE_FT_KY}" "${GRID_MAX_STEPS}")
echo "Grid search job:  ${GRID_JOB_ID}"

# Step 3: Pick best grid run
PICK_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterany:${GRID_JOB_ID} \
    --output="${_LOG}/grid-winner-%j.log" \
    --error="${_LOG}/grid-winner-%j.err" \
    jobs/pick_best_grid.sh \
    "${MODEL_SHORT}" "${GRID_JOB_ID}")
echo "Grid winner job:  ${PICK_JOB_ID}"

# Step 4: Apply winner to config + submit training for all 3 variants
APPLY_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:${PICK_JOB_ID} \
    --output="${_LOG}/apply-winner-%j.log" \
    --error="${_LOG}/apply-winner-%j.err" \
    jobs/apply_winner_and_train.sh \
    "${MODEL}" "${CONFIG_FILE}" "${EXPERIMENT}" "${ENGLISH_DATASET_ID}" \
    "${DATASET_FT_KY}" "${DATASET_FT_KZ}" "${DATASET_FT_PL}" "${RUN_ID}")
echo "Apply+train job:  ${APPLY_JOB_ID}"

echo ""
echo "Chain: prep ${PREP_JOB_ID} -> grid ${GRID_JOB_ID} -> pick ${PICK_JOB_ID} -> apply+train ${APPLY_JOB_ID}"
echo "(apply+train submits 3 prep+train chains at runtime, one per language variant)"
