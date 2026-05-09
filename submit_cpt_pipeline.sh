#!/bin/bash
# Submit one full CPT pipeline for a single model — all jobs upfront from
# the login node, since PLGrid blocks recursive sbatch from compute nodes.
#
# Chain (parallel where possible):
#   prep_KY → grid (4-array) → pick
#   prep_KZ ─┐
#   prep_PL ─┤
#            ├── train_KY  (afterok: pick + prep_KY)
#            ├── train_KZ  (afterok: pick + prep_KZ)
#            └── train_PL  (afterok: pick + prep_PL)
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
                jobs/train_cpt.sh; do
    if [ ! -f "${required}" ]; then
        echo "ERROR: missing ${required}"
        exit 1
    fi
done

mkdir -p logs
_LOG="${CPT_LOG_DIR:-logs}"
mkdir -p "${_LOG}"

MODEL_SHORT="${MODEL##*/}"
BUDGET=100000000

# Build dataset-name strings (must match what train_cpt.sh expects)
make_dataset_safe() {
    local variant=$1 dataset_id=$2
    local id_safe
    id_safe=$(printf '%s' "${dataset_id}" | tr '/:' '__')
    printf '%s' "${variant}_${EXPERIMENT}_${MODEL_SHORT}_${id_safe}" | tr '/:' '__'
}

DATASET_SAFE_FT_KY=$(make_dataset_safe FT-KY "${DATASET_FT_KY}")
DATASET_SAFE_FT_KZ=$(make_dataset_safe FT-KZ "${DATASET_FT_KZ}")
DATASET_SAFE_FT_PL=$(make_dataset_safe FT-PL "${DATASET_FT_PL}")

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

# Setup-venv dependency for first-time runs
_SETUP_DEPS=()
if [ -n "${CPT_SETUP_JOB_ID:-}" ]; then
    _SETUP_DEPS=(--dependency=afterok:${CPT_SETUP_JOB_ID})
fi

# 1. Data prep for all 3 variants (parallel; KY also feeds the grid)
PREP_KY_JOB=$(sbatch --parsable "${_SETUP_DEPS[@]}" \
    --job-name="prep-${MODEL_SHORT}-FT-KY" \
    --output="${_LOG}/%x-%j.log" --error="${_LOG}/%x-%j.err" \
    jobs/prepare_cpt_data.sh \
    "${DATASET_FT_KY}" "${MODEL}" "FT-KY" "${EXPERIMENT}" "${BUDGET}" "${DATASET_SAFE_FT_KY}" "${ENGLISH_DATASET_ID}")
echo "Prep FT-KY:    ${PREP_KY_JOB}"

PREP_KZ_JOB=$(sbatch --parsable "${_SETUP_DEPS[@]}" \
    --job-name="prep-${MODEL_SHORT}-FT-KZ" \
    --output="${_LOG}/%x-%j.log" --error="${_LOG}/%x-%j.err" \
    jobs/prepare_cpt_data.sh \
    "${DATASET_FT_KZ}" "${MODEL}" "FT-KZ" "${EXPERIMENT}" "${BUDGET}" "${DATASET_SAFE_FT_KZ}" "${ENGLISH_DATASET_ID}")
echo "Prep FT-KZ:    ${PREP_KZ_JOB}"

PREP_PL_JOB=$(sbatch --parsable "${_SETUP_DEPS[@]}" \
    --job-name="prep-${MODEL_SHORT}-FT-PL" \
    --output="${_LOG}/%x-%j.log" --error="${_LOG}/%x-%j.err" \
    jobs/prepare_cpt_data.sh \
    "${DATASET_FT_PL}" "${MODEL}" "FT-PL" "${EXPERIMENT}" "${BUDGET}" "${DATASET_SAFE_FT_PL}" "${ENGLISH_DATASET_ID}")
echo "Prep FT-PL:    ${PREP_PL_JOB}"

# 2. Grid search on FT-KY — 4 separate jobs (A/B/C/D), readable filenames
declare -a GRID_JOB_IDS=()
for LABEL in A B C D; do
    GRID_JOB=$(sbatch --parsable \
        --dependency=afterok:${PREP_KY_JOB} \
        --job-name="grid-${MODEL_SHORT}-${LABEL}" \
        --output="${_LOG}/%x-%j.log" --error="${_LOG}/%x-%j.err" \
        jobs/grid_search.sh \
        "${MODEL}" "${DATASET_SAFE_FT_KY}" "${GRID_MAX_STEPS}" "${LABEL}")
    GRID_JOB_IDS+=("${GRID_JOB}")
    echo "Grid ${LABEL}:        ${GRID_JOB}"
done
GRID_DEP=$(IFS=:; echo "${GRID_JOB_IDS[*]}")  # job1:job2:job3:job4

# 3. Pick best grid run (waits on all 4 grid jobs, succeed or fail)
PICK_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:${GRID_DEP} \
    --job-name="pick-${MODEL_SHORT}" \
    --output="${_LOG}/%x-%j.log" --error="${_LOG}/%x-%j.err" \
    jobs/pick_best_grid.sh \
    "${MODEL_SHORT}" "${GRID_DEP}")
echo "Pick winner:   ${PICK_JOB_ID}"

# 4. Training for each variant (afterok: pick + own prep)
TRAIN_KY_JOB=$(sbatch --parsable \
    --dependency=afterok:${PICK_JOB_ID}:${PREP_KY_JOB} \
    --job-name="train-${MODEL_SHORT}-FT-KY" \
    --output="${_LOG}/%x-%j.log" --error="${_LOG}/%x-%j.err" \
    jobs/train_cpt.sh \
    "${MODEL}" "${DATASET_SAFE_FT_KY}" "FT-KY" "${RUN_ID}")
echo "Train FT-KY:   ${TRAIN_KY_JOB}"

TRAIN_KZ_JOB=$(sbatch --parsable \
    --dependency=afterok:${PICK_JOB_ID}:${PREP_KZ_JOB} \
    --job-name="train-${MODEL_SHORT}-FT-KZ" \
    --output="${_LOG}/%x-%j.log" --error="${_LOG}/%x-%j.err" \
    jobs/train_cpt.sh \
    "${MODEL}" "${DATASET_SAFE_FT_KZ}" "FT-KZ" "${RUN_ID}")
echo "Train FT-KZ:   ${TRAIN_KZ_JOB}"

TRAIN_PL_JOB=$(sbatch --parsable \
    --dependency=afterok:${PICK_JOB_ID}:${PREP_PL_JOB} \
    --job-name="train-${MODEL_SHORT}-FT-PL" \
    --output="${_LOG}/%x-%j.log" --error="${_LOG}/%x-%j.err" \
    jobs/train_cpt.sh \
    "${MODEL}" "${DATASET_SAFE_FT_PL}" "FT-PL" "${RUN_ID}")
echo "Train FT-PL:   ${TRAIN_PL_JOB}"

echo ""
echo "Chain summary:"
echo "  prep_KY ${PREP_KY_JOB} -> grid {A,B,C,D} -> pick ${PICK_JOB_ID}"
echo "  prep_KZ ${PREP_KZ_JOB}  (parallel)"
echo "  prep_PL ${PREP_PL_JOB}  (parallel)"
echo "  pick + prep_KY -> train_KY ${TRAIN_KY_JOB}"
echo "  pick + prep_KZ -> train_KZ ${TRAIN_KZ_JOB}"
echo "  pick + prep_PL -> train_PL ${TRAIN_PL_JOB}"
