#!/bin/bash -l
# Reads the grid winner for one model, patches its config YAML, then
# submits data-prep + training chains for all three language variants.
#
# Submitted by submit_cpt_pipeline.sh with --dependency=afterok:PICK_JOB_ID.
# sbatch called from a compute node — works on PLGrid SLURM.
#
# Arguments:
#   $1  MODEL
#   $2  CONFIG_FILE         path relative to REPO_DIR (e.g. configs/llama_cpt.yaml)
#   $3  EXPERIMENT          words | tokens
#   $4  ENGLISH_DATASET_ID
#   $5  DATASET_FT_KY
#   $6  DATASET_FT_KZ
#   $7  DATASET_FT_PL
#   $8  RUN_ID              (default: resume)

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:0
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=logs/apply-winner-%j.log
#SBATCH --error=logs/apply-winner-%j.err

ml ML-bundle/24.06a

set -euo pipefail

MODEL=${1:?"MODEL required"}
CONFIG_FILE=${2:?"CONFIG_FILE required"}
EXPERIMENT=${3:?"EXPERIMENT required"}
ENGLISH_DATASET_ID=${4:?"ENGLISH_DATASET_ID required"}
DATASET_FT_KY=${5:?"DATASET_FT_KY required"}
DATASET_FT_KZ=${6:?"DATASET_FT_KZ required"}
DATASET_FT_PL=${7:?"DATASET_FT_PL required"}
RUN_ID=${8:-${CPT_RUN_ID:-resume}}

SCRATCH_ROOT=${SCRATCH}/cpt_experiments
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv

MODEL_SHORT="${MODEL##*/}"
WINNER_FILE="${REPO_DIR}/logs/grid_winner_${MODEL_SHORT}.txt"
BUDGET=100000000

_LOG="${CPT_LOG_DIR:-${REPO_DIR}/logs}"
mkdir -p "${_LOG}"

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

echo "=========================================="
echo "Apply Winner and Submit Training: ${MODEL_SHORT}"
echo "=========================================="
echo "Winner file: ${WINNER_FILE}"
echo "Config:      ${CONFIG_FILE}"
echo "Experiment:  ${EXPERIMENT}"
echo "=========================================="
echo ""

if [ ! -f "${WINNER_FILE}" ]; then
    echo "ERROR: Winner file not found: ${WINNER_FILE}"
    exit 1
fi

# Extract winner hyperparams
LORA_R=$(python3 -c "import json; d=json.load(open('${WINNER_FILE}')); print(d['lora_r'])")
LR=$(python3 -c "import json; d=json.load(open('${WINNER_FILE}')); print(d['learning_rate'])")

echo "Winner: lora_r=${LORA_R}, learning_rate=${LR}"
echo ""

# Patch config YAML: update lora.r, lora.alpha (=r*2), training.learning_rate
python3 - "${REPO_DIR}/${CONFIG_FILE}" "${LORA_R}" "${LR}" <<'PY'
import sys, yaml
config_path, lora_r, lr = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
with open(config_path) as f:
    data = yaml.safe_load(f)
data['lora']['r'] = lora_r
data['lora']['alpha'] = lora_r * 2
data['training']['learning_rate'] = lr
with open(config_path, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f"Patched {config_path}: lora.r={lora_r}, lora.alpha={lora_r*2}, lr={lr}")
PY

echo ""
echo "Submitting prep + training for all 3 variants..."
echo ""

declare -A DATASETS=(
    [FT-KY]="${DATASET_FT_KY}"
    [FT-KZ]="${DATASET_FT_KZ}"
    [FT-PL]="${DATASET_FT_PL}"
)

for VARIANT in FT-KY FT-KZ FT-PL; do
    DATASET_ID="${DATASETS[$VARIANT]}"
    DATASET_ID_SAFE=$(printf '%s' "${DATASET_ID}" | tr '/:' '__')
    DATASET_SAFE=$(printf '%s' "${VARIANT}_${EXPERIMENT}_${MODEL_SHORT}_${DATASET_ID_SAFE}" | tr '/:' '__')

    PREP_JOB_ID=$(sbatch \
        --parsable \
        --output="${_LOG}/prepare-cpt-%j.log" \
        --error="${_LOG}/prepare-cpt-%j.err" \
        jobs/prepare_cpt_data.sh \
        "${DATASET_ID}" "${MODEL}" "${VARIANT}" "${EXPERIMENT}" "${BUDGET}" "${DATASET_SAFE}" "${ENGLISH_DATASET_ID}")

    TRAIN_JOB_ID=$(sbatch \
        --parsable \
        --dependency=afterok:${PREP_JOB_ID} \
        --output="${_LOG}/train-cpt-%j.log" \
        --error="${_LOG}/train-cpt-%j.err" \
        jobs/train_cpt.sh \
        "${MODEL}" "${DATASET_SAFE}" "${VARIANT}" "${LORA_R}" "${LR}" "${RUN_ID}")

    echo "${VARIANT}: prep ${PREP_JOB_ID} -> train ${TRAIN_JOB_ID}"
done

echo ""
echo "All training chains submitted for ${MODEL_SHORT}."
