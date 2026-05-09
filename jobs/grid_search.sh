#!/bin/bash -l
# Grid search: 4 predefined runs (A/B/C/D) on FT-KY only per model.
# Submits as SLURM array --array=0-3.
#
# Run table:
#   Index | Label | LR    | Rank
#   0     | A     | 1e-4  | 64
#   1     | B     | 1e-4  | 128
#   2     | C     | 1e-4  | 256
#   3     | D     | 2e-4  | 128
#
# Arguments:
#   $1  MODEL            HF model ID (e.g. meta-llama/Llama-3.1-8B)
#   $2  FT_KY_DATASET    dataset name under data/cpt_processed/
#   $3  GRID_MAX_STEPS   steps per run (default: 500)

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --array=0-3
#SBATCH --output=logs/grid-search-%A-%a.log
#SBATCH --error=logs/grid-search-%A-%a.err

set -euo pipefail

MODEL=${1:?"MODEL required"}
FT_KY_DATASET=${2:?"FT_KY_DATASET required"}
GRID_MAX_STEPS=${3:-500}
GRID_PHASE1_STEPS=$((GRID_MAX_STEPS / 10))
if [ "${GRID_PHASE1_STEPS}" -lt 1 ]; then GRID_PHASE1_STEPS=1; fi

# Predefined run table
LR_VALUES=(1e-4 1e-4 1e-4 2e-4)
RANK_VALUES=(64 128 256 128)
RUN_LABELS=(A B C D)

LR=${LR_VALUES[$SLURM_ARRAY_TASK_ID]}
LORA_R=${RANK_VALUES[$SLURM_ARRAY_TASK_ID]}
LORA_ALPHA=$((LORA_R * 2))
RUN_LABEL=${RUN_LABELS[$SLURM_ARRAY_TASK_ID]}
MODEL_SHORT="${MODEL##*/}"

SCRATCH_ROOT=${SCRATCH}/cpt_experiments
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv
HF_HOME=${SCRATCH_ROOT}/cache
OUTPUT_DIR="${SCRATCH_ROOT}/checkpoints/grid_${MODEL_SHORT}_r${LORA_R}_lr${LR}_${SLURM_ARRAY_JOB_ID}"

echo "=========================================="
echo "CPT Grid Search: Run ${RUN_LABEL}"
echo "=========================================="
echo "Model:      ${MODEL}"
echo "LR:         ${LR}"
echo "Rank:       ${LORA_R}"
echo "Alpha:      ${LORA_ALPHA}"
echo "Steps:      ${GRID_MAX_STEPS}"
echo "Save steps: ${GRID_PHASE1_STEPS}"
echo "Dataset:    ${FT_KY_DATASET}"
echo "Output:     ${OUTPUT_DIR}"
echo "Array task: ${SLURM_ARRAY_TASK_ID} / job ${SLURM_ARRAY_JOB_ID}"
echo "=========================================="
echo ""

ml ML-bundle/24.06a

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

export HF_HOME
export TRANSFORMERS_CACHE=${HF_HOME}

ENV_FILE="${SCRATCH_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a; source "${ENV_FILE}"; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    exit 1
fi
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

mkdir -p logs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting grid run ${RUN_LABEL}..."
echo ""

python scripts/train_cpt.py \
    --model "${MODEL}" \
    --data_path "${REPO_DIR}/data/cpt_processed/${FT_KY_DATASET}" \
    --lang_variant "FT-KY" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --learning_rate "${LR}" \
    --use_rslora \
    --lora_dropout 0.05 \
    --max_steps "${GRID_MAX_STEPS}" \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --bf16 \
    --gradient_checkpointing \
    --save_steps "${GRID_PHASE1_STEPS}" \
    --save_total_limit 2 \
    --run_label "${RUN_LABEL}" \
    --run_name "${MODEL_SHORT}_grid_${RUN_LABEL}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Grid run ${RUN_LABEL} complete."
echo "Result: ${OUTPUT_DIR}/grid_search_result.json"
echo "Metrics JSONL: ${OUTPUT_DIR}/metrics/train_metrics.jsonl"
