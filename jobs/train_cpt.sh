#!/bin/bash -l
# Single-GPU CPT training job for one model/variant/dataset.
# Reads winning lora_r/lr from logs/grid_winner_${MODEL_SHORT}.txt at runtime.
#
# Arguments:
#   $1  MODEL        HF model ID
#   $2  DATASET      dataset name under data/cpt_processed/
#   $3  LANG_VARIANT FT-KY | FT-KZ | FT-PL
#   $4  RUN_ID       stable run identifier

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=logs/train-cpt-%j.log
#SBATCH --error=logs/train-cpt-%j.err

ml ML-bundle/25.10

set -euo pipefail

MODEL=${1:?"MODEL required"}
DATASET=${2:?"DATASET required"}
LANG_VARIANT=${3:?"LANG_VARIANT required"}
RUN_ID=${4:-$(date +%Y%m%d%H%M%S)}

MODEL_SHORT="${MODEL##*/}"
SCRATCH_ROOT=${SCRATCH}/cpt_experiments
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv
HF_HOME=${SCRATCH}/hf_home
OUTPUT_DIR="${SCRATCH_ROOT}/checkpoints/cpt_${MODEL_SHORT}_${LANG_VARIANT}_${DATASET}_${RUN_ID}"

WINNER_FILE="${REPO_DIR}/logs/grid_winner_${MODEL_SHORT}.txt"
if [ ! -f "${WINNER_FILE}" ]; then
    echo "ERROR: Winner file not found: ${WINNER_FILE}"
    echo "       Did pick_best_grid.sh run successfully?"
    exit 1
fi
LORA_R=$(python3 -c "import json; print(json.load(open('${WINNER_FILE}'))['lora_r'])")
LR=$(python3 -c "import json; print(json.load(open('${WINNER_FILE}'))['learning_rate'])")
LORA_ALPHA=$((LORA_R * 2))

echo "=========================================="
echo "CPT Training"
echo "=========================================="
echo "Model:        ${MODEL}"
echo "Dataset:      data/cpt_processed/${DATASET}"
echo "Lang variant: ${LANG_VARIANT}"
echo "LoRA rank:    ${LORA_R}"
echo "LoRA alpha:   ${LORA_ALPHA}"
echo "LR:           ${LR}"
echo "Epochs:       3 (max_steps auto-computed from dataset size)"
echo "Run ID:       ${RUN_ID}"
echo "Output:       ${OUTPUT_DIR}"
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

if [ ! -d "${REPO_DIR}/data/cpt_processed/${DATASET}" ]; then
    echo "ERROR: Dataset not found at data/cpt_processed/${DATASET}"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting CPT training..."
echo ""

python scripts/train_cpt.py \
    --model "${MODEL}" \
    --data_path "${REPO_DIR}/data/cpt_processed/${DATASET}" \
    --lang_variant "${LANG_VARIANT}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --learning_rate "${LR}" \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --use_rslora \
    --lora_dropout 0.05 \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --bf16 \
    --gradient_checkpointing \
    --save_steps 250 \
    --save_total_limit 4 \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${MODEL_SHORT}_${LANG_VARIANT}_${RUN_ID}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] CPT training complete."
echo "Adapter: ${OUTPUT_DIR}/final"
echo "Metrics JSONL: ${OUTPUT_DIR}/metrics/train_metrics.jsonl"
echo "Metrics CSV:   ${OUTPUT_DIR}/metrics/trainer_log_history.csv"
