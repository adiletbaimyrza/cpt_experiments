#!/bin/bash -l
# Main 4-GPU CPT training job for one model/variant/dataset.
#
# Arguments:
#   $1  MODEL        HF model ID
#   $2  DATASET      dataset name under data/cpt_processed/
#   $3  LANG_VARIANT FT-KY | FT-KZ | FT-PL
#   $4  LORA_R       LoRA rank
#   $5  LR           learning rate
#   $6  RUN_ID       stable run identifier

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=cpt/logs/train-cpt-%j.log
#SBATCH --error=cpt/logs/train-cpt-%j.err

set -euo pipefail

MODEL=${1:?"MODEL required"}
DATASET=${2:?"DATASET required"}
LANG_VARIANT=${3:?"LANG_VARIANT required"}
LORA_R=${4:?"LORA_R required"}
LR=${5:?"LR required"}
RUN_ID=${6:-$(date +%Y%m%d%H%M%S)}

LORA_ALPHA=$((LORA_R * 2))
MODEL_SHORT="${MODEL##*/}"
SCRATCH_ROOT=${SCRATCH}/kyrgyzLLM
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv
HF_HOME=${SCRATCH_ROOT}/cache
OUTPUT_DIR="${SCRATCH_ROOT}/checkpoints/cpt_${MODEL_SHORT}_${LANG_VARIANT}_${DATASET}_${RUN_ID}"

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
    echo "ERROR: HF_TOKEN is not set. Add it to ${ENV_FILE}."
    exit 1
fi
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

mkdir -p cpt/logs

if [ ! -d "${REPO_DIR}/data/cpt_processed/${DATASET}" ]; then
    echo "ERROR: Dataset not found at data/cpt_processed/${DATASET}"
    exit 1
fi

ACCELERATE_CONFIG="${REPO_DIR}/helios/accelerate_config.yaml"
if [ ! -f "${ACCELERATE_CONFIG}" ]; then
    echo "ERROR: ${ACCELERATE_CONFIG} not found"
    exit 1
fi

unset CUDA_VISIBLE_DEVICES

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting CPT training..."
echo ""

accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes 4 \
    --mixed_precision bf16 \
    cpt/scripts/train_cpt.py \
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
    --gradient_accumulation_steps 4 \
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
