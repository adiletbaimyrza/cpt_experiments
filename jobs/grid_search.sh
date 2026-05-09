#!/bin/bash -l
# Grid search: ONE predefined run (A/B/C/D) on FT-KY for one model.
# Submitted 4× per model from submit_cpt_pipeline.sh, one job per label.
#
# Run table:
#   Label | LR    | Rank
#   A     | 1e-4  | 64
#   B     | 1e-4  | 128
#   C     | 1e-4  | 256
#   D     | 2e-4  | 128
#
# Arguments:
#   $1  MODEL            HF model ID (e.g. meta-llama/Llama-3.1-8B)
#   $2  FT_KY_DATASET    dataset name under data/cpt_processed/
#   $3  GRID_MAX_STEPS   steps for this run
#   $4  RUN_LABEL        A | B | C | D

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err

ml ML-bundle/25.10

set -euo pipefail

MODEL=${1:?"MODEL required"}
FT_KY_DATASET=${2:?"FT_KY_DATASET required"}
GRID_MAX_STEPS=${3:-500}
RUN_LABEL=${4:?"RUN_LABEL required (A|B|C|D)"}
GRID_PHASE1_STEPS=$((GRID_MAX_STEPS / 10))
if [ "${GRID_PHASE1_STEPS}" -lt 1 ]; then GRID_PHASE1_STEPS=1; fi

case "${RUN_LABEL}" in
    A) LR=1e-4; LORA_R=64  ;;
    B) LR=1e-4; LORA_R=128 ;;
    C) LR=1e-4; LORA_R=256 ;;
    D) LR=2e-4; LORA_R=128 ;;
    *) echo "ERROR: RUN_LABEL must be A, B, C, or D"; exit 1 ;;
esac
LORA_ALPHA=$((LORA_R * 2))
MODEL_SHORT="${MODEL##*/}"

SCRATCH_ROOT=${SCRATCH}/cpt_experiments
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv
HF_HOME=${SCRATCH}/hf_home
OUTPUT_DIR="${SCRATCH_ROOT}/checkpoints/grid_${MODEL_SHORT}_${RUN_LABEL}_r${LORA_R}_lr${LR}_${SLURM_JOB_ID}"

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
echo "Job:        ${SLURM_JOB_ID}"
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
