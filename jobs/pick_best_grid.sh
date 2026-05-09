#!/bin/bash -l
# Collector job: picks the best grid run by lowest final training loss.
# Submitted with --dependency=afterany:ARRAY_ID so one crashed run won't block.
#
# Arguments:
#   $1  MODEL_SHORT   Short model name (e.g. Llama-3.1-8B) — used to glob checkpoints
#   $2  GRID_JOB_ID   Optional current SLURM array job ID, scopes result search

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:0
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=logs/grid-winner-%j.log
#SBATCH --error=logs/grid-winner-%j.err

ml ML-bundle/25.10

set -euo pipefail

MODEL_SHORT=${1:?"MODEL_SHORT required (e.g. Llama-3.1-8B)"}
GRID_JOB_ID=${2:-}

SCRATCH_ROOT=${SCRATCH}/cpt_experiments
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv

mkdir -p logs

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

WINNER_FILE="${REPO_DIR}/logs/grid_winner_${MODEL_SHORT}.txt"

echo "Picking best grid run for model: ${MODEL_SHORT}"
if [ -n "${GRID_JOB_ID}" ]; then
    echo "Grid job ID: ${GRID_JOB_ID}"
    echo "Searching: ${SCRATCH_ROOT}/checkpoints/grid_${MODEL_SHORT}_*_${GRID_JOB_ID}/"
else
    echo "WARNING: GRID_JOB_ID not provided; searching all historical grid runs for this model."
    echo "Searching: ${SCRATCH_ROOT}/checkpoints/grid_${MODEL_SHORT}_*/"
fi
echo ""

python3 - <<EOF
import json, glob, sys, os, math

grid_job_id = "${GRID_JOB_ID}"
if grid_job_id:
    pattern = "${SCRATCH_ROOT}/checkpoints/grid_${MODEL_SHORT}_*_${GRID_JOB_ID}/grid_search_result.json"
else:
    pattern = "${SCRATCH_ROOT}/checkpoints/grid_${MODEL_SHORT}_*/grid_search_result.json"
files = glob.glob(pattern)

if not files:
    print(f"ERROR: No grid_search_result.json found matching: {pattern}")
    sys.exit(1)

results = []
for f in files:
    try:
        data = json.load(open(f))
        data["checkpoint_dir"] = os.path.dirname(f)
        results.append(data)
    except Exception as e:
        print(f"Warning: could not read {f}: {e}")

if not results:
    print("ERROR: All result files were unreadable.")
    sys.exit(1)

valid = [r for r in results if r.get("final_train_loss") is not None]
if not valid:
    print("ERROR: no runs have final_train_loss.")
    sys.exit(1)
best = min(valid, key=lambda x: x["final_train_loss"])

print("All results:")
for r in sorted(valid, key=lambda x: x.get("final_train_loss") or float("inf")):
    loss = r.get("final_train_loss")
    loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) and math.isfinite(loss) else "N/A"
    print(f"  Run {r.get('run_label','?')}: r={r.get('lora_r')} lr={r.get('learning_rate')} "
          f"final_train_loss={loss_text}")

print()
print("WINNER:")
print(json.dumps(best, indent=2))

with open("${WINNER_FILE}", "w") as f:
    json.dump(best, f, indent=2)
EOF

echo ""
echo "Winner written to: ${WINNER_FILE}"
echo ""
echo "Next step: update configs/<model>_cpt.yaml with winning lora_r and learning_rate,"
echo "then submit the full matrix:"
echo "  bash submit_cpt_matrix.sh words"
