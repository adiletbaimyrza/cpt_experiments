#!/bin/bash -l
# Collector job: picks the best grid run by lowest final eval loss
# (falls back to final training loss when eval is unavailable, e.g. on smoke runs).
# Submitted with --dependency=afterany:<grid_job_ids> so one crashed run won't block.
#
# Arguments:
#   $1  MODEL_SHORT   Short model name (e.g. Llama-3.1-8B) — used to glob checkpoints
#   $2  GRID_JOB_ID   Optional colon-separated list of grid job IDs; scopes result search

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:0
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err

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
    echo "Grid job IDs: ${GRID_JOB_ID}"
    echo "Searching: ${SCRATCH_ROOT}/checkpoints/grid_${MODEL_SHORT}_*_<job_id>/"
else
    echo "WARNING: GRID_JOB_ID not provided; searching all historical grid runs for this model."
    echo "Searching: ${SCRATCH_ROOT}/checkpoints/grid_${MODEL_SHORT}_*/"
fi
echo ""

python3 - <<EOF
import json, glob, sys, os, math

grid_job_id_arg = "${GRID_JOB_ID}"
files = []
if grid_job_id_arg:
    # GRID_JOB_ID can be a single ID or colon-separated list of IDs (one per A/B/C/D run)
    for jid in grid_job_id_arg.split(":"):
        jid = jid.strip()
        if not jid:
            continue
        pattern = "${SCRATCH_ROOT}/checkpoints/grid_${MODEL_SHORT}_*_" + jid + "/grid_search_result.json"
        files.extend(glob.glob(pattern))
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

# Prefer eval loss (generalization). Fall back to train loss for short smoke runs
# where the eval interval never fires.
def _score(r):
    ev = r.get("final_eval_loss")
    if ev is not None:
        return ev
    return r.get("final_train_loss", float("inf"))

valid = [r for r in results if (r.get("final_eval_loss") is not None
                                 or r.get("final_train_loss") is not None)]
if not valid:
    print("ERROR: no runs have final_eval_loss or final_train_loss.")
    sys.exit(1)

# Note in output which metric drove selection
any_eval = any(r.get("final_eval_loss") is not None for r in valid)
metric_used = "final_eval_loss (preferred)" if any_eval else "final_train_loss (eval unavailable)"
best = min(valid, key=_score)

print(f"Selection metric: {metric_used}")
print("All results:")
def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) and math.isfinite(v) else "N/A"
for r in sorted(valid, key=_score):
    train_l = r.get("final_train_loss")
    eval_l = r.get("final_eval_loss")
    print(f"  Run {r.get('run_label','?')}: r={r.get('lora_r')} lr={r.get('learning_rate')} "
          f"train_loss={_fmt(train_l)} eval_loss={_fmt(eval_l)}")

print()
print("WINNER:")
print(json.dumps(best, indent=2))

with open("${WINNER_FILE}", "w") as f:
    json.dump(best, f, indent=2)
EOF

echo ""
echo "Winner written to: ${WINNER_FILE}"
echo "(Read by jobs/train_cpt.sh at runtime — no manual config patch needed.)"
