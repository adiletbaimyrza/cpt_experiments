#!/bin/bash
# Wipe state for a clean re-run on Helios.
# Removes: venv, HF cache, processed data, logs, checkpoints, grid winners.
# Preserves: $SCRATCH/cpt_experiments/.env (HF_TOKEN, dataset IDs).
#
# Usage:
#   bash clean.sh               # interactive — asks before deleting
#   bash clean.sh --yes         # skip confirmation
#   bash clean.sh --keep-cache  # preserve HF cache (saves model redownload)
#
# Always cancels your queued/running SLURM jobs first.

set -euo pipefail

ASSUME_YES=0
KEEP_CACHE=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=1 ;;
        --keep-cache) KEEP_CACHE=1 ;;
        *) echo "ERROR: unknown flag: $arg"; exit 1 ;;
    esac
done

SCRATCH_ROOT="${SCRATCH}/cpt_experiments"
HF_HOME_DIR="${SCRATCH}/hf_home"

if [ ! -d "${SCRATCH_ROOT}" ]; then
    echo "Nothing to clean: ${SCRATCH_ROOT} does not exist."
    exit 0
fi

# Build the target list
declare -a TARGETS=(
    "${SCRATCH_ROOT}/venv"
    "${SCRATCH_ROOT}/data/cpt_processed"
    "${SCRATCH_ROOT}/checkpoints"
    "${SCRATCH_ROOT}/logs"
)
if [ "${KEEP_CACHE}" -eq 0 ]; then
    TARGETS+=("${HF_HOME_DIR}")
fi

echo "=========================================="
echo "Cleanup plan"
echo "=========================================="
echo "Will cancel all your SLURM jobs and remove:"
for t in "${TARGETS[@]}"; do
    if [ -e "$t" ]; then
        size=$(du -sh "$t" 2>/dev/null | cut -f1)
        printf "  %-12s %s\n" "${size:-?}" "$t"
    else
        printf "  %-12s %s (not present)\n" "—" "$t"
    fi
done
echo ""
echo "Will preserve:"
echo "  ${SCRATCH_ROOT}/.env"
if [ "${KEEP_CACHE}" -eq 1 ]; then
    echo "  ${HF_HOME_DIR} (--keep-cache)"
fi
echo "=========================================="

if [ "${ASSUME_YES}" -ne 1 ]; then
    read -r -p "Proceed? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

# 1. Cancel SLURM jobs first (idempotent if none queued)
echo ""
echo "[1/2] Cancelling SLURM jobs for $(whoami)..."
scancel -u "$(whoami)" 2>/dev/null || true
sleep 1
remaining=$(squeue -u "$(whoami)" -h 2>/dev/null | wc -l | tr -d ' ')
echo "  Remaining in queue: ${remaining}"

# 2. Remove targets
echo ""
echo "[2/2] Removing files..."
for t in "${TARGETS[@]}"; do
    if [ -e "$t" ]; then
        rm -rf "$t"
        echo "  removed: $t"
    fi
done

echo ""
echo "Done. Re-run with: bash setup_and_submit.sh words"
