#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=64
#SBATCH --mem=234gb
#SBATCH --time=48:00:00
#SBATCH --array=0-9
#SBATCH --job-name=bing-annotate
#SBATCH --output=logs/bing_annotate_%A_%a.out
#SBATCH --error=logs/bing_annotate_%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"

PYTHON_SCRIPT="src.annotation_scripts.download_bing_annotate"

NUM_INSTANCES=10
SPLIT_SEED=42
INSTANCE_ID="${SLURM_ARRAY_TASK_ID}"

parse_overrides "$@"
build_override_args

init_log "bing_annotate_${INSTANCE_ID}"
enable_err_trap
rm -f "${LOG_DIR}"/bing_annotate_*_*.out "${LOG_DIR}"/bing_annotate_*_*.err

ensure_src_importable

log "Running download_bing_annotate instance $INSTANCE_ID of $NUM_INSTANCES"
run_stage "${PYTHON_SCRIPT}" ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" "$INSTANCE_ID" --num-instances "$NUM_INSTANCES" --split-seed "$SPLIT_SEED" "${OVERRIDE_ARGS[@]}"
log "Completed download_bing_annotate instance $INSTANCE_ID"
