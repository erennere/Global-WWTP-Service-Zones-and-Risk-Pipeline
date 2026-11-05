#!/bin/bash
# Sweep verification ranking across the create_voronoi_param_sweep_parallel grid.
# Splits each pop-output into ver/unver/single subsets and scores against HW and EU.
#
# WORKER MODE (SLURM array, 10 tasks):
#   sbatch --array=0-9 sensitivity_analysis_scripts/sweep_ver_ranking.sh [--shuffle-seed <n>]
#   Each task processes ~1/10 of the sweep grid.
#
# MERGE MODE (run after all 10 worker tasks complete):
#   bash sensitivity_analysis_scripts/sweep_ver_ranking.sh --merge
#
#SBATCH --partition=cpu-single
#SBATCH --time=96:00:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9
#SBATCH --job-name=sweep-ver-ranking
#SBATCH --output=logs/sweep_ver_ranking_%A_%a.out
#SBATCH --error=logs/sweep_ver_ranking_%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"

PYTHON_MODULE="src.sensitivity_analysis_scripts.sweep_ver_ranking"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
MAX_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

MERGE_MODE="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --merge)
            MERGE_MODE="true"
            shift
            ;;
        --shuffle-seed)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --shuffle-seed requires a value" >&2
                echo "Usage: $0 [--merge] [--shuffle-seed <n>]" >&2
                exit 1
            fi
            SHUFFLE_SEED="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument '$1'" >&2
            echo "Usage: $0 [--merge] [--shuffle-seed <n>]" >&2
            exit 1
            ;;
    esac
done

if [[ "${MERGE_MODE}" == "true" ]]; then
    init_log "sweep_ver_ranking_merge"
    enable_err_trap
    log "Running sweep_ver_ranking merge pass"
    run_stage "${PYTHON_MODULE}" ${PYTHON_CMD} -m "${PYTHON_MODULE}" --merge
    log "Merge complete"
else
    TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    init_log "sweep_ver_ranking_${TASK_ID}"
    enable_err_trap
    rm -f "${LOG_DIR}"/sweep_ver_ranking_"${TASK_ID}"_*.out \
          "${LOG_DIR}"/sweep_ver_ranking_"${TASK_ID}"_*.err

    if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 9 )); then
        log "ERROR: TASK_ID must be between 0 and 9 (got '${TASK_ID}')"
        exit 1
    fi

    log "Worker task ${TASK_ID}/9 (seed=${SHUFFLE_SEED}, workers=${MAX_WORKERS})"
    run_stage "${PYTHON_MODULE}" ${PYTHON_CMD} -m "${PYTHON_MODULE}" \
        --task-id "${TASK_ID}" --shuffle-seed "${SHUFFLE_SEED}"
    log "Task ${TASK_ID} complete"
fi
