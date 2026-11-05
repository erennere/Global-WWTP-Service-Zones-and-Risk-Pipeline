#!/bin/bash
# Parallel parameter sweep for create_voronoi (additive-only)
#SBATCH --partition=cpu-single
#SBATCH --time=96:00:00
#SBATCH --mem=234gb
#SBATCH --cpus-per-task=64
#SBATCH --array=0-9
#SBATCH --job-name=voronoi-additive-sweep
#SBATCH --output=logs/voronoi_additive_sweep_%A_%a.out
#SBATCH --error=logs/voronoi_additive_sweep_%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"

PYTHON_SCRIPT="src.sensitivity_analysis_scripts.create_voronoi_parallel_sweep"
APPROACH="1"
VERSION=""
DYNAMIC_BUFFERING=""
DYNAMIC_BUFFER_K=""
NUM_JOBS=16

while [[ $# -gt 0 ]]; do
    case "$1" in
        --approach)
            APPROACH="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --dynamic-buffering)
            DYNAMIC_BUFFERING="$2"
            shift 2
            ;;
        --dynamic-buffer-k)
            DYNAMIC_BUFFER_K="$2"
            shift 2
            ;;
        --num-jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument '$1'" >&2
            echo "Usage: $0 [--approach <n>] [--version <v>] [--dynamic-buffering <true|false>] [--dynamic-buffer-k <k>] [--num-jobs <n>]" >&2
            exit 1
            ;;
    esac
done

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-${NUM_JOBS}}"
MEM_PER_NODE="${SLURM_MEM_PER_NODE:-unknown}"
init_log "voronoi_additive_sweep_${TASK_ID}"
enable_err_trap
rm -f "${LOG_DIR}"/voronoi_additive_sweep_*.out \
      "${LOG_DIR}"/voronoi_additive_sweep_*.err

if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    log "ERROR: Invalid task id '${TASK_ID}'"
    exit 1
fi
if (( TASK_ID < 0 || TASK_ID > 9 )); then
    log "ERROR: TASK_ID must be between 0 and 9 (got ${TASK_ID})"
    exit 1
fi

log "Starting additive-only Voronoi parallel parameter sweep task ${TASK_ID}/9"
log "Configuration:"
log "  - Parallel jobs: ${NUM_JOBS}"
log "  - CPUs per job: $((CPUS_PER_TASK / NUM_JOBS))"
log "  - Total CPUs: ${CPUS_PER_TASK}"
log "  - Total memory: ${MEM_PER_NODE}MB"
log "  - Approach: ${APPROACH}"
log "  - Version: ${VERSION:-default}"
log "  - Dynamic buffering: ${DYNAMIC_BUFFERING:-default}"
log "  - Dynamic buffer k: ${DYNAMIC_BUFFER_K:-default}"

log "Running additive-only parallel sweep with ${NUM_JOBS} concurrent jobs"
${PYTHON_CMD} -m "${PYTHON_SCRIPT}" \
    --task-id "${TASK_ID}" \
    --version "${VERSION}" \
    --dynamic-buffering "${DYNAMIC_BUFFERING}" \
    --dynamic-buffer-k "${DYNAMIC_BUFFER_K}" \
    --approach "${APPROACH}" \
    --num-jobs "${NUM_JOBS}" \
    --weight-func-filter add \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    log "Task ${TASK_ID}: Completed successfully"
else
    log "Task ${TASK_ID}: FAILED with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi
