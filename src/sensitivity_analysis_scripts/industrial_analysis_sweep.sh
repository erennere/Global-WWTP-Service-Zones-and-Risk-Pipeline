#!/bin/bash
# Parameter sweep for industrial analysis across multiple configurations
#SBATCH --partition=cpu-single
#SBATCH --time=96:00:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9
#SBATCH --job-name=industrial-analysis-sweep
#SBATCH --output=logs/industrial_analysis_sweep_%A_%a.out
#SBATCH --error=logs/industrial_analysis_sweep_%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"

DOWNLOAD_SCRIPT="src.industrial_analysis.download_and_vectorize"
FIND_UNCONNECTED_SCRIPT="src.industrial_analysis.find_unconnected_industrial_areas"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
VERSION=""

# The sweep varies level/buffer/weight/dynamic-k itself, but version is not a
# swept dimension - it has to be passed through or every run lands in whatever
# version config happens to default to.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --version requires a value" >&2
                exit 1
            fi
            VERSION="$2"
            shift 2
            ;;
        --shuffle-seed)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --shuffle-seed requires a value" >&2
                exit 1
            fi
            SHUFFLE_SEED="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument '$1'" >&2
            echo "Usage: $0 [--version <v>] [--shuffle-seed <n>]" >&2
            exit 1
            ;;
    esac
done

VERSION_ARGS=()
if [[ -n "${VERSION}" ]]; then
    VERSION_ARGS=("--version" "${VERSION}")
fi

# Parameter grids for industrial analysis sweep
LEVELS=(6 7 8 9)
WEIGHT_FUNCS=("mult" "add" "")
WEIGHT_METHODS=("linear" "logarithmic" "square_root" "sigmoid")
BUFFERS=(9000 11000 13000 15000)
DYNAMIC_K_VALUES=(0.6 0.7 0.8)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
init_log "industrial_sweep_${TASK_ID}"
enable_err_trap
rm -f "${LOG_DIR}"/industrial_sweep_*.out "${LOG_DIR}"/industrial_sweep_*.err

if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    log "ERROR: Invalid task id '${TASK_ID}'"
    exit 1
fi
if (( TASK_ID < 0 || TASK_ID > 9 )); then
    log "ERROR: TASK_ID must be between 0 and 9 (got ${TASK_ID})"
    exit 1
fi

log "Starting industrial analysis sweep task ${TASK_ID}/9"

ensure_src_importable

run_count=0
mapfile -t ASSIGNED_COMBOS < <(
    "${PYTHON_CMD}" -c "
import sys
from src.sensitivity_analysis_scripts.create_voronoi_parallel_sweep import print_task_combinations
print_task_combinations(int(sys.argv[1]), int(sys.argv[2]))
" "${TASK_ID}" "${SHUFFLE_SEED}"
)

for combo in "${ASSIGNED_COMBOS[@]}"; do
    IFS=$'\t' read -r level buffer weight_method weight_func dynamic_buffering dynamic_buffer_k <<< "${combo}"
    level=$(echo "${level}" | xargs)
    buffer=$(echo "${buffer}" | xargs)
    weight_method=$(echo "${weight_method}" | xargs)
    weight_func=$(echo "${weight_func}" | xargs)
    dynamic_buffering=$(echo "${dynamic_buffering}" | xargs)
    dynamic_buffer_k=$(echo "${dynamic_buffer_k}" | xargs)

    if [[ "${weight_func}" == "__EMPTY__" ]]; then
        weight_func=""
    fi
    if [[ "${dynamic_buffer_k}" == "__EMPTY__" ]]; then
        dynamic_buffer_k=""
    fi

    run_count=$((run_count + 1))
    log "Run ${run_count}: level=${level} buffer=${buffer} weight_method=${weight_method} weight_func='${weight_func}' dynamic_buffering=${dynamic_buffering} dynamic_buffer_k=${dynamic_buffer_k}"

    # Download and vectorize
    run_stage "${DOWNLOAD_SCRIPT}" ${PYTHON_CMD} -m "${DOWNLOAD_SCRIPT}" "${VERSION_ARGS[@]}" --level "${level}" --buffer "${buffer}" --weight-method "${weight_method}" --weight-func "${weight_func}" --dynamic-buffering "${dynamic_buffering}" --dynamic-buffer-k "${dynamic_buffer_k}"

    # Find unconnected areas
    run_stage "${FIND_UNCONNECTED_SCRIPT}" ${PYTHON_CMD} -m "${FIND_UNCONNECTED_SCRIPT}" "${VERSION_ARGS[@]}" --level "${level}" --buffer "${buffer}" --weight-method "${weight_method}" --weight-func "${weight_func}" --dynamic-buffering "${dynamic_buffering}" --dynamic-buffer-k "${dynamic_buffer_k}"
done

log "Completed task ${TASK_ID}. Executed ${run_count} parameter combinations."
