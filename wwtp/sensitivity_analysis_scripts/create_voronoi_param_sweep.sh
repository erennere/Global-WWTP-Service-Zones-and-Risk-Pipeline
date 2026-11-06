#!/bin/bash
# Lightweight parameter sweep for create_voronoi (approach fixed to 1)
#SBATCH --partition=cpu-single
#SBATCH --time=96:00:00
#SBATCH --mem=192gb
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9
#SBATCH --job-name=voronoi-sweep
#SBATCH --output=logs/voronoi_sweep_%A_%a.out
#SBATCH --error=logs/voronoi_sweep_%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"

PYTHON_SCRIPT="wwtp.create_voronoi"
APPROACH="1"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

parse_overrides "$@"

# The parameter grid lives in config.yaml (create_voronoi_parallel_sweep.sweep_*).

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
init_log "voronoi_sweep_${TASK_ID}"
enable_err_trap
rm -f "${LOG_DIR}"/voronoi_sweep_*.out "${LOG_DIR}"/voronoi_sweep_*.err

if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    log "ERROR: Invalid task id '${TASK_ID}'"
    exit 1
fi
if (( TASK_ID < 0 || TASK_ID > 9 )); then
    log "ERROR: TASK_ID must be between 0 and 9 (got ${TASK_ID})"
    exit 1
fi

log "Starting Voronoi parameter sweep task ${TASK_ID}/9 (approach=${APPROACH}, shuffle_seed=${SHUFFLE_SEED})"

ensure_package_importable

run_count=0
mapfile -t ASSIGNED_COMBOS < <(
    "${PYTHON_CMD}" -c "
import sys
from wwtp.sensitivity_analysis_scripts.create_voronoi_parallel_sweep import print_task_combinations
print_task_combinations(int(sys.argv[1]), int(sys.argv[2]))
" "${TASK_ID}" "${SHUFFLE_SEED}"
)

for combo in "${ASSIGNED_COMBOS[@]}"; do
    IFS=$'\t' read -r level buffer weight_method weight_func dynamic_buffering size_measure <<< "${combo}"
    level="${level//$'\r'/}"
    buffer="${buffer//$'\r'/}"
    weight_method="${weight_method//$'\r'/}"
    weight_func="${weight_func//$'\r'/}"
    dynamic_buffering="${dynamic_buffering//$'\r'/}"
    size_measure="${size_measure//$'\r'/}"
    if [[ "${weight_func}" == "__EMPTY__" ]]; then
        weight_func=""
    fi
    if [[ "${size_measure}" == "__EMPTY__" ]]; then
        size_measure=""
    fi
    run_count=$((run_count + 1))
    log "Run ${run_count}: level=${level} buffer=${buffer} weight_method=${weight_method} weight_func='${weight_func}' dynamic_buffering=${dynamic_buffering} size_measure=${size_measure}"
    cmd=(
        "${PYTHON_CMD}" -m "${PYTHON_SCRIPT}"
        --level "${level}"
        --version "${VERSION}"
        --buffer "${buffer}"
        --weight-method "${weight_method}"
        --weight-func "${weight_func}"
        --dynamic-buffering "${dynamic_buffering}"
        --approach "${APPROACH}"
    )
    if [[ -n "${size_measure}" ]]; then
        cmd+=(--size-measure "${size_measure}")
    fi
    "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
done

log "Completed task ${TASK_ID}. Executed ${run_count} parameter combinations."
