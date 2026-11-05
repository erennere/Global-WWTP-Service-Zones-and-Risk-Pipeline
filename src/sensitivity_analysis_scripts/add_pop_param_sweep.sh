#!/bin/bash
# Lightweight parameter sweep for add_pop
#SBATCH --partition=cpu-single
#SBATCH --time=96:00:00
#SBATCH --mem=192gb
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9
#SBATCH --job-name=add-pop-sweep
#SBATCH --output=logs/add_pop_sweep_%A_%a.out
#SBATCH --error=logs/add_pop_sweep_%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"

PYTHON_SCRIPT="src.add_pop"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

parse_overrides "$@"

# Parameter grids
LEVELS=(6 7 8 9)
WEIGHT_FUNCS=("mult" "add" "")
WEIGHT_METHODS=("linear" "logarithmic" "square_root" "sigmoid")
BUFFERS=(9000 11000 13000 15000)
DYNAMIC_K_VALUES=(0.6 0.7 0.8)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
init_log "add_pop_sweep_${TASK_ID}"
enable_err_trap
rm -f "${LOG_DIR}"/add_pop_sweep_*.out "${LOG_DIR}"/add_pop_sweep_*.err

if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    log "ERROR: Invalid task id '${TASK_ID}'"
    exit 1
fi
if (( TASK_ID < 0 || TASK_ID > 9 )); then
    log "ERROR: TASK_ID must be between 0 and 9 (got ${TASK_ID})"
    exit 1
fi

log "Starting add_pop parameter sweep task ${TASK_ID}/9, shuffle_seed=${SHUFFLE_SEED}"

#wwinstall_package

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
    level="${level//$'\r'/}"
    buffer="${buffer//$'\r'/}"
    weight_method="${weight_method//$'\r'/}"
    weight_func="${weight_func//$'\r'/}"
    dynamic_buffering="${dynamic_buffering//$'\r'/}"
    dynamic_buffer_k="${dynamic_buffer_k//$'\r'/}"
    if [[ "${weight_func}" == "__EMPTY__" ]]; then
        weight_func=""
    fi
    if [[ "${dynamic_buffer_k}" == "__EMPTY__" ]]; then
        dynamic_buffer_k=""
    fi

    # Discover how many voronoi files exist for this parameter combination.
    # Each (level, buffer, weight_func) maps to a different voronoi directory,
    # so the file count must be determined at runtime rather than using TASK_ID.
    raw_num_files=$(${PYTHON_CMD} - "${level}" "${VERSION}" "${buffer}" "${weight_method}" "${weight_func}" "${dynamic_buffering}" "${dynamic_buffer_k}" <<'PY'
import os, sys
from src.starter import load_config, parse_config_overrides

level, version, buffer, weight_method, weight_func, dynamic_buffering, dynamic_buffer_k = sys.argv[1:]
argv = [
    '',
    '--level', level,
    '--version', version,
    '--buffer', buffer,
    '--weight-method', weight_method,
    '--weight-func', weight_func,
    '--dynamic-buffering', dynamic_buffering,
]
if dynamic_buffer_k:
    argv.extend(['--dynamic-buffer-k', dynamic_buffer_k])
sys.argv = argv

overrides = parse_config_overrides()
cfg = load_config(script_name='add_pop', **overrides)
d = cfg['paths']['voronoi_dir']
files = sorted(
    f for f in os.listdir(d)
    if f.endswith('.gpkg') and not f.startswith('temp_')
) if os.path.isdir(d) else []
print(len(files))
PY
    2>/dev/null || true)

    num_files=$(printf '%s\n' "${raw_num_files}" | tr -d '\r' | grep -E '^[0-9]+$' | tail -n 1)
    if [[ -z "${num_files}" ]]; then
        num_files=0
    fi

    if (( num_files == 0 )); then
        log "WARNING: No voronoi files found for level=${level} buffer=${buffer} weight_method=${weight_method} weight_func='${weight_func}' dynamic_buffering=${dynamic_buffering} dynamic_buffer_k=${dynamic_buffer_k} - skipping"
        continue
    fi

    for file_idx in $(seq 0 $((num_files - 1))); do
        run_count=$((run_count + 1))
        log "Run ${run_count}: file_index=${file_idx}/${num_files} level=${level} buffer=${buffer} weight_method=${weight_method} weight_func='${weight_func}' dynamic_buffering=${dynamic_buffering} dynamic_buffer_k=${dynamic_buffer_k}"
        cmd=(
            "${PYTHON_CMD}" -m "${PYTHON_SCRIPT}"
            --index "${file_idx}"
            --level "${level}"
            --version "${VERSION}"
            --buffer "${buffer}"
            --weight-method "${weight_method}"
            --weight-func "${weight_func}"
            --dynamic-buffering "${dynamic_buffering}"
        )
        if [[ -n "${dynamic_buffer_k}" ]]; then
            cmd+=(--dynamic-buffer-k "${dynamic_buffer_k}")
        fi
        "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
    done
done

log "Completed task ${TASK_ID}. Executed ${run_count} parameter combinations."
