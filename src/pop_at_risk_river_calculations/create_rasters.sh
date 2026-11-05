#!/bin/bash

################################################################################
# create_rasters.sh - Raster Processing Pipeline
#
# Orchestrates raster execution with configurable modes.
#
# Execution Modes:
#   ARRAY JOB:  One index per SLURM task (via SLURM_ARRAY_TASK_ID)
#   SEQUENTIAL: Single local run with index 0 and total 1
#   PARALLEL:   Multiple indices concurrently on different CPUs
#
# Configuration (from config.yaml):
#   annotations.default_mode: array | sequential | parallel
#
#SBATCH --partition=cpu-single
#SBATCH --time=48:00:00
#SBATCH --mem=192gb
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9
#SBATCH --output=logs/create_rasters_%j.out
#SBATCH --error=logs/create_rasters_%j.err
#
################################################################################

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "create_rasters"
enable_err_trap

PYTHON_SCRIPT="src.pop_at_risk_river_calculations.create_rasters"

#install_package

if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
    log "ERROR: PYTHON_CMD '${PYTHON_CMD}' not found on PATH"
    exit 1
fi

export_thread_vars

parse_overrides "$@"

build_override_args

# Resolve runtime execution settings through starter.py so section inheritance
# and CLI overrides apply consistently.
RASTER_CONFIG=()
mapfile -t RASTER_CONFIG < <(
    LEVEL_OVERRIDE="${LEVEL}" \
    VERSION_OVERRIDE="${VERSION}" \
    BUFFER_OVERRIDE="${BUFFER}" \
    WEIGHT_METHOD_OVERRIDE="${WEIGHT_METHOD}" \
    WEIGHT_FUNC_OVERRIDE="${WEIGHT_FUNC}" \
    DYNAMIC_BUFFERING_OVERRIDE="${DYNAMIC_BUFFERING}" \
    DYNAMIC_BUFFER_K_OVERRIDE="${DYNAMIC_BUFFER_K}" \
    "${PYTHON_CMD}" - <<'PY'
import os

from src.starter import load_config


def env_value(name, preserve_empty=False):
    value = os.environ.get(name)
    if value is None:
        return None
    if preserve_empty:
        return value
    return value or None


cfg = load_config(
    script_name="create_rasters",
    level=env_value("LEVEL_OVERRIDE"),
    version=env_value("VERSION_OVERRIDE"),
    buffer=env_value("BUFFER_OVERRIDE"),
    weight_method=env_value("WEIGHT_METHOD_OVERRIDE"),
    weight_func=env_value("WEIGHT_FUNC_OVERRIDE", preserve_empty=True),
    dynamic_buffering=env_value("DYNAMIC_BUFFERING_OVERRIDE"),
    dynamic_buffer_k=env_value("DYNAMIC_BUFFER_K_OVERRIDE"),
)
annotations = cfg.get("annotations") or {}
print(os.path.abspath("config.yaml"))
print(annotations.get("default_mode", ""))
print(annotations.get("max_workers", ""))
PY
)
RESOLVED_CONFIG_PATH="${RASTER_CONFIG[0]:-}"
DEFAULT_MODE="${RASTER_CONFIG[1]:-}"
CONFIG_MAX_WORKERS="${RASTER_CONFIG[2]:-}"

if [[ -z "$DEFAULT_MODE" ]]; then
    log "ERROR: Missing create_rasters.annotations.default_mode in config resolution"
    exit 1
fi

MODE="${MODE:-$DEFAULT_MODE}"

log "Mode resolution diagnostics:"
log "  - Working directory: ${PROJECT_ROOT}"
log "  - Resolved config path: ${RESOLVED_CONFIG_PATH:-unresolved}"
log "  - Config default_mode: ${DEFAULT_MODE:-unset}"
log "  - Environment MODE: ${MODE:-unset}"
log "Execution mode: ${MODE}"

if [[ "$MODE" == "array" ]] && [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Array mode: one index per SLURM task.
    JOB_INDEX="$SLURM_ARRAY_TASK_ID"
    if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
        TOTAL_JOBS="${SLURM_ARRAY_TASK_COUNT}"
    elif [[ -n "${SLURM_ARRAY_TASK_MIN:-}" && -n "${SLURM_ARRAY_TASK_MAX:-}" ]]; then
        TOTAL_JOBS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
    else
        TOTAL_JOBS=1
    fi
    log "Running raster job ${JOB_INDEX} of ${TOTAL_JOBS} in array mode"
    run_stage "${PYTHON_SCRIPT}" ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" --job-index "$JOB_INDEX" --total-jobs "$TOTAL_JOBS" "${OVERRIDE_ARGS[@]}"
elif [[ "$MODE" == "sequential" ]]; then
    # Sequential: only run on task 0 (skip other array tasks if present)
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && [[ $SLURM_ARRAY_TASK_ID -ne 0 ]]; then
        log "Sequential mode: skipping task $SLURM_ARRAY_TASK_ID (only task 0 runs)"
        exit 0
    fi
    log "Running raster processing in sequential mode"
    run_stage "${PYTHON_SCRIPT}" ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" --job-index 0 --total-jobs 1 "${OVERRIDE_ARGS[@]}"
elif [[ "$MODE" == "parallel" ]]; then
    # Parallel: run indices 0..X-1 concurrently.
    TOTAL_JOBS=${CONFIG_MAX_WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 1)}}
    FAILED_COUNT=0
    
    log "Running ${TOTAL_JOBS} raster jobs in parallel"
    
    for ((JOB_INDEX=0; JOB_INDEX<TOTAL_JOBS; JOB_INDEX++)); do
        log "Launching raster job ${JOB_INDEX} in background"
        ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" --job-index "$JOB_INDEX" --total-jobs "$TOTAL_JOBS" "${OVERRIDE_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}" &
    done
    
    # Wait for all background jobs to complete
    for job in $(jobs -p); do
        if ! wait $job; then
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    done
    
    if [[ $FAILED_COUNT -gt 0 ]]; then
        log "ERROR: $FAILED_COUNT parallel job(s) failed"
        exit 1
    fi
    log "All raster jobs completed successfully"
else
    log "ERROR: Unknown execution mode '$MODE' (valid: array, sequential, parallel)"
    exit 1
fi

log "create_rasters execution completed"
