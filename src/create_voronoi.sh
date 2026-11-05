#!/bin/bash

################################################################################
# create_voronoi.sh - Voronoi Spatial Allocation Pipeline
#
# Orchestrates approach execution with configurable modes.
#
# Execution Modes:
#   ARRAY JOB:  One approach per SLURM task (via SLURM_ARRAY_TASK_ID)
#   SEQUENTIAL: One approach after another (default for local)
#   PARALLEL:   Multiple approaches concurrently on different CPUs
#
# Configuration (from config.yaml):
#   execution.mode: array | sequential | parallel
#
#SBATCH --partition=cpu-single
#SBATCH --time=48:00:00
#SBATCH --mem=192gb
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2
#SBATCH --output=logs/create_voronoi_%j.out
#SBATCH --error=logs/create_voronoi_%j.err
#
################################################################################

set -Eeuo pipefail

# Change to project root from the current working directory.
PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "create_voronoi"
enable_err_trap

PYTHON_SCRIPT="src.create_voronoi"

ensure_src_importable

export_thread_vars

parse_overrides "$@"

build_override_args

# Determine execution mode: environment default + config override
# Defaults: array on HPC, sequential on local
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    DEFAULT_MODE="array"
else
    DEFAULT_MODE="sequential"
fi

# Resolve execution.mode through starter.py so section inheritance and CLI
# overrides apply consistently.
MODE=$(
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
    script_name="create_voronoi",
    level=env_value("LEVEL_OVERRIDE"),
    version=env_value("VERSION_OVERRIDE"),
    buffer=env_value("BUFFER_OVERRIDE"),
    weight_method=env_value("WEIGHT_METHOD_OVERRIDE"),
    weight_func=env_value("WEIGHT_FUNC_OVERRIDE", preserve_empty=True),
    dynamic_buffering=env_value("DYNAMIC_BUFFERING_OVERRIDE"),
    dynamic_buffer_k=env_value("DYNAMIC_BUFFER_K_OVERRIDE"),
)
print((cfg.get("execution") or {}).get("mode", ""))
PY
)
MODE=${MODE:-$DEFAULT_MODE}

log "Execution mode: ${MODE}"

if [[ "$MODE" == "array" ]] && [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Array job mode: one approach per SLURM task
    APPROACHES=('0' '1' '2')
    APPROACH="${APPROACHES[$SLURM_ARRAY_TASK_ID]}"
    log "Running approach ${APPROACH} in array mode (task ${SLURM_ARRAY_TASK_ID})"
    run_stage "${PYTHON_SCRIPT}" ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" "${OVERRIDE_ARGS[@]}" --approach "$APPROACH"
elif [[ "$MODE" == "sequential" ]]; then
    # Sequential: only run on task 0 (skip other array tasks if present)
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && [[ $SLURM_ARRAY_TASK_ID -ne 0 ]]; then
        log "Sequential mode: skipping task $SLURM_ARRAY_TASK_ID (only task 0 runs)"
        exit 0
    fi
    log "Running all approaches in sequential mode"
    run_stage "${PYTHON_SCRIPT}" ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" "${OVERRIDE_ARGS[@]}"
elif [[ "$MODE" == "parallel" ]]; then
    # Parallel: run multiple approaches concurrently on different CPUs
    log "Running all approaches in parallel mode"
    APPROACHES=('0' '1' '2')
    FAILED_APPROACHES=()
    
    for APPROACH in "${APPROACHES[@]}"; do
        log "Launching approach ${APPROACH} in background"
        # Launch approach in background
        ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" "${OVERRIDE_ARGS[@]}" --approach "$APPROACH" 2>&1 | tee -a "${LOG_FILE}" &
    done
    
    # Wait for all background jobs to complete
    for job in $(jobs -p); do
        if ! wait $job; then
            FAILED_APPROACHES+=("$job")
        fi
    done
    
    if [[ ${#FAILED_APPROACHES[@]} -gt 0 ]]; then
        log "ERROR: Some approaches failed"
        exit 1
    fi
    log "All approaches completed successfully"
else
    log "ERROR: Unknown execution mode '$MODE' in config.yaml (valid: array, sequential, parallel)"
    exit 1
fi

log "create_voronoi execution completed"
