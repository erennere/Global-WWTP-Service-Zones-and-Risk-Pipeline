#!/bin/bash
#
# Population Data Integration Script
# Processes Voronoi polygon layers with population raster data
# Can run locally by specifying file index, or in SLURM job array mode
#
# Usage:
#   ./add_pop.sh --index <index> [--level <n>] [--version <v>] [--buffer <m>] [--weight-method <method>] [--weight-func <func>] [--dynamic-buffering <true|false>] [--dynamic-buffer-k <k>]
#   sbatch add_pop.sh              (SLURM array job - uses SLURM_ARRAY_TASK_ID)
#
# SLURM Configuration
#SBATCH --partition=cpu-single
#SBATCH --time=24:00:00
#SBATCH --mem=192gb
#SBATCH --cpus-per-task=8
#SBATCH --array=0-12
#SBATCH --job-name=add-pop-array
#SBATCH --output=logs/add_pop_%A_%a.out
#SBATCH --error=logs/add_pop_%A_%a.err

set -Eeuo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "add_pop_array"
enable_err_trap
rm -f "${LOG_DIR}"/add_pop_*.out "${LOG_DIR}"/add_pop_*.err

PYTHON_SCRIPT="src.add_pop"

# Parse arguments (named index + named overrides)
INDEX_ARG=""
OVERRIDE_INPUT=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --index)
            if [[ $# -lt 2 ]]; then
                log "ERROR: --index requires a value"
                exit 1
            fi
            INDEX_ARG="$2"
            shift 2
            ;;
        --level|--version|--buffer|--weight-method|--weight-func|--dynamic-buffering|--dynamic-buffer-k)
            if [[ $# -lt 2 ]]; then
                log "ERROR: $1 requires a value"
                exit 1
            fi
            OVERRIDE_INPUT+=("$1" "$2")
            shift 2
            ;;
        *)
            log "ERROR: Unknown argument '$1'"
            log "Usage: $0 --index <file_index> [--level <n>] [--version <v>] [--buffer <m>] [--weight-method <method>] [--weight-func <func>] [--dynamic-buffering <true|false>] [--dynamic-buffer-k <k>]"
            exit 1
            ;;
    esac
done

parse_overrides "${OVERRIDE_INPUT[@]}"

build_override_args

log "=========================================="
log "Population Data Integration Task Started"
log "=========================================="
log "Project root directory: ${PROJECT_ROOT}"

# Determine task ID: from SLURM or command-line argument
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Running in SLURM job array
    TASK_ID="${SLURM_ARRAY_TASK_ID}"
    log "Running in SLURM job array"
    log "Job array ID: ${SLURM_ARRAY_JOB_ID:-unknown}"
    log "Array task ID: ${TASK_ID}"
elif [[ -n "${INDEX_ARG}" ]]; then
    # Local mode with explicit named argument
    TASK_ID="${INDEX_ARG}"
    log "Running in local mode"
    log "Task ID from --index: ${TASK_ID}"
else
    # No task ID provided
    log "ERROR: Task ID not provided"
    log "Usage: $0 --index <file_index> [--level <n>] [--version <v>] [--buffer <m>] [--weight-method <method>] [--weight-func <func>] [--dynamic-buffering <true|false>] [--dynamic-buffer-k <k>]"
    log "   or: sbatch $0 (SLURM mode)"
    exit 1
fi

# Validate task ID is numeric
if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    log "ERROR: Invalid task ID '${TASK_ID}' - must be a non-negative integer"
    exit 1
fi

log "Python command: ${PYTHON_CMD}"

ensure_src_importable

# Validate Python script exists
if ! "${PYTHON_CMD}" -c "import ${PYTHON_SCRIPT}" &> /dev/null; then
    log "ERROR: Python script not found or cannot be imported: ${PYTHON_SCRIPT}"
    exit 1
fi

# Verify Python is available
if ! command -v "${PYTHON_CMD}" &> /dev/null; then
    log "ERROR: Python command '${PYTHON_CMD}' not found"
    exit 1
fi

log "Python version: $(${PYTHON_CMD} --version 2>&1)"
log "Processing Voronoi file index: ${TASK_ID}"

# Run the population data integration
START_TIME=$(date +%s)

if ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" --index "${TASK_ID}" "${OVERRIDE_ARGS[@]}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log "=========================================="
    log "Task ${TASK_ID} Completed Successfully"
    log "Duration: ${DURATION} seconds ($(($DURATION / 60)) minutes)"
    log "=========================================="
    exit 0
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log "=========================================="
    log "ERROR: Task ${TASK_ID} Failed"
    log "Duration: ${DURATION} seconds"
    log "Check error output above for details"
    log "=========================================="
    exit 1
fi