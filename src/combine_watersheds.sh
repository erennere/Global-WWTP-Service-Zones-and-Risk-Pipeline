#!/bin/bash
#
# Watershed Archive Merge Script
# Extracts watershed zip archives and merges discovered geospatial layers.
#
# SLURM Configuration
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --job-name=combine-watersheds
#SBATCH --output=logs/combine_watersheds_%j.out
#SBATCH --error=logs/combine_watersheds_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "combine_watersheds"
enable_err_trap
rm -f "${LOG_DIR}/combine_watersheds_"*.out "${LOG_DIR}/combine_watersheds_"*.err

PYTHON_SCRIPT="src.combine_watersheds"

parse_overrides "$@"

build_override_args

ensure_src_importable

if ! command -v "${PYTHON_CMD}" &> /dev/null; then
    log "ERROR: Python command '${PYTHON_CMD}' not found"
    exit 1
fi

if ! ${PYTHON_CMD} -c "import ${PYTHON_SCRIPT}" &> /dev/null; then
    log "ERROR: Python script not found or cannot be imported: ${PYTHON_SCRIPT}"
    exit 1
fi

log "=========================================="
log "Watershed Merge Started"
log "=========================================="
log "Project root directory: ${PROJECT_ROOT}"
log "Python command: ${PYTHON_CMD}"
log "Python version: $(${PYTHON_CMD} --version 2>&1)"
log "Starting watershed archive merge"

START_TIME=$(date +%s)

if ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" "${OVERRIDE_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log "=========================================="
    log "Watershed Merge Completed Successfully"
    log "Duration: ${DURATION} seconds ($(($DURATION / 60)) minutes)"
    log "=========================================="
    exit 0
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log "=========================================="
    log "ERROR: Watershed Merge Failed"
    log "Duration: ${DURATION} seconds"
    log "Check SLURM error output for details"
    log "=========================================="
    exit 1
fi