#!/bin/bash
#
# Population Data Processing Script
# Downloads and processes global population data from WorldPop
# Supports both GeoTIFF mosaicing and CSV rasterization
#
# SLURM Configuration
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=96:00:00
#SBATCH --job-name=pop-processing
#SBATCH --output=logs/pop_%j.out
#SBATCH --error=logs/pop_%j.err

set -Eeuo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "pop_run"
enable_err_trap
rm -f "${LOG_DIR}"/pop_*.out "${LOG_DIR}"/pop_*.err

PYTHON_SCRIPT="src.download_pop"

parse_overrides "$@"

build_override_args

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

log "=========================================="
log "Population Data Processing Started"
log "=========================================="
log "Project root directory: ${PROJECT_ROOT}"
log "Python command: ${PYTHON_CMD}"
log "Processing with 8 parallel workers"
log "Python version: $(${PYTHON_CMD} --version 2>&1)"

# Run the population data processing
log "Starting population data download and processing..."
START_TIME=$(date +%s)

if ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" "${OVERRIDE_ARGS[@]}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log "=========================================="
    log "Population Data Processing Completed Successfully"
    log "Duration: ${DURATION} seconds ($(($DURATION / 60)) minutes)"
    log "=========================================="
    exit 0
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    log "=========================================="
    log "ERROR: Population Data Processing Failed"
    log "Duration: ${DURATION} seconds"
    log "Check SLURM error output for details"
    log "=========================================="
    exit 1
fi