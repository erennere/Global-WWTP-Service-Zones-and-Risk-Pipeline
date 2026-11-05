#!/bin/bash
# Industrial analysis pipeline: download/vectorize and find unconnected areas
#SBATCH --partition=cpu-single
#SBATCH --time=96:00:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=16
#SBATCH --job-name=industrial-analysis
#SBATCH --output=logs/industrial_analysis_%j.out
#SBATCH --error=logs/industrial_analysis_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "industrial_analysis"
enable_err_trap

# Script paths
DOWNLOAD_SCRIPT="src.industrial_analysis.download_and_vectorize"
FIND_UNCONNECTED_SCRIPT="src.industrial_analysis.find_unconnected_industrial_areas"

# Optional config overrides (all optional)
parse_overrides "$@"

build_override_args

ensure_src_importable

log "Starting industrial analysis pipeline"
log "Configuration:"
log "  - Level: ${LEVEL:-default}"
log "  - Version: ${VERSION:-default}"
log "  - Buffer: ${BUFFER:-default}"
log "  - Weight method: ${WEIGHT_METHOD:-default}"
log "  - Weight func: ${WEIGHT_FUNC:-default}"
log "  - Dynamic buffering: ${DYNAMIC_BUFFERING:-default}"
log "  - Dynamic buffer k: ${DYNAMIC_BUFFER_K:-default}"

# Step 1: Download and vectorize industrial land rasters
log "=========================================="
log "Step 1: Downloading and vectorizing industrial land rasters"
log "=========================================="

if ${PYTHON_CMD} -m "${DOWNLOAD_SCRIPT}" "${OVERRIDE_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
    log "Step 1 completed successfully"
else
    log "ERROR: Step 1 failed"
    exit 1
fi

# Step 2: Find unconnected industrial areas
log "=========================================="
log "Step 2: Finding unconnected industrial areas"
log "=========================================="

if ${PYTHON_CMD} -m "${FIND_UNCONNECTED_SCRIPT}" "${OVERRIDE_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
    log "Step 2 completed successfully"
else
    log "ERROR: Step 2 failed"
    exit 1
fi

log "=========================================="
log "Industrial analysis pipeline completed successfully"
log "=========================================="