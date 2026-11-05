#!/bin/bash
#
# Combined Location Data Merge Pipeline
# Processes: correct OSM locations -> merge segmentation results -> combine locations -> final merge
# Orchestrates multiple sequential data processing steps
#
# Usage:
#   ./combine_locations.sh [level] [version] [buffer] [weight_method] [weight_func] [dynamic_buffering] [dynamic_buffer_k]
#   sbatch combine_locations.sh (SLURM job)
#
# SLURM Configuration
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --job-name=combine-locations
#SBATCH --output=logs/combine_locations_%j.out
#SBATCH --error=logs/combine_locations_%j.err

set -Eeuo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "combine_locations"
enable_err_trap
rm -f "${LOG_DIR}/merge_seg_results.log" "${LOG_DIR}/combine_locations_"*.out "${LOG_DIR}/combine_locations_"*.err

parse_overrides "$@"

build_override_args

log "=========================================="
log "Combined Location Data Merge Started"
log "=========================================="
log "Project root directory: ${PROJECT_ROOT}"

log "Checking package importability..."
ensure_src_importable

log "Starting data merge pipeline..."

# Step 1: Correct OSM locations
log "Step 1: Correcting locations with OSM data..."
run_stage "correct_locations_w_OSM" ${PYTHON_CMD} -m src.data_merge.correct_locations_w_OSM "${OVERRIDE_ARGS[@]}"
log "Step 1 completed"

# Step 2: Optionally merge legacy segmentation outputs.
# Whether this actually runs is controlled by booleans.legacy_merge in config.yaml.
log "Step 2: Running legacy segmentation merge if enabled in config.yaml..."
${PYTHON_CMD} -m src.data_merge.merge_seg_results "${OVERRIDE_ARGS[@]}" --variant old 2>&1 | tee -a "${LOG_DIR}/merge_seg_results.log"
log "Step 2 completed"

# Step 3: Combine locations
log "Step 3: Combining location data..."
run_stage "final_data_merge" ${PYTHON_CMD} -m src.data_merge.final_data_merge "${OVERRIDE_ARGS[@]}"
log "Step 3 completed"

log "Running merge_seg_results (variant=new)"
${PYTHON_CMD} -m src.data_merge.merge_seg_results "${OVERRIDE_ARGS[@]}" --variant new 2>&1 | tee -a "${LOG_DIR}/merge_seg_results.log"
log "Completed merge_seg_results (variant=new)"

log "=========================================="
log "Combined Location Data Merge Completed"
log "=========================================="