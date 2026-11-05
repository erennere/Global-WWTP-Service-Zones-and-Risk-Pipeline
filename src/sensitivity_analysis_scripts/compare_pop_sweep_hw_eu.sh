#!/bin/bash
# Run sensitivity analysis across all pop_voronoi_layers outputs (HW vs EU).
#SBATCH --partition=cpu-single
#SBATCH --time=96:00:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8
#SBATCH --job-name=compare-pop-sweep-hw-eu
#SBATCH --output=logs/compare_pop_sweep_hw_eu_%j.out
#SBATCH --error=logs/compare_pop_sweep_hw_eu_%j.err
# Usage:
#   bash src/sensitivity_analysis_scripts/compare_pop_sweep_hw_eu.sh [level] [version] [buffer] [weight_method] [weight_func] [dynamic_buffering] [dynamic_buffer_k]

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "compare_pop_sweep_hw_eu"
enable_err_trap

PYTHON_MODULE="src.sensitivity_analysis_scripts.compare_pop_sweep_hw_eu"
MAX_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

#install_package

log "Running HW/EU sensitivity analysis for all pop-output GPKGs (workers=${MAX_WORKERS})"
run_stage "${PYTHON_MODULE}" ${PYTHON_CMD} -m "${PYTHON_MODULE}" "$@"

log "Completed HW/EU sensitivity analysis"
