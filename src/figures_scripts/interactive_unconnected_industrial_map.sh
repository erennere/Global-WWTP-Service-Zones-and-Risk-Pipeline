#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --time=02:00:00
#SBATCH --mem=4gb
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/interactive_unconnected_industrial_map_%j.out
#SBATCH --error=logs/interactive_unconnected_industrial_map_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."

# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "interactive_unconnected_industrial_map"
enable_err_trap

# Usage:
#   ./figures_scripts/interactive_unconnected_industrial_map.sh [level] [version] [buffer] [weight_method] [weight_func] [dynamic_buffering] [dynamic_buffer_k]
#
# Works in both:
#   1) plain bash: bash figures_scripts/interactive_unconnected_industrial_map.sh
#   2) HPC: sbatch figures_scripts/interactive_unconnected_industrial_map.sh
parse_overrides "$@"

build_override_args

PYTHON_SCRIPT="src.figures_scripts.interactive_unconnected_industrial_map"

ensure_src_importable
log "Editable install completed"

log "Running interactive unconnected industrial demo map generation"
run_stage "${PYTHON_SCRIPT}" ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" "${OVERRIDE_ARGS[@]}"
log "Completed interactive unconnected industrial demo map generation"
