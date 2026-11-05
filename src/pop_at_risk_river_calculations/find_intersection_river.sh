#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/find_intersection_river_%j.out
#SBATCH --error=logs/find_intersection_river_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "find_intersection_river"
enable_err_trap

parse_overrides "$@"

build_override_args

ensure_src_importable

log "Running find_intersection_river with 32 workers"
run_stage "find_intersection_river" ${PYTHON_CMD} -m src.pop_at_risk_river_calculations.find_intersection_river --max-workers 32 "${OVERRIDE_ARGS[@]}"
log "Completed find_intersection_river"