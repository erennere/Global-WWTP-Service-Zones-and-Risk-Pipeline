#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=96:00:00
#SBATCH --output=logs/find_pop_in_danger_pop_%j.out
#SBATCH --error=logs/find_pop_in_danger_pop_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "find_pop_in_danger_pop"
enable_err_trap

parse_overrides "$@"

build_override_args

ensure_src_importable

log "Running find_pop_in_danger_pop"
run_stage "find_pop_in_danger_pop" ${PYTHON_CMD} -m src.pop_at_risk_river_calculations.find_pop_in_danger_pop "${OVERRIDE_ARGS[@]}"
log "Completed find_pop_in_danger_pop"






