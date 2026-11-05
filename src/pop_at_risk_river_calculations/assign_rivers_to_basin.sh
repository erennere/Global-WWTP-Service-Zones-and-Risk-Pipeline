#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/assign_rivers_to_basin_%j.out
#SBATCH --error=logs/assign_rivers_to_basin_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "assign_rivers_to_basin"
enable_err_trap

parse_overrides "$@"

build_override_args

ensure_src_importable

log "Running assign_rivers_to_basin with 2 workers"
run_stage "assign_rivers_to_basin" ${PYTHON_CMD} -m src.pop_at_risk_river_calculations.assign_rivers_to_basin --max-workers 2 "${OVERRIDE_ARGS[@]}"
log "Completed assign_rivers_to_basin"