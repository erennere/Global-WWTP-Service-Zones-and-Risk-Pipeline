#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=64
#SBATCH --mem=234gb
#SBATCH --time=96:00:00
#SBATCH --output=logs/pop_differences_and_impact_polygons_%j.out
#SBATCH --error=logs/pop_differences_and_impact_polygons_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
UTILS_PATH="${PROJECT_ROOT}/lib/utils.sh"

if [[ ! -f "${UTILS_PATH}" ]]; then
	echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: ${UTILS_PATH} not found. Submit from src with: sbatch pop_at_risk_river_calculations/pop_differences_and_impact_polygons.sh" >&2
	exit 1
fi

# shellcheck source=lib/utils.sh
source "${UTILS_PATH}"
init_log "pop_differences_and_impact_polygons"

log "Starting pop_differences_and_impact_polygons pipeline"

enable_err_trap

parse_overrides "$@"

build_override_args

log "Checking package importability"
ensure_src_importable

run_stage "find_unserved_pop" ${PYTHON_CMD} -m src.pop_at_risk_river_calculations.find_unserved_pop "${OVERRIDE_ARGS[@]}"

run_stage "assign_rivers_to_basin" ${PYTHON_CMD} -m src.pop_at_risk_river_calculations.assign_rivers_to_basin --max-workers 2 "${OVERRIDE_ARGS[@]}"

run_stage "find_intersection_river" ${PYTHON_CMD} -m src.pop_at_risk_river_calculations.find_intersection_river --max-workers 32 "${OVERRIDE_ARGS[@]}"

run_stage "impact_polygons_pop" ${PYTHON_CMD} -m src.pop_at_risk_river_calculations.impact_polygons_pop --max-workers 64 "${OVERRIDE_ARGS[@]}"

# A dead end: nothing downstream depends on find_diff_pop's output. Sequenced
# last so its failure never aborts the stages that don't need it.
run_stage "find_diff_pop" ${PYTHON_CMD} -m src.pop_at_risk_river_calculations.find_diff_pop --index 0 --is-parallel true "${OVERRIDE_ARGS[@]}"

log "All pop_at_risk pipeline stages completed"

