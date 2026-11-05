#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --time=24:00:00
#SBATCH --mem=8gb
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/pop_at_risk_figures_%j.out
#SBATCH --error=logs/pop_at_risk_figures_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "pop_at_risk_figures"
enable_err_trap

parse_overrides "$@"

build_override_args

PYTHON_SCRIPT="src.figures_scripts.pop_at_risk_figures"

ensure_src_importable

log "Running pop_at_risk_figures"
run_stage "${PYTHON_SCRIPT}" ${PYTHON_CMD} -m "${PYTHON_SCRIPT}" "${OVERRIDE_ARGS[@]}"

log "Completed pop_at_risk_figures"
