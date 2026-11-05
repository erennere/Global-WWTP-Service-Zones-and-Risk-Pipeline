#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --time=24:00:00
#SBATCH --mem=16gb
#SBATCH --cpus-per-task=4
#SBATCH --array=0
#SBATCH --output=logs/comparison_%j.out
#SBATCH --error=logs/comparison_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "pop_validation_comparison"
enable_err_trap

parse_overrides "$@"

build_override_args

ensure_src_importable

log "Running verification_script"
run_stage "verification_script" ${PYTHON_CMD} -m src.pop_validation_scripts.verification_script "${OVERRIDE_ARGS[@]}"

log "Running hw_comparison"
run_stage "hw_comparison" ${PYTHON_CMD} -m src.pop_validation_scripts.hw_comparison "${OVERRIDE_ARGS[@]}"

log "Running eu_comparison"
run_stage "eu_comparison" ${PYTHON_CMD} -m src.pop_validation_scripts.eu_comparison "${OVERRIDE_ARGS[@]}"

log "All pop validation comparisons completed"