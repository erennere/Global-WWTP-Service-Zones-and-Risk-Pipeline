#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/annotations_inspection_%j.out
#SBATCH --error=logs/annotations_inspection_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "annotations_inspection"
enable_err_trap

parse_overrides "$@"
build_override_args

log "Checking package importability..."
ensure_src_importable

log "Running annotations_inspection.py"
run_stage "annotations_inspection" ${PYTHON_CMD} -m src.annotation_scripts.annotations_inspection "${OVERRIDE_ARGS[@]}"
