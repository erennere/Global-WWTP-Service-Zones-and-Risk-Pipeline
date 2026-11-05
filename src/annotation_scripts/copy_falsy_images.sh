#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/copy_falsy_images_%j.out
#SBATCH --error=logs/copy_falsy_images_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "copy_falsy_images"
enable_err_trap

parse_overrides "$@"
build_override_args

log "Checking package importability..."
ensure_src_importable

log "Running copy_falsy_images.py"
run_stage "copy_falsy_images" ${PYTHON_CMD} -m src.annotation_scripts.copy_falsy_images "${OVERRIDE_ARGS[@]}"
log "Completed copy_falsy_images.py"
