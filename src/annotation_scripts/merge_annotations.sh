#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/merge_annotations_%j.out
#SBATCH --error=logs/merge_annotations_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "merge_annotations"
enable_err_trap

parse_overrides "$@"
build_override_args

log "Checking package importability..."
ensure_src_importable

log "Running merge_annotations.py"
run_stage "merge_annotations" ${PYTHON_CMD} -m src.annotation_scripts.merge_annotations "${OVERRIDE_ARGS[@]}"
