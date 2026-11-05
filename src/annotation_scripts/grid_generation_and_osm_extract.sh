#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/grid_generation_and_osm_extract_%j.out
#SBATCH --error=logs/grid_generation_and_osm_extract_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "grid_generation_osm_extract"
enable_err_trap

parse_overrides "$@"
build_override_args

log "Installing src module"
ensure_src_importable

log "Running NEW_01_GENERATEGRIDS"
run_stage "NEW_01_GENERATEGRIDS" ${PYTHON_CMD} -m src.annotation_scripts.NEW_01_GENERATEGRIDS "${OVERRIDE_ARGS[@]}"

log "Running NEW_02_EXTRACTOSMDATAFULL_GEOJSON"
run_stage "NEW_02_EXTRACTOSMDATAFULL_GEOJSON" ${PYTHON_CMD} -m src.annotation_scripts.NEW_02_EXTRACTOSMDATAFULL_GEOJSON "${OVERRIDE_ARGS[@]}"

log "Grid generation and OSM extraction completed"