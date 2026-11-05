#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --time=24:00:00
#SBATCH --mem=8gb
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/composite_area_population_plots_%j.out
#SBATCH --error=logs/composite_area_population_plots_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "composite_area_population_plots"
enable_err_trap

PYTHON_SCRIPT="src.figures_scripts.composite_area_population_plots"

#
# Usage:
#   ./composite_area_population_plots.sh [level] [version] [buffer] [weight_method] [weight_func] [dynamic_buffering] [dynamic_buffer_k] [approach] [color_col]
#
# Optional config overrides:
#   level, version, buffer, weight_method, weight_func, dynamic_buffering, dynamic_buffer_k
# Optional plotting args:
#   approach  - key from create_pop_output_paths: 0 | 1 | 2 | 0_only_round | 1_only_round
#   color_col - boundary column used for color coding (default: ECONOMY)
parse_overrides "$@"
APPROACH="${8:-}"
COLOR_COL="${9:-ECONOMY}"

build_override_args

ensure_src_importable

CMD=(
    "${PYTHON_CMD}" -m "${PYTHON_SCRIPT}"
    "${OVERRIDE_ARGS[@]}"
    --color-col "${COLOR_COL}"
)

if [[ -n "${APPROACH}" ]]; then
    CMD+=(--approach "${APPROACH}")
fi

log "Running composite area/population plots"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
log "Completed composite area/population plots"
