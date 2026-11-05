#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --time=24:00:00
#SBATCH --mem=8gb
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/piechart_%j.out
#SBATCH --error=logs/piechart_%j.err

set -Eeuo pipefail

PROJECT_ROOT="."
# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"
init_log "piechart"
enable_err_trap

# Usage:
#   ./figures_scripts/piechart.sh [mode] [level] [version] [buffer] [weight_method] [weight_func] [dynamic_buffering] [dynamic_buffer_k]
#
# mode:
#   static       -> run only piechart_figure.py
#   interactive  -> run only piechart_interactive.py
#   sizes        -> run only sizes_interactive_map.py
#   both         -> run both (default)

MODE="${1:-both}"
case "${MODE}" in
	static|interactive|sizes|both)
		shift || true
		;;
	*)
		# If first arg is not a mode token, treat all args as overrides and run both.
		MODE="both"
		;;
esac

parse_overrides "$@"

build_override_args

STATIC_SCRIPT="src.figures_scripts.piechart_figure"
INTERACTIVE_SCRIPT="src.figures_scripts.piechart_interactive"
SIZES_SCRIPT="src.figures_scripts.sizes_interactive_map"

ensure_src_importable

if [[ "${MODE}" == "static" || "${MODE}" == "both" ]]; then
	log "Running piechart_figure"
	run_stage "${STATIC_SCRIPT}" ${PYTHON_CMD} -m "${STATIC_SCRIPT}" "${OVERRIDE_ARGS[@]}"
	log "Completed piechart_figure"
fi

if [[ "${MODE}" == "interactive" || "${MODE}" == "both" ]]; then
	log "Running piechart_interactive"
	run_stage "${INTERACTIVE_SCRIPT}" ${PYTHON_CMD} -m "${INTERACTIVE_SCRIPT}" "${OVERRIDE_ARGS[@]}"
	log "Completed piechart_interactive"
fi

if [[ "${MODE}" == "sizes" || "${MODE}" == "both" || "${MODE}" == "interactive" ]]; then
	log "Running sizes_interactive_map"
	run_stage "${SIZES_SCRIPT}" ${PYTHON_CMD} -m "${SIZES_SCRIPT}" "${OVERRIDE_ARGS[@]}"
	log "Completed sizes_interactive_map"
fi
