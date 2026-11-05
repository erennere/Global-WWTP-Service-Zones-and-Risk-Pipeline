#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --time=12:00:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=16
#SBATCH --job-name=voronoi-additive-single
#SBATCH --output=logs/voronoi_additive_single_%j.out
#SBATCH --error=logs/voronoi_additive_single_%j.err

# Quick single-run sanity check for additive Voronoi mode.
#
# Usage:
#   sbatch sensitivity_analysis_scripts/create_voronoi_single_additive_test.sh [--approach <n>] [--level <n>] [--version <v>] [--buffer <m>] [--weight-method <method>] [--dynamic-buffering <true|false>] [--size-measure <reported|footprint>]
#   bash sensitivity_analysis_scripts/create_voronoi_single_additive_test.sh [--approach <n>] [--level <n>] [--version <v>] [--buffer <m>] [--weight-method <method>] [--dynamic-buffering <true|false>] [--size-measure <reported|footprint>]
#
# Defaults:
#   approach=1 level=8 version='' buffer=9000 weight_method=logarithmic dynamic_buffering=false size_measure=''

set -Eeuo pipefail

PROJECT_ROOT="."

# shellcheck source=lib/utils.sh
source "${PROJECT_ROOT}/lib/utils.sh"

APPROACH="1"
LEVEL="8"
VERSION=""
BUFFER="9000"
WEIGHT_METHOD="logarithmic"
DYNAMIC_BUFFERING="false"
SIZE_MEASURE=""
WEIGHT_FUNC="add"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --approach)
            APPROACH="$2"
            shift 2
            ;;
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --buffer)
            BUFFER="$2"
            shift 2
            ;;
        --weight-method)
            WEIGHT_METHOD="$2"
            shift 2
            ;;
        --dynamic-buffering)
            DYNAMIC_BUFFERING="$2"
            shift 2
            ;;
        --size-measure)
            SIZE_MEASURE="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument '$1'" >&2
            echo "Usage: $0 [--approach <n>] [--level <n>] [--version <v>] [--buffer <m>] [--weight-method <method>] [--dynamic-buffering <true|false>] [--size-measure <reported|footprint>]" >&2
            exit 1
            ;;
    esac
done

init_log "create_voronoi_single_additive_test"
enable_err_trap
log "Starting single additive Voronoi sanity run"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    log "SLURM job detected: id=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK:-unknown}"
fi

ensure_src_importable
export_thread_vars

build_override_args

log "Parameters: approach=${APPROACH} level=${LEVEL} version=${VERSION:-default} buffer=${BUFFER} weight_method=${WEIGHT_METHOD} weight_func=${WEIGHT_FUNC} dynamic_buffering=${DYNAMIC_BUFFERING} size_measure=${SIZE_MEASURE:-default}"

run_stage "create_voronoi" ${PYTHON_CMD} -m src.create_voronoi "${OVERRIDE_ARGS[@]}" --approach "${APPROACH}"

log "Single additive Voronoi sanity run completed"
