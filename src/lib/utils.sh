#!/bin/bash
# src/lib/utils.sh — Shared shell utilities for all pipeline scripts.
#
# LAUNCH DIRECTORY INVARIANT
# ────────────────────────────────────────────────────────────────────────────
# Every wrapper in this repo is run from src/ and nowhere else, for both plain
# bash and sbatch:
#
#     cd src
#     bash pop_at_risk_river_calculations/create_rasters.sh --level 8
#     sbatch pop_at_risk_river_calculations/create_rasters.sh --level 8
#
# That is why PROJECT_ROOT is the literal ".". Source this file with:
#   PROJECT_ROOT="."
#   source "${PROJECT_ROOT}/lib/utils.sh"
#
# Provides
# ────────────────────────────────────────────────────────────────────────────
#   init_log <name>        — set LOG_FILE="${LOG_DIR}/<name>.log", create log
#                            directory, rotate any previous log aside
#   log <message>          — timestamped line written to stdout and LOG_FILE
#   enable_err_trap        — install the standard ERR trap (needs set -E)
#   run_stage <name> <cmd> — run one pipeline stage, tee it, abort on failure
#   parse_overrides <args> — populate the seven standard config-override vars:
#                            LEVEL VERSION BUFFER WEIGHT_METHOD WEIGHT_FUNC
#                            DYNAMIC_BUFFERING DYNAMIC_BUFFER_K
#                            Unknown --flags are rejected, not ignored.
#   build_override_args    — assemble OVERRIDE_ARGS from the parsed vars
#   ensure_src_importable  — editable install only when 'src' is not importable
#   export_thread_vars     — export OMP/OPENBLAS/MKL/NUMEXPR thread counts

LOG_DIR="${PROJECT_ROOT}/logs"
PYTHON_CMD="${PYTHON_CMD:-python}"

# ── Logging ───────────────────────────────────────────────────────────────────

# init_log <name>
#   Sets LOG_FILE and creates ${LOG_DIR}. Any previous log of that name is
#   rotated aside (not deleted) so consecutive runs stay diffable - this is
#   what a parity gate diffs against.
init_log() {
    local name="$1"
    LOG_FILE="${LOG_DIR}/${name}.log"
    mkdir -p "${LOG_DIR}"
    if [[ -f "${LOG_FILE}" ]]; then
        mv -f "${LOG_FILE}" "${LOG_FILE}.$(date +'%Y%m%d%H%M%S')"
    fi
}

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# ── Failure handling ──────────────────────────────────────────────────────────

# enable_err_trap
#   Installs the standard ERR trap. Requires `set -E` so the trap is inherited
#   by shell functions and subshells; call after init_log so log() has a file.
enable_err_trap() {
    trap 'rc=$?; log "ERROR: command failed (exit ${rc}) near line ${BASH_LINENO[0]}: ${BASH_COMMAND}"; exit ${rc}' ERR
}

# run_stage <stage_name> <command> [args...]
#   Runs one pipeline stage, tees its output to the log, and aborts the script
#   with the stage's own exit code if it fails. Without this, `cmd | tee` masks
#   the command's status behind tee's.
run_stage() {
    local stage_name="$1"
    shift
    log "Running ${stage_name}"
    # The ERR trap fires regardless of `set +e`, so it is suspended here to let
    # this function report which *stage* failed instead of which line did.
    local saved_err_trap
    saved_err_trap="$(trap -p ERR)"
    trap - ERR
    set +e
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    local cmd_rc=${PIPESTATUS[0]}
    set -e
    if [[ -n "${saved_err_trap}" ]]; then
        eval "${saved_err_trap}"
    fi
    if ((cmd_rc != 0)); then
        log "ERROR: stage ${stage_name} failed with exit ${cmd_rc}: $*"
        exit "${cmd_rc}"
    fi
}

# ── Config override argument parsing ──────────────────────────────────────────

# parse_overrides "$@"
# Standard named flags:
#   --level --version --buffer --weight-method --weight-func
#   --dynamic-buffering --dynamic-buffer-k
# Unknown --flags are rejected with exit 2: silently shifting them away meant a
# typo like `--dynamic-buffer 0.5` (missing the -k) ran a whole job against
# config defaults and filed its output under the wrong directory. Scripts that
# genuinely take extra positional args must pass them after a `--` separator.
parse_overrides() {
    LEVEL=""
    VERSION=""
    BUFFER=""
    WEIGHT_METHOD=""
    WEIGHT_FUNC=""
    DYNAMIC_BUFFERING=""
    DYNAMIC_BUFFER_K=""

    while (($#)); do
        case "$1" in
            --level)
                LEVEL="${2:-}"
                shift 2
                ;;
            --version)
                VERSION="${2:-}"
                shift 2
                ;;
            --buffer)
                BUFFER="${2:-}"
                shift 2
                ;;
            --weight-method)
                WEIGHT_METHOD="${2:-}"
                shift 2
                ;;
            --weight-func)
                WEIGHT_FUNC="${2:-}"
                shift 2
                ;;
            --dynamic-buffering)
                DYNAMIC_BUFFERING="${2:-}"
                shift 2
                ;;
            --dynamic-buffer-k)
                DYNAMIC_BUFFER_K="${2:-}"
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                if [[ "$1" == --* ]]; then
                    log "ERROR: unknown option $1"
                    exit 2
                fi
                shift
                ;;
        esac
    done
}

# build_override_args
#   Build OVERRIDE_ARGS from parsed override vars using standardized named flags.
build_override_args() {
    OVERRIDE_ARGS=()
    if [[ -n "${LEVEL}" ]]; then
        OVERRIDE_ARGS+=("--level" "${LEVEL}")
    fi
    if [[ -n "${VERSION}" ]]; then
        OVERRIDE_ARGS+=("--version" "${VERSION}")
    fi
    if [[ -n "${BUFFER}" ]]; then
        OVERRIDE_ARGS+=("--buffer" "${BUFFER}")
    fi
    if [[ -n "${WEIGHT_METHOD}" ]]; then
        OVERRIDE_ARGS+=("--weight-method" "${WEIGHT_METHOD}")
    fi
    if [[ -n "${WEIGHT_FUNC}" ]]; then
        OVERRIDE_ARGS+=("--weight-func" "${WEIGHT_FUNC}")
    fi
    if [[ -n "${DYNAMIC_BUFFERING}" ]]; then
        OVERRIDE_ARGS+=("--dynamic-buffering" "${DYNAMIC_BUFFERING}")
    fi
    if [[ -n "${DYNAMIC_BUFFER_K}" ]]; then
        OVERRIDE_ARGS+=("--dynamic-buffer-k" "${DYNAMIC_BUFFER_K}")
    fi
}

# ── Package install ────────────────────────────────────────────────────────────

# ensure_src_importable
#   Installs the package in editable mode only when 'import src' would fail, and
#   aborts if it still cannot be imported afterwards. This is the only install
#   entry point: an unconditional `pip install -e` reran on every array task.
ensure_src_importable() {
    if ${PYTHON_CMD} -c "import src" >/dev/null 2>&1; then
        log "src already importable; skipping editable install"
        return 0
    fi
    log "src not importable; installing editable package"
    ${PYTHON_CMD} -m pip install -e "${PROJECT_ROOT}" 2>&1 | tee -a "${LOG_FILE}"
    if ! ${PYTHON_CMD} -c "import src" >/dev/null 2>&1; then
        log "ERROR: src still not importable after editable install (PYTHON_CMD=${PYTHON_CMD})"
        exit 1
    fi
}

# ── Thread environment variables ───────────────────────────────────────────────

# export_thread_vars
#   Propagates SLURM CPU count (or nproc fallback) to numerical-library env vars.
export_thread_vars() {
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 8)}
    export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
    export MKL_NUM_THREADS=$OMP_NUM_THREADS
    export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
}
