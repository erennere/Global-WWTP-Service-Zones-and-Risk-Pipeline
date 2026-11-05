#!/bin/bash
# Capture what every shell wrapper would invoke, without running any Python.
#
# The interpreter is replaced by a stub that records its argv and exits 0, so a
# run produces one snapshot file per wrapper containing the exact command lines
# the wrapper would have executed, plus its exit code and normalized log output.
#
# Usage (always from src/, like every other wrapper):
#   bash tests/harness/wrapper_argv_snapshot.sh <output-dir>
#
# Compare a pre-change and a post-change capture with:
#   diff -r <before-dir> <after-dir>
#
# An empty diff means the migration preserved wrapper behavior.

set -Eeuo pipefail

OUT_DIR="${1:?usage: wrapper_argv_snapshot.sh <output-dir>}"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

> "${OUT_DIR}/.keep"

# The stub records argv for every call. Module runs (`-m pkg.mod`, `-m pip`) are
# the pipeline work and are never executed. Anything else - the inline `-c` and
# heredoc config reads some wrappers depend on to pick an execution mode - is
# delegated to the real interpreter so those wrappers reach their real branches.
REAL_PYTHON="${REAL_PYTHON:-$(command -v python3)}"
# Delegated `-c` calls import pipeline modules; without this they drop
# __pycache__ beside the sources, which edits a read-only baseline copy.
export PYTHONDONTWRITEBYTECODE=1
STUB="${OUT_DIR}/fakepython"
cat > "${STUB}" <<STUB_EOF
#!/bin/sh
{
    printf 'CALL\n'
    for arg in "\$@"; do printf '%s\n' "\$arg"; done
} >> "\${ARGV_LOG}"
if [ "\$1" = "-m" ]; then
    exit 0
fi
exec "${REAL_PYTHON}" "\$@"
STUB_EOF
chmod +x "${STUB}"

# Standard override set used for every wrapper that accepts the seven flags.
STD_OVERRIDES=(--level 8 --version 2 --buffer 9000 --weight-method linear
               --weight-func mult --dynamic-buffering false --dynamic-buffer-k 0.7)

# Wrapper-specific argument sets. Anything not listed here gets STD_OVERRIDES.
# The listed wrappers accept a narrower flag set and reject the full seven.
wrapper_args() {
    case "$1" in
        ./add_pop.sh)
            printf '%s\n' --index 3 "${STD_OVERRIDES[@]}" ;;
        ./sensitivity_analysis_scripts/sweep_ver_ranking.sh)
            printf '%s\n' --merge ;;
        ./sensitivity_analysis_scripts/industrial_analysis_sweep.sh)
            printf '%s\n' --version 2 ;;
        ./sensitivity_analysis_scripts/create_voronoi_param_sweep_parallel.sh|\
        ./sensitivity_analysis_scripts/create_voronoi_param_sweep_parallel_additive.sh)
            printf '%s\n' --approach 1 --version 2 --dynamic-buffering false --dynamic-buffer-k 0.7 --num-jobs 2 ;;
        ./sensitivity_analysis_scripts/create_voronoi_single_additive_test.sh)
            printf '%s\n' --approach 1 --level 8 --version 2 --buffer 9000 \
                          --weight-method linear --dynamic-buffering false --dynamic-buffer-k 0.7 ;;
        *)
            printf '%s\n' "${STD_OVERRIDES[@]}" ;;
    esac
}

# Strip everything that legitimately varies between two runs.
normalize() {
    sed -E \
        -e 's/\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\]/[TS]/g' \
        -e "s#${OUT_DIR}#<OUTDIR>#g" \
        -e 's#/tmp/[A-Za-z0-9._-]+#<TMP>#g'
}

# A frozen baseline copy kept beside the live tree must not be swept as if it
# were part of it - its wrappers are captured by pointing this harness at a
# scratch copy of that tree instead (never at the tree itself: init_log does
# `rm -f` on ${PROJECT_ROOT}/logs, which would edit a read-only baseline).
mapfile -t WRAPPERS < <(find . -name '*.sh' -not -path './.venv/*' -not -path './lib/*' \
                             -not -path './tests/*' \
                             -not -path '*old-version*' -not -path '*baseline*' | sort)

echo "Capturing ${#WRAPPERS[@]} wrappers into ${OUT_DIR}"

for wrapper in "${WRAPPERS[@]}"; do
    name="$(echo "${wrapper#./}" | tr '/' '_')"
    argv_log="${OUT_DIR}/${name}.argv"
    : > "${argv_log}"

    mapfile -t args < <(wrapper_args "${wrapper}")

    set +e
    output="$(PYTHON_CMD="${STUB}" ARGV_LOG="${argv_log}" \
              SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 SLURM_CPUS_PER_TASK=4 \
              bash "${wrapper}" "${args[@]}" 2>&1)"
    rc=$?
    set -e

    {
        printf '=== wrapper: %s\n' "${wrapper}"
        printf '=== args: %s\n' "${args[*]}"
        printf '=== exit: %s\n' "${rc}"
        printf '=== invocations:\n'
        normalize < "${argv_log}"
        printf '=== output:\n'
        printf '%s\n' "${output}" | normalize
    } > "${OUT_DIR}/${name}.snap"

    rm -f "${argv_log}"
    printf '  %-70s exit=%s\n' "${wrapper}" "${rc}"
done

rm -f "${STUB}"
echo "Done. Snapshots in ${OUT_DIR}"
