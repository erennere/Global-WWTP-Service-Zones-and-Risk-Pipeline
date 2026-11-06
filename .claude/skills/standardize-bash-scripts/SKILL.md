---
name: Standardize Bash Scripts
description: Converge all .sh wrappers under wwtp/ onto one shared shape that works identically under sbatch and plain bash when launched from wwtp/, deduplicating shared logic into wwtp/lib/utils.sh.
disable-model-invocation: true
---

# Standardize Bash Scripts

Audit and converge every `.sh` wrapper under `wwtp/` onto one shared shape that behaves identically whether launched via `sbatch` on HPC or plain `bash` from any directory.

## Scope

Every `.sh` file under `wwtp/`, including but not limited to: `add_pop.sh`, `create_voronoi.sh`, `combine_watersheds.sh`, `download_pop.sh`, and one or more per subsystem directory (`annotation_scripts/`, `data_merge/`, `figures_scripts/`, `industrial_analysis/`, `pop_at_risk_river_calculations/`, `pop_validation_scripts/`, `sensitivity_analysis_scripts/`). Check `wwtp/lib/utils.sh` first — this is the existing shared-bash-helpers location; converge onto and extend it rather than proposing a new shared file.

## The standard shape

- `set -Eeuo pipefail` at the top of every script (`-E` so the shared `ERR` trap fires inside functions).
- A SLURM `#SBATCH ...` header block that is inert/ignored when the script is run as plain `bash script.sh`, and honored correctly when submitted via `sbatch`.
- Consistent logging/echo style across scripts (pick one convention and apply it everywhere).
- Executable bit set (`+x`) on every `.sh` file — flag any missing it; this is a known recurring issue (a filesystem/checkout artifact, not a logic problem, but it silently breaks direct invocation).

## Every wrapper is launched from `wwtp/`

This repo has one invariant, and it is deliberate: **every `.sh` file is run from `wwtp/` and nowhere else**, for both `bash` and `sbatch`:

```bash
cd wwtp
bash pop_at_risk_river_calculations/create_rasters.sh --level 8
sbatch pop_at_risk_river_calculations/create_rasters.sh --level 8
```

Because of that, `PROJECT_ROOT="."` is the standard and correct form, and every script resolves `lib/utils.sh`, `config.yaml`, and sibling paths relative to it. Do **not** migrate scripts to `${BASH_SOURCE[0]}`-based resolution: it adds indirection for a launch scenario this project does not support, and it makes the deviation table noisier without making any supported invocation work better.

What to enforce instead:

- `PROJECT_ROOT="."` followed by `source "${PROJECT_ROOT}/lib/utils.sh"`, identically in every script.
- A guard that fails loudly with a usage hint when `lib/utils.sh` is not found — that is the signal the operator launched from the wrong directory. `pop_at_risk_river_calculations/pop_differences_and_impact_polygons.sh` has the reference form.
- The invariant restated in each script's usage comment, so the launch directory is never a guess.

## Deduplicate bash logic itself

Apply the same centralization-by-default principle used for Python (`find-duplication`) to bash: any logic repeated across multiple `.sh` scripts — argument validation, logging helpers, environment/conda activation, override-flag assembly, per-stage failure handling — should be extracted into `wwtp/lib/utils.sh` (sourced by each script) rather than repeated per-script. Check what's already in `utils.sh` before proposing anything new; extend it, don't duplicate it.

## Method

1. Read `wwtp/lib/utils.sh` in full first, so you know what's already shared before flagging anything as "needs deduplicating."
2. Use `Explore` agents to inventory every `.sh` script's current header (shebang, `set` flags, SBATCH block, `PROJECT_ROOT` form and utils guard, executable bit) in parallel.
3. Diff each script against the proposed standard shape and against `utils.sh`'s current contents.

## Output format

- The proposed standard header/shape, written out once in full.
- A per-script table: current state, specific deviations, what needs to move into `utils.sh` (if anything is duplicated elsewhere), missing `+x` flag or not.

## Decision policy

Report the standard shape and the full deviation table before editing any script. Migrate scripts one at a time once approved, and confirm each migrated script's actual invoked Python entry point / arguments are unchanged unless a behavior change was explicitly requested.
