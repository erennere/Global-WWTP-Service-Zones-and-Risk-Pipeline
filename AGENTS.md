# AGENTS.md

Purpose: make coding agents immediately productive in this repository while enforcing strict config-driven behavior.

## Scope
- Applies to the whole repository.
- If a change conflicts with this file, pause and ask the user before proceeding.

## Project Orientation
- Root overview and canonical workflow: [README.md](README.md)
- Package-level technical index: [wwtp/README.md](wwtp/README.md)
- Runtime configuration source of truth: [wwtp/config.yaml](wwtp/config.yaml)
- Config resolver/orchestration entry point: [wwtp/starter.py](wwtp/starter.py)
- Shared path/helper resolver utilities: [wwtp/pipelines.py](wwtp/pipelines.py)
- Tests root: [wwtp/tests](wwtp/tests)

## Environment And Commands
- Create env and editable install: see [README.md](README.md)
- Test config: [pytest.ini](pytest.ini)
- Package metadata/dependencies: [pyproject.toml](pyproject.toml)

Use these commands from repository root:
- `python -m pip install -e .`
- `python -m pip install -e ".[test]"`
- `pytest -q`
- `pytest -m unit -q`
- `pre-commit run --all-files` (ruff lint + format, gitleaks, shellcheck, ...)
- `snakemake -n analysis` (what the workflow would run; targets: `analysis`, `figures`, `sweep`, `all`, `fig1`)

## Architecture Boundaries
- Keep `load_config` in [wwtp/starter.py](wwtp/starter.py) orchestration-focused.
- Put defaults and runtime values in [wwtp/config.yaml](wwtp/config.yaml), not hardcoded in processing modules, and declare each new key in its section's model in [wwtp/settings.py](wwtp/settings.py) (type, allowed values).
- Treat stage folders as bounded contexts (`wwtp/data_merge`, `wwtp/annotation_scripts`, `wwtp/pop_at_risk_river_calculations`, `wwtp/industrial_analysis`, etc.).
- Run stages through the Snakemake workflow (`workflow/Snakefile`); a new stage gets a rule there, its files in `wwtp/workflow.py` and its SLURM resources in `config.yaml` → `workflow.resources`. Prefer section-level config inheritance over ad-hoc parameter plumbing.

## Mandatory Coding Policy (User-Enforced)
1. Config-driven only:
- Do not introduce hardcoded runtime values in Python modules when values can live in [wwtp/config.yaml](wwtp/config.yaml).
- Do not add legacy compatibility branches unless explicitly approved by the user.

2. Dynamic fallback access is restricted:
- Do not add `hasattr(...)` checks for normal control flow.
- Do not add `.get(...)` fallbacks for config keys.
- Do not add `getattr(..., default)` fallbacks for config-driven behavior.

3. Exceptions require explicit user confirmation:
- Any new `hasattr`, `.get`, `getattr(..., default)`, or hardcoded literal used as runtime logic requires prior user confirmation in chat.

4. Fail fast for missing config:
- Prefer direct key access for required config values so missing keys fail loudly.

## Implementation Guidance
- Before coding, identify the owning config section in [wwtp/config.yaml](wwtp/config.yaml).
- If a value is currently hardcoded in code, propose/implement moving it to config first.
- Keep changes minimal and localized; avoid incidental refactors.
- Preserve existing path-format conventions (`{data_dir}`, `{version}`, `{level}`, `{weight_type}` placeholders).

## Audit Checklist For PRs
- No new `hasattr` introduced.
- No new `.get` on config dictionaries introduced.
- No new `getattr(..., default)` for config/runtime policy introduced.
- New numeric/string constants are either:
  - configuration-backed, or
  - true constants with documented rationale.
- No new legacy compatibility code unless user-approved.

## Current Audit Artifacts (May 21, 2026)
Generated files:
- [notes/hasattr_issues.txt](notes/hasattr_issues.txt)
- [notes/get_issues.txt](notes/get_issues.txt)
- [notes/getattr_issues.txt](notes/getattr_issues.txt)
- [notes/literal_candidates_raw.txt](notes/literal_candidates_raw.txt)

These are tracking artifacts for cleanup; do not treat existing occurrences as permission to add more.
