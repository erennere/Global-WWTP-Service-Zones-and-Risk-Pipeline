# AGENTS.md

Purpose: make coding agents immediately productive in this repository while enforcing strict config-driven behavior.

## Scope
- Applies to the whole repository.
- If a change conflicts with this file, pause and ask the user before proceeding.

## Project Orientation
- Root overview and canonical workflow: [README.md](README.md)
- Package-level technical index: [src/README.md](src/README.md)
- Runtime configuration source of truth: [src/config.yaml](src/config.yaml)
- Config resolver/orchestration entry point: [src/starter.py](src/starter.py)
- Shared path/helper resolver utilities: [src/pipelines.py](src/pipelines.py)
- Tests root: [src/tests](src/tests)

## Environment And Commands
- Create env and editable install: see [README.md](README.md)
- Test config: [pytest.ini](pytest.ini)
- Package metadata/dependencies: [src/pyproject.toml](src/pyproject.toml)

Use these commands from repository root:
- `python -m pip install -e ./src`
- `python -m pip install -e "./src[test]"`
- `pytest -q`
- `pytest -m unit -q`

## Architecture Boundaries
- Keep `load_config` in [src/starter.py](src/starter.py) orchestration-focused.
- Put defaults and runtime values in [src/config.yaml](src/config.yaml), not hardcoded in processing modules.
- Treat stage folders as bounded contexts (`src/data_merge`, `src/annotation_scripts`, `src/pop_at_risk_river_calculations`, `src/industrial_analysis`, etc.).
- Prefer existing wrappers and section-level config inheritance over ad-hoc parameter plumbing.

## Mandatory Coding Policy (User-Enforced)
1. Config-driven only:
- Do not introduce hardcoded runtime values in Python modules when values can live in [src/config.yaml](src/config.yaml).
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
- Before coding, identify the owning config section in [src/config.yaml](src/config.yaml).
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
