---
name: Find Duplication
description: Scan wwtp/ for duplicated or near-duplicate logic (functions, algorithms, boilerplate blocks) and propose consolidation into a shared helper module, defaulting to centralization over new locations.
disable-model-invocation: true
---

# Find Duplication

Search this repository for duplicated or near-duplicate logic — not just copy-pasted text, but structurally equivalent operations reimplemented independently (same intent, different code shape, possibly already drifted in behavior).

## Method

1. Delegate the scan to parallel `Explore` (or `general-purpose`) agents, one per subsystem directory, so no single context window has to hold the whole tree at once. Cover at minimum: `wwtp/` top-level scripts, `wwtp/data_merge/`, `wwtp/pop_at_risk_river_calculations/`, `wwtp/figures_scripts/`, `wwtp/sensitivity_analysis_scripts/`, `wwtp/pop_validation_scripts/`, `wwtp/industrial_analysis/`, `wwtp/annotation_scripts/`.
2. Ask each agent to report candidate clusters with concrete `file:line` evidence, not vague "these seem similar" claims — it must quote or paraphrase the actual overlapping logic.
3. Merge the per-directory reports yourself and de-duplicate/cluster overlapping findings (the same helper often gets flagged from both call sites).
4. For each cluster, check whether the operation is spatial/geometric (candidate: `wwtp/geo_utils.py`) or domain-agnostic (candidate: `wwtp/utils.py`) — these are the existing canonical shared modules from prior consolidation work in this repo. Read them first so you don't propose reinventing something that already exists there.

## Centralization is the default

Prefer extending `wwtp/geo_utils.py` or `wwtp/utils.py` over creating a new shared module. A **new** location (e.g. a sibling `<subdir>/utils.py`) is only acceptable when the cluster is genuinely specific to one subsystem's own concerns — never propose a new location just to avoid touching an existing file. If you're unsure whether a cluster belongs in an existing module or needs its own, say so explicitly and let the user decide rather than picking silently. You can suggest a new location under wwtp/*.py if the functionality cannot be meaningfully put under `wwtp/geo_utils.py` or `wwtp/utils.py`, but do not create a new module without user confirmation.

## What counts as a duplication cluster

- The same algorithm implemented with different code (e.g. two different point-clustering implementations that produce different results on the same input).
- The same scaffolding pattern repeated verbatim or near-verbatim (e.g. repeated DuckDB spatial-extension setup + WKT round-trip blocks).
- The same parameter grid / constant table copy-pasted across multiple files.
- Two functions that started as one copy-paste and have since silently diverged (different default, one has a guard the other lacks) — flag the divergence explicitly as a bug risk, not just as duplication.

## Output format

For each cluster:
- **Cluster name** and severity (how many sites, how much drift risk).
- **Sites**: `file:line` for every occurrence.
- **Proposed shared signature**: the function/class signature you'd extract, and its target module.
- **Behavioral differences across sites** (if any) that the consolidated version needs to support as parameters rather than silently picking one behavior.

## Decision policy

Do not extract or edit any file. Present the full cluster report and let the user pick which clusters to consolidate and confirm the target module before you touch any code.
