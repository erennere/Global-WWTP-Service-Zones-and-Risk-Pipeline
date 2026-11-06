---
name: Find Hardcoded Values
description: Search wwtp/ for hardcoded literals and Python-level defaults that represent tunable pipeline behavior, and propose wiring them into config.yaml via starter.py's required-key resolution — no fallback defaults.
disable-model-invocation: true
---

# Find Hardcoded Values

Search this repository for hardcoded literals, magic numbers, and Python-level default kwargs that represent tunable pipeline behavior (thresholds, worker/process counts, sigma/buffer/level defaults, output filenames/paths, feature toggles) which should instead live in `wwtp/config.yaml` and be propagated through `wwtp/starter.py`.

## Method

1. Use `Explore` agents to grep across `wwtp/` (top-level scripts and every subsystem directory) for numeric literals, default kwargs (`def f(x, threshold=500, workers=12, ...)`), and hardcoded string paths/filenames used inside function bodies rather than passed in from config.
2. For each candidate, check `wwtp/config.yaml` for an existing key that should already cover it (just not threaded through) versus a genuinely missing key.
3. Read `wwtp/starter.py`'s `resolve_config` to understand the file-order/null-inheritance rule: a config section can only inherit a null value from a section defined **earlier** in `config.yaml`, never a later one. Use this to determine which section a new key should be declared on first (if it's meant to be shared/inherited) versus where it's only ever overridden.
4. Also check the inverse direction: keys declared in `config.yaml` that no script actually reads, and any `cfg['x'] if 'x' in cfg else <default>` guard for a key that isn't declared anywhere in `config.yaml` — this is dead defensive code for config surface that was never wired up, and should be flagged for both `find-hardcoded-values` and `enforce-fail-fast`.
5. check for similar keys in config.yaml and propose a single canonical key if the same value is being wired through multiple keys (e.g. `cfg['x']` vs `cfg['x_override']` vs `cfg['x_tuning']` all wired to the same hardcoded default) or if they are used in similar manners.
6. Check for any hardcoded credentials or API keys (e.g. a committed API key) — these are a separate class of finding and should be flagged as "move to environment variable / secret store," never as a proposed `config.yaml` entry.
7. Merge the per-directory reports yourself and de-duplicate/cluster overlapping findings (the same hardcoded value often gets flagged from multiple call sites).

## No fallback defaults, ever

When proposing to wire a hardcoded value into `config.yaml`:
- The new key must be **required** and resolved strictly through `resolve_config`'s inheritance chain — no Python-level fallback default (`kwargs=None; if x is None: x = 500`), no `.get(key, default)`, no legacy shim kept "just in case" to avoid breaking old call sites.
- This is unconditional unless the user explicitly asks for a specific key to remain optional. Don't assume optionality to be safe — ask instead.
- If a value is *not* actually meant to vary per environment/run (e.g. a mathematical constant, not a tuning knob), say so and don't force it into config just for the sake of it — only flag genuinely tunable values.

## Secrets are not config values

Hardcoded credentials or API keys (e.g. a committed API key) are a separate class of finding — flag them as "move to environment variable / secret store," never as a proposed `config.yaml` entry, and call out the exposure explicitly regardless of what else is found.

## Output format

Per finding: `file:line`, the current hardcoded value/default, the proposed `config.yaml` key + section (with the null-inheritance rule reasoning for why that section), and confirmation of whether the key already exists elsewhere and is just unwired.

## Decision policy

Report all findings first. Don't edit `config.yaml` or any script until the user confirms which values to wire through and which section should own each new key.
