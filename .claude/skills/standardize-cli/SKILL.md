---
name: Standardize CLI
description: Audit every script's argument handling against starter.py's standard override mechanism and propose converging all scripts onto one simple CLI pattern, not documenting bespoke variants.
disable-model-invocation: true
---

# Standardize CLI

Audit every executable script under `wwtp/` for how it parses command-line arguments and propagate config overrides, and converge them onto **one** standardized, simple pattern. The goal is convergence, not a catalog of "it works, just differently" — don't let a script keep bespoke CLI logic just because it currently functions.

## The reference standard

Read `wwtp/starter.py` first. It defines the canonical mechanism every script should use:
- `add_standard_override_arguments` — the shared argparse setup for config overrides.
- `_OVERRIDE_FIELDS` / `_OVERRIDE_FLAG_TO_FIELD` — the declared set of overridable fields.
- `_apply_runtime_overrides` — how CLI flags get merged into the resolved config.
- `resolve_config` / `load_config` — the config resolution entry point every script should call.

## What to flag

- Scripts using raw positional `sys.argv[N]` parsing instead of `argparse`.
- Scripts that hand-roll their own override dict/argparse setup instead of calling the shared `starter.py` helpers — especially any that reconstruct only a *subset* of the override fields (the known failure mode in this repo: a per-level loop that manually rebuilds a partial override dict and silently drops fields that were added to `starter.py` later, because the hand-rolled copy was never updated to match).
- Inconsistent flag names, `--help` text, or short/long flag forms for the same semantic override across different scripts.
- Scripts with no CLI at all where one would let the script be run standalone/tested in isolation consistent with its siblings.

## Method

1. Use `Explore` agents to enumerate every script with a `__main__` block or shebang-style entry point across `wwtp/` and its subsystem directories, and report each one's current argument-handling approach with `file:line`.
2. Cross-reference against `starter.py`'s current `_OVERRIDE_FIELDS` list to check every hand-rolled override dict is complete relative to it.
3. Group findings by "already standard" vs. "needs migration," and for the latter, describe the target CLI surface using the same `add_standard_override_arguments`/`resolve_config` shape used by an already-compliant sibling script.

## Output format

- **Per-script table**: script path, current approach (argparse-standard / argparse-bespoke / positional sys.argv / none), deviation description, proposed target shape.
- Call out any override field silently dropped by a hand-rolled dict, separately and prominently — this is a live correctness bug, not just a style issue.

## Decision policy

Report the full audit and the single converged CLI shape you're proposing before migrating any script. Migrate one script at a time once approved, and confirm each migration preserves the script's existing observable behavior (same flags still work, same config keys still resolve) unless the user asked for an actual behavior change.
