---
name: Enforce Fail-Fast
description: Aggressively find and remove defensive fallbacks, legacy shims, and backward-compatibility paths across wwtp/, converting them to loud immediate failures, unless explicitly allowed to remain.
disable-model-invocation: true
---

# Enforce Fail-Fast

General principle for this repo: **no defensive fallback, no legacy shim, no backward-compatibility path, no silent "make it work" patch — anywhere — unless the user has explicitly said to keep it.** When something required is missing or invalid, the code should fail loudly and immediately at the point closest to the actual missing precondition, not degrade silently or paper over it deeper in the call stack.

## What to find and convert to loud failures

- Bare `except: pass` / `except Exception: pass` (or any except-block that swallows an error without re-raising or aborting).
- `hasattr()`-gated no-ops — code that silently does nothing when an expected attribute/key isn't present, instead of raising.
- `.get(key, default)` reads for config keys that are not declared anywhere in `config.yaml` — dead defensive code for config surface that was never actually wired up.
- Pipeline steps that are commented out (or otherwise disabled) but still logged/reported as having succeeded — e.g. a wrapper script that logs a step name and an unconditional "completed successfully" message regardless of whether that step's actual invocation is still present.
- **Backward-compatibility re-exports, shims, or dual-behavior branches kept only to avoid breaking old callers.** Flag these for removal by default — not only the silent/hidden ones. If a function has two code paths because of an old calling convention that's no longer used anywhere, the old path should go, not be preserved "just in case."

## Explicit allowlist — do not flag these

- (The `try: from .module import x` / `except ImportError` dual-import pattern was removed in 167e036; stages run only as `python -m wwtp.<stage>`. Flag any new instance.)

When you find a pattern that resembles the allowlisted one but differs in a material way, describe the difference explicitly rather than silently including or excluding it.

## Method

1. Use `Explore`/`general-purpose` agents to grep across `wwtp/` for the patterns above in parallel, per subsystem directory.
2. For every candidate, read the surrounding code to confirm it's genuinely a fallback/shim and not something structurally load-bearing (like the allowlisted import pattern) before flagging it.
3. For each flagged fallback, identify what the loud-failure replacement should look like (what exception, what message, at what point in the call chain it should surface — as close to the actual missing precondition as possible).

## Output format

Per finding: `file:line`, current fallback/shim behavior, why it's silently masking a real problem (the concrete scenario where this currently hides an error instead of surfacing it), and the proposed fail-fast replacement.

## Decision policy

Report all findings first, ranked by how much real failure they could currently be masking. Do not edit any file until the user confirms which ones to convert. When you're genuinely unsure whether something is legitimate (load-bearing, like the allowlisted pattern) or removable-by-default legacy code, say so and ask — don't assume either way.
