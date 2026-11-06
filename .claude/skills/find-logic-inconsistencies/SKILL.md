---
name: Find Logic Inconsistencies
description: Search wwtp/ for structurally-identical operations handled inconsistently (fail-fast vs fail-soft, silently diverged near-duplicate functions, tautological/inverted booleans) and propose explicit toggles or bug fixes.
disable-model-invocation: true
---

# Find Logic Inconsistencies

Search this repository for cases where the same *kind* of situation is handled differently in different places, with no documented reason — as distinct from an outright bug (which should be flagged separately and more urgently).

## What to look for

- **Fail-fast vs. fail-soft drift**: one function raises immediately on a missing/invalid precondition, another structurally-similar function logs and continues with partial data, for equivalent failure classes with no stated policy governing the choice.
- **Silently-diverged near-duplicates**: two functions that clearly started from the same copy-paste and have since drifted (e.g. one applies a guard/behavior the other doesn't, one hides an empty subplot axis while its sibling doesn't) — where the difference looks like an accident of independent edits rather than an intentional design choice.
- **Tautological or inverted boolean expressions**: conditions that are always true/false regardless of input (`x != 1 or x != 2` is always `True`), or where De Morgan's law was applied backwards. These are usually real bugs, not "inconsistencies" — treat them as high-severity correctness findings, not background inconsistency, and identify the likely originally-intended expression from context (e.g. a sibling line in the same file using the correct `== 1 or == 2` form).
- **Inconsistent observability policy**: some files use `logging`, others bare `print()`, others mix both for equivalent events — not a bug, but worth surfacing if it obscures whether the fail-fast/fail-soft choice above is even visible to an operator. Ideally, there should not be any print() calls in production code, and logging should be used consistently with the same log level for equivalent events.

## Method

1. Use `Explore`/`general-purpose` agents to sweep for these patterns across subsystem directories in parallel — this needs reading actual logic bodies, not just names, so give agents specific instructions to quote the relevant lines rather than paraphrase.
2. For every candidate divergence between two functions, diff them side by side yourself to confirm the divergence is real and characterize it precisely (what exact behavior differs, under what input).
3. Classify each finding as either (a) an intentional-looking divergence that should become an explicit parameter/config toggle, or (b) an outright bug with a clear intended fix — don't blur the two.

## Output format

Per finding:
- **Classification**: bug vs. inconsistency-to-toggle.
- **file:line** for every site involved.
- For bugs: the likely intended correct logic, with the evidence that led you to that conclusion.
- For inconsistencies: the proposed explicit parameter/config name and its per-call-site value, so the difference becomes documented and intentional instead of silent.

## Decision policy

Report findings first, ranked with bugs at the top. Do not edit any file until the user confirms which findings to act on — for suspected bugs, note any downstream consumers of the affected value/column that should be checked before changing behavior that other code may already depend on (even if that dependency is itself wrong).
