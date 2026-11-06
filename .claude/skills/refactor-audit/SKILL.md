---
name: Refactor Audit
description: Run the full repo-quality checklist (duplication, convoluted code, CLI consistency, hardcoded values, logic inconsistencies, doc drift, fail-fast violations, workflow consistency) as one coordinated sweep and produce a single prioritized report.
disable-model-invocation: false
---

# Refactor Audit

The comprehensive, periodic pass over the whole repository — the "run this before starting a big feature, or every few months" entry point. Runs the checklists from the other 9 project skills as one coordinated sweep, then merges and deduplicates overlapping findings into a single prioritized report, mirroring the shape of a full repo assessment.

## Method

1. Invoke each of the other 8 skills' checklists rather than re-deriving them inline — either via the `Skill` tool (`find-duplication`, `simplify-code`, `standardize-cli`, `find-hardcoded-values`, `find-logic-inconsistencies`, `sync-docs`, `enforce-fail-fast`, `check-workflow`), or by spawning one `Agent` per area in parallel carrying that skill's checklist verbatim as its brief, whichever fits the available tooling better at the time.
2. Since the same file frequently shows up under more than one checklist (e.g. a hardcoded value that's also part of a duplicated block, or a fallback pattern that's also a logic inconsistency), merge findings across areas by file/location rather than reporting the same spot multiple times under multiple headings.
3. Rank the merged list by real-world risk/impact, not by which checklist happened to surface it first — an actual correctness bug (tautological condition, silently-dropped CLI override, a step that logs success without running) outranks a stylistic duplication cluster, which in turn outranks a forward-looking architectural opportunity.

## Output format

A single report structured like:
1. **Correctness bugs** (from `find-logic-inconsistencies`, `enforce-fail-fast`) — highest priority, these are wrong today regardless of any refactor.
2. **Consistency/maintainability risks** (duplication clusters, CLI divergence, Snakefile/stage drift) — ranked by how many sites/how much drift.
3. **Hardcoded-value / config-surface gaps.**
4. **Documentation drift** (no runtime behavior risk, but still worth listing).

For every item, cite `file:line`, which underlying checklist it came from, and the proposed fix — don't just link out to "run skill X for details," surface the actual finding here.

## Decision policy

This is a reporting pass across all 8 sub-checklists — do not edit any file as part of running this skill. Once the user has reviewed the merged report and picked which items to act on, invoke the specific relevant skill (or make the edit directly, if trivial and already fully approved) rather than applying everything at once.
