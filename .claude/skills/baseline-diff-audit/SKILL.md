---
name: Baseline Diff Audit
description: Compare the working tree against a frozen baseline copy of wwtp/ (or a git ref) surface by surface, classify every behavioral divergence as intended / intended-undocumented / regression, and prove the scripts still run.
disable-model-invocation: true
---

# Baseline Diff Audit

The "did we break anything since the last known-good state?" pass. Runs after a large refactor, when
`pytest` passing proves little because the tests were written against the *new* code. Answers one
question: **does the pipeline still do what it did, and where it doesn't, was that on purpose?**

## Finding the baseline

A directory under `wwtp/` matching `*old*`, `*baseline*` or `*DO-NOT-CHANGE*`. If none exists, make
one with `git worktree add <scratch> <ref>`. Confirm what it actually is before trusting it — check
whether it matches any commit (`git show <c>:<path> | diff - <baseline>/<path>`); a hand-frozen copy
is usually an *older lineage*, not last week's HEAD, and the gap is wider than you'd assume.

**The baseline is read-only.** Never write, `chmod`, or `git add` it. Never import it as a package —
load modules by file path under a private alias, because the live package is installed and
`import <name>` silently resolves to the wrong tree. Never run anything inside it (old shell wrappers
deleted logs, Snakemake writes `.snakemake/`); copy it to scratch and run there.

Read-only includes **bytecode**: loading a baseline module writes `__pycache__` beside its source.
Set `sys.dont_write_bytecode = True` before the first baseline import and export
`PYTHONDONTWRITEBYTECODE=1` for anything that shells out. Verify by counting files in the baseline
before and after a run — the count must be identical.

## Method

**Do not `diff -r` the trees.** Past a few thousand changed lines that output is unreadable and
proves nothing. Compare surfaces, each small enough to read in full. Layers 1–3 are cheap and always
run; layer 4 is the expensive one and is scoped to the consolidations the change set actually
touched; layer 5 is not a comparison at all.

1. **Static surfaces** — `wwtp/tests/harness/baseline_diff.py <baseline>`. Per-module top-level API
   with signatures, config sections and dotted leaf keys (both directions *and* values changed in
   place), module-level constants, and (for baselines that still have them) per-wrapper `set` line / parsed
   flags. Crucially, a name missing from its old module is matched against every new module before
   being called a deletion — **moved is not deleted**, and that distinction is most of the value.
2. **Stage commands** — `snakemake -n -p` in each tree (old trees: the wrappers' `-m <module> …`
   lines) prints every job's command; diff them. This is what proves a stage is still invoked with
   the same arguments and overrides. Lay the baseline out as `<scratch>/wwtp` (`<scratch>/src` for a baseline from before the rename) and run with `PYTHONPATH=<scratch>`
   (see Traps).
3. **Resolved config** — `wwtp/tests/harness/baseline_config_diff.py <baseline>`. Compares config as
   `resolve_config` delivers it, after null-inheritance and `{placeholder}` interpolation. A key can
   be textually unchanged and still resolve differently because a section it inherits from moved;
   conversely, a large key diff with zero resolved-value changes is proof the moves were safe.
4. **Differential tests** — `wwtp/tests/harness/baseline_differential.py <baseline>`. Load both
   implementations, feed identical seeded inputs, compare. The only layer that catches a consolidated
   helper computing a different *number*. Each case declares `expect="same"` or `expect="differ"`, so
   the report flags both a regression **and** a fix that silently did not land.
5. **Runnability** — `snakemake -n` on the mock world; `compileall`; import every module; **`--help` must exit
   0 for every `__main__` module**; `pyflakes` for undefined names; `pytest` from the repo root,
   diffed against the recorded failure list.

## Output format

`notes/BASELINE_DIFF_AUDIT.md`. A verdict table by layer, then every divergence in one of three
buckets:

- **intended** — cite the approving decision. Where the real behavior turned out worse than the
  decision described it, say so; the write-up is now the record.
- **intended-undocumented** — a genuine decision nobody wrote down. Write it down here, and say
  plainly if it went beyond the approved scope.
- **regression** — fix it, then record the fix.

State the harness's own limitations at the end. A clean report from a harness with a blind spot is
worse than a noisy one.

## Decision policy

Fix regressions found during the audit; that is the point of running it. Do not "tidy" anything else
you notice — a baseline audit that also refactors cannot prove what it changed.

**An "expected to differ" case is demonstrated, never asserted.** Record the baseline output and the
current output side by side. Two traps this catches: a case that passes because the current call
raised `TypeError` from a wrong fixture signature, and a case predicted to differ that is actually
`same` because some earlier guard already rejected the input. Both happened; both looked green.

When transcribing baseline logic into a fixture, copy it verbatim with a comment naming the
`file:line` it came from, rather than paraphrasing what you believe it did.

## Traps

- **Baseline imports resolving to the live package.** The single most dangerous failure mode: the
  baseline appears to fail with the *current* code's errors, inventing regressions. Lay it out as
  `<scratch>/wwtp` (or `src`), set `PYTHONPATH=<scratch>`, and sanity-check that a symbol you know differs
  between the trees reports the baseline's value.
- **A stubbed interpreter changes exit codes.** Stubbing `pip` makes a wrapper that unconditionally
  reinstalls fail its follow-up import check. Classify by *cause*, not by exit code.
- **`pytest` from the repo root**, never from `wwtp/` — `testpaths` keeps the baseline's own `tests/`
  out. Run from `wwtp/` and it gets collected, producing hundreds of spurious errors.
- **`hasattr` is not a safe probe.** It swallows only `AttributeError`; a property raising anything
  else propagates, and under a blanket `except Exception` becomes a silent `False`.
- **Dual-import blocks** must stay symmetric. A name added to one branch and not the other fails
  only on the path that branch serves — the `--help`-exits-0 sweep and `pyflakes` are what catch it.
