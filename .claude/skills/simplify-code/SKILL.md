---
name: Simplify Code
description: Find convoluted code in wwtp/ (oversized functions, deep nesting, fragile closures, dead/superseded logic, encoding corruption) and propose behavior-preserving simplifications.
disable-model-invocation: true
---

# Simplify Code

Search this repository for code that is harder to read and maintain than the problem it solves requires — not style nitpicks, but real convolutedness that makes bugs easy to introduce and hard to spot.

## What to look for

- **Oversized functions** mixing multiple concerns (config parsing, business logic, statistics, I/O) in one body — usually a sign a function should be split along those concern boundaries.
- **Deep nesting** / long branch chains that could be flattened with early returns or extracted helpers.
- **Fragile closures**: a nested function referencing a free variable from the enclosing scope that is only assigned *later* in the enclosing function (works only by accident of Python's late-binding + execution order; breaks silently if the enclosing function is ever reordered).
- **Dual return-type contracts**: a function that returns e.g. `bool` on one call path and a `tuple` on another depending on caller-supplied args, forcing every caller to type-check the result.
- **Encoding corruption**: mojibake baked into docstrings, comments, or printed output (e.g. garbled multi-byte characters) — indicates a broken save/edit at some point that was never cleaned up.
- **Dead or superseded code**: code that is no longer reachable, or whose logic has been fully absorbed by another function elsewhere. Identify this by *tracing actual call sites and comparing logic*, not by naming convention — a function fully superseded by a differently-named newer sibling is just one possible shape this takes. Don't assume something is dead just because it looks legacy or unused at a glance; confirm via grep for all call sites (including dynamic ones — CLI wrappers, config-driven dispatch) before flagging it as removable.
- AI generated code which looks too convoluted to be human-written that does not fit the style of the rest of the repo. Assess whether it is behavior-preserving and whether it can be simplified to match the style of the rest of the repo. Flag it for review and suggest improvements.

## Method

1. Use `Explore` agents to sweep large files/directories in parallel rather than reading everything inline — protect the context window for synthesis.
2. For every candidate, confirm it actually meets the bar above by reading the real code (don't take a subagent's summary at face value for anything you'd propose changing) and cite exact `file:line`.
3. For "dead code" candidates specifically, grep the entire repo (including `.sh` wrappers and `config.yaml` script-entry references) for all call sites before concluding something is unreferenced.

## Output format

For each finding:
- **What's convoluted** and why it's a real risk (not just aesthetics) — the concrete failure scenario if this isn't touched.
- **file:line** evidence.
- **Proposed simplification**, described concretely enough to implement, and whether it's behavior-preserving or needs a test written first to be safe to touch.
- Explicitly flag anything you're *not* confident is safe to simplify blind (e.g. tightly coupled to caller assumptions) rather than proposing a change anyway.

## Decision policy

Report findings first; do not edit any file until the user confirms which findings to act on. Never simplify away actual behavior differences without calling them out — if two "convoluted" branches turn out to encode a real distinction, that's a `find-logic-inconsistencies` finding, not a simplification target.
