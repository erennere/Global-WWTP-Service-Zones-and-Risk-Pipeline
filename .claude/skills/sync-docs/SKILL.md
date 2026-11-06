---
name: Sync Docs
description: Diff current script behavior, CLI flags, and config.yaml keys against every README and diagram in the repo, and update stale documentation to match reality.
disable-model-invocation: true
---

# Sync Docs

Diff the actual current state of the codebase — scripts, CLI flags, `config.yaml` keys — against every README and any diagram/flowchart asset, and bring the documentation back in sync. Meant to be run periodically, and especially right after a refactor lands (e.g. after `standardize-cli`, `find-hardcoded-values`, or `find-duplication` changes have been applied).

## Scope

Check every README under the repo, at minimum:
- Root `README.md`
- `wwtp/README.md`
- One per subsystem directory: `wwtp/annotation_scripts/README.md`, `wwtp/data_merge/README.md`, `wwtp/figures_scripts/README.md`, `wwtp/industrial_analysis/README.md`, `wwtp/pop_at_risk_river_calculations/README.md`, `wwtp/pop_validation_scripts/README.md`, `wwtp/sensitivity_analysis_scripts/README.md`, `wwtp/tests/README.md`

Plus any flowchart/diagram assets referenced from those docs or living alongside them (Mermaid blocks embedded in markdown, `.drawio`/`.svg`/`.dot` files, etc.).

## What to check

- **Scripts documented that no longer exist** in the working tree (stale references — a common failure mode when a script gets renamed/superseded and its README entry isn't updated).
- **Scripts that exist but aren't documented anywhere.**
- **Config-reference tables** whose documented default values have drifted from the actual current values in `wwtp/config.yaml`.
- **CLI flag documentation** that no /longer matches a script's actual `argparse` surface (especially relevant after `standardize-cli` migrations).
- **Flowcharts/diagrams** whose depicted pipeline stages, script names, or data flow no longer match the actual call graph between scripts (trace actual imports/subprocess calls/`.sh` invocation order, don't assume the existing diagram was ever fully accurate).
- **Convoluted text, explanations** that doesn't clearly describe what the code actually does. Subdirectory READMEs should be more than just a copy of the root README; they should describe the subsystem's purpose, its inputs/outputs, and any special usage notes, how they are run and in which order, and how they fit into the overall pipeline, how they relate to other subsystems, and any special caveats or gotchas that a user should be aware of. While being concise is good, being clear and accurate is better. If a README is too terse to be useful, add more detail; if it's too verbose and repetitive, trim it down. After reading the README, a user should be able to answer the following questions:
  - What is this subsystem for?
  - What are its inputs and outputs?
  - How do I run it, and in what order relative to other subsystems?
  - How does it fit into the overall pipeline?
  - Are there any special caveats or gotchas I should be aware of?
- It should be written for humans to read and understand, not just for machines to parse, so make it human-readable and clear even though it might be a bit longer. **Use simple language.**
- **Add a table of contents** for each README and hyperlink each section heading, so that users can quickly jump to the section they need. This is especially important for long READMEs with many sections.
- Root README should have a **high-level overview** of the entire repo, including a diagram of the overall pipeline and how each subsystem fits into it. Each subdirectory README should have a **subsystem-level overview**, including a diagram of the subsystem's internal flow and how it interacts with other subsystems. The Root README should also have a **table of contents** that links to each subdirectory README, so that users can quickly navigate to the subsystem they are interested in. The Root README should have flowcharts/diagrams that show the overall pipeline and how each subsystem fits into it. Each subdirectory README should have flowcharts/diagrams that show the subsystem's internal flow and how it interacts with other subsystems. The flowcharts/diagrams should be clear and easy to understand, with labels for each stage and arrows showing the flow of data between stages. The flowcharts/diagrams should also be updated whenever the code changes, so that they always reflect the current state of the codebase. After having read the root README, a user should be able to answer the following questions:
  - What is the overall purpose of this repo?
  - What are the main subsystems, and how do they fit together?
  - How do I run the entire pipeline, and in what order?
  - Are there any special caveats or gotchas I should be aware of when running the pipeline?
  - How can I find more detailed information about each subsystem if I need it?
  - How can I use this repo to solve my own problem (i.e. how can I use (weighted) voronoi implementation in my own work, say, health or educational infrastructure involving not watersheds but admin level boundaries; can I use the annotation logic in other ways), and what are the limitations of the current implementation?

## Method

1. Use `Explore` agents to build a ground-truth inventory in parallel: actual script list per directory, actual CLI flags per script (via reading `argparse` setup, not docstrings), actual `config.yaml` key/value pairs.
2. Diff that inventory against each README's claims, file by file.
3. For flowcharts, trace the real invocation chain (`.sh` → `python` script → imported modules) rather than trusting the existing diagram's structure as a starting point.

## Decision policy

This is the one skill in this set allowed to apply changes with less friction, since it only touches documentation text (no behavior change) — but still: summarize the full diff (what's stale, what's missing, what changed) to the user before writing any file, so they can veto specific edits (e.g. if a "stale" reference is actually a deliberate placeholder for planned work). Never invent documentation for something you haven't verified actually exists in the code.
