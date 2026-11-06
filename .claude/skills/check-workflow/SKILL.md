---
name: Check Workflow
description: Check that workflow/Snakefile, wwtp/workflow.py and the stage command lines agree, that every stage has SLURM resources, and that the remaining shell scripts pass shellcheck.
disable-model-invocation: true
---

# Check Workflow

The pipeline runs through Snakemake (`workflow/Snakefile`). Each rule calls `python -m wwtp.<module>`
with arguments it builds from `wwtp/workflow.py`, and declares input and output files that
`wwtp/workflow.py` computes with the stages' own config resolution and path helpers. Drift between
the three is the failure this checklist looks for.

## Checks

- **Rule vs. stage CLI**: every `params: module=..., cli=[...]` passes only flags the module's argparse
  accepts (`--approach`, `--job-index`, `--merge-shards`, `--max-workers`, the seven run overrides, ...).
  `python -m wwtp.<module> --help` lists them.
- **Declared vs. written files**: for each rule, the outputs are exactly what the stage writes for that
  variant (run the mock-world workflow test, `wwtp/tests/integration/test_workflow_mock.py`, and compare
  `{data_dir}` with the outputs Snakemake lists). A stage that writes a file no rule declares, or a rule
  that declares a file nobody writes, is a finding. Stamps (`{data_dir}/workflow/stamps`) are only for
  stages whose outputs cannot be listed up front (downloads, plot sets).
- **Shared files**: two rules (or two variants of one rule) must never write the same path, except through
  `utils.atomic_output` for a cache whose content is identical for all writers (e.g. `watershed_with_countries`).
- **Late binding**: a value from a `for` loop must reach a rule through `params`, `input` or `output`,
  never through the `shell:` string (it is formatted when the job runs, when the loop variable already
  holds its last value).
- **Resources**: every stage a rule runs has an entry in `wwtp/config.yaml` → `workflow.resources`
  (`wwtp/tests/unit/test_workflow.py` checks this).
- **Shell scripts**: the remaining `.sh` files (`figures/fig1/*/render_*.sh`) pass `shellcheck` (the
  pre-commit hook) and read their settings through `figures/fig1/fig1_config.py`.

## Output

A table: rule or file, what disagrees, which side is right (the stage code is the reference for what is
written; config.yaml for values), and the fix.
