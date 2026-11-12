# sensitivity_analysis_scripts

## What This Module Does
The sweep builds service areas and population layers for many settings (basin level, rigid buffer or
size-based radius, weighting) and both approaches. `sweep_validation.py` scores every candidate
against people per plant from real catchments and recommends one.

## How It Fits In
It is an evaluation layer on top of the main pipeline: the same `create_voronoi` and `add_pop` rules,
run for every combination of the `sweep` section (`wwtp.workflow.sweep_variants`, 108 variants), then
one validation job. Approach 0 does not read the basins, so it is built once per weighting and buffer and
hard-linked to the other levels (`wwtp.workflow.approach0_source`); it is scored once.

## Scripts in This Folder
| Script | What it does | Key inputs | Key outputs |
| --- | --- | --- | --- |
| `sweep_validation.py references` | Our plants and their groups (ver / unver / single, on the configured variant's basins); reference units matched to them | configured population layer, reference files (below), GB population raster | `references/plants.csv`, `units.csv`, `matches.csv`, `gb_catchments.gpkg` |
| `sweep_validation.py score` | People per reference unit for every candidate, error measures per reference and group, bootstrap choice | every sweep population layer, the references | `units.parquet`, `scores.csv`, `choice.csv`, `summary.md`, `served_by_country.csv` |

## References and roles
| Key | People per unit | Our plant | Role |
| --- | --- | --- | --- |
| `gb` | population raster in Hoffmann et al.'s catchments (same raster as the service areas: geometry only) | nearest to the catchment's UWWTD plant (1 km), or the only plant in an unlinked catchment | independent, decides |
| `nl` | CBS inhabitants per RWZI, 1 Jan 2024 | `uwwCode` = `NL` + RWZI code | independent, decides |
| `us` | census 2020 people of EPA sourced sewersheds | CWNS ID, one to one, plant at its polygon | calibration (the radius rule was fitted on these) |
| `hw` | HydroWASTE `POP_SERVED`, reported (QUAL_POP 1) | the plant itself | input for candidates with reported sizes, else independent |
| `uwwtd` | UWWTD capacity x `eu_reference_factor` (p.e., not only residents) | nearest UWWTD plant within `threshold` | as `hw` |

Groups (the former verification split, `verification_script.find_verification_watersheds`), fixed once
so that every candidate is scored on the same plants: `ver` = basins with more than one plant, at least
`percent_verification` of them with a detected footprint; `unver` = other basins with more than one
plant; `single` = the only plant of its basin. Measures are given for `all` and each group.

## Measures and the choice
Per candidate, reference and group: class-balanced error (mean over <10k, 10k-100k, >100k people of the
class median |log10 predicted / reference|), typical error, share within a factor 2, median bias, total
ratio, ratio per size class, units without people (counted as infinitely wrong), and for `gb` the
capture (reference people inside the candidate's service area).

Choice: score = mean class-balanced error over `decide_on` (gb, nl). A paired bootstrap over the
reference units gives the shortlist (not worse than the best at `confidence`). The best candidate is
recommended only if it beats the configured one in every `decide_on` reference at `confidence`;
otherwise the configured one stays. `summary.md` lists the top 10, the decision and both candidates'
measures for every reference and group.

## Run Instructions
```bash
snakemake --profile workflow/profiles/slurm sweep          # layers, validation, choice
snakemake --profile workflow/profiles/slurm sweep_layers   # only the service areas and population layers
python -m wwtp.sensitivity_analysis_scripts.sweep_validation references   # by hand
python -m wwtp.sensitivity_analysis_scripts.sweep_validation score
```
Reference files go to `data/validation_data/` as the laptop has them: `gbr_catchments/`
(`catchments_consolidated.zip`, `waterbase_catchment_lookup.csv`, `waterbase_consolidated.csv`),
`nld_cbs_rwzi/` (the CBS xlsx), `usa_epa_sewersheds/Current_Release.zip` (read zipped).

## Parameters (`sweep_validation` in config.yaml)
| Key | Default | Effect |
| --- | --- | --- |
| `reference_years` | gb 2024, nl 2024, us 2020, hw 2024, uwwtd 2024 | `<year>_zonal_sum` compared with each reference; gb also its raster |
| `decide_on` | `['gb', 'nl']` | references of the decision score |
| `bootstrap_samples`, `bootstrap_seed`, `confidence` | 1000, 0, 0.95 | the paired bootstrap |
| `gb_match_distance_m`, `us_match_distance_m` | 1000, 500 | plant matching |
| `threshold`, `eu_reference_factor` | from `eu_comparison` | UWWTD matching and capacity factor |
| `percent_verification` | from `verification_script` | the `ver` group |
| `paths.*` | `data/validation_data/...` | reference files; `output_dir` `data/sensitivity/sweep_validation` |
