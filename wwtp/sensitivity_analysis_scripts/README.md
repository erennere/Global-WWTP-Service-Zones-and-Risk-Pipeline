# sensitivity_analysis_scripts

## What This Module Does
The sweep builds service areas and population layers for many settings (basin level, rigid buffer or
size-based radius, weighting) and both approaches. `sweep_validation.py` scores every candidate at two
levels, plants and countries, and recommends one.

## How It Fits In
It is an evaluation layer on top of the main pipeline: the same `create_voronoi` and `add_pop` rules,
run for every combination of the `sweep` section (`wwtp.workflow.sweep_variants`, 108 variants), then
one validation job. Approach 0 does not read the basins, so it is built once per weighting and buffer and
hard-linked to the other levels (`wwtp.workflow.approach0_source`); it is scored once.

## Scripts in This Folder
| Script | What it does | Key inputs | Key outputs |
| --- | --- | --- | --- |
| `sweep_validation.py references` | Our plants and their groups (ver / unver / single, on the configured variant's basins); reference units matched to them; people per country (raster) and the national references | configured population layer, reference files (below), population rasters | `references/plants.csv`, `units.csv`, `matches.csv`, `gb_catchments.gpkg`, `national.csv` |
| `sweep_validation.py score` | People per reference unit and per country for every candidate; measures per reference, group, country and SDG region; bootstrap choice | every sweep population layer, the references | `units.parquet`, `scores.csv`, `national_countries.csv`, `national_regions.csv`, `national_scores.csv`, `choice.csv`, `summary.md` |

## Plants: references and roles
| Key | People per unit | Our plant | Role |
| --- | --- | --- | --- |
| `gb` | population raster in Hoffmann et al.'s catchments (same raster as the service areas: geometry only) | nearest to the catchment's UWWTD plant (1 km), or the only plant in an unlinked catchment | independent |
| `nl` | CBS inhabitants per RWZI, 1 Jan 2024 | `uwwCode` = `NL` + RWZI code | independent |
| `us` | census 2020 people of EPA sourced sewersheds | CWNS ID, one to one, plant at its polygon | calibration (the radius rule was fitted on these) |
| `hw` | HydroWASTE `POP_SERVED`, reported (QUAL_POP 1) | the plant itself | input for candidates with reported sizes, else independent |
| `uwwtd` | UWWTD capacity x `eu_reference_factor` (p.e., not only residents) | nearest UWWTD plant within `threshold` | as `hw` |

Groups (the former verification split, `verification_script.find_verification_watersheds`), fixed once
so that every candidate is scored on the same plants: `ver` = basins with more than one plant, at least
`percent_verification` of them with a detected footprint; `unver` = other basins with more than one
plant; `single` = the only plant of its basin. Measures for `all` and each group: class-balanced error
(mean over <10k, 10k-100k, >100k people of the class median |log10 predicted / reference|), typical
error, share within a factor 2, median bias, total ratio, ratio per size class, units without people
(counted as infinitely wrong), and for `gb` the capture (reference people inside the service area).

## Countries and SDG regions
Every country with at least `national_min_plants` plants: people in the candidate's service areas /
people of the country's population raster (`national_year`, 2022), against the share connected to
treatment (value of the year nearest 2022, at most `national_max_year_gap` years off):

| Key | Share | Role |
| --- | --- | --- |
| `jmp` | JMP "safely managed via sewer" (sewered and treated), 160+ countries | independent, global |
| `jmp_sewer` | JMP sewer connection, treated or not | upper bound |
| `eurostat` | Eurostat `env_ww_con`, urban and other wastewater treatment | independent (Europe) |
| `oecd` | OECD public total treatment | independent (OECD) |

Measures: mean, median and population-weighted absolute error (pp), bias, share within 10 pp, the
SDG-region error (population-weighted regional shares; JMP's `region_sdg`) and the world ratio. Part of
the gap is common to every candidate (plants missing from our list); the choice uses the differences.

## The choice
Components and weights: `decision_weights` (default GB 0.25, NL 0.25, JMP 0.5: half plants, half
countries). Each component's error (plants: class-balanced; countries: mean absolute error) relative to
the configured candidate's, weighted; configured = 1. A paired bootstrap resamples the units of every
component. The best candidate is recommended only if its score is below 1 at `confidence` **and** no
component is worse than the configured candidate's at `confidence`; otherwise the configured one stays.
This blocks, for example, a larger radius that fits the national shares by giving our plants the people
of plants missing from the list (worse in GB). `summary.md` names the best overall, by plants only and
by countries only, the top 10, and both candidates' measures for every reference, group and SDG region.

## Run Instructions
```bash
snakemake --profile workflow/profiles/slurm sweep          # layers, validation, choice
snakemake --profile workflow/profiles/slurm sweep_layers   # only the service areas and population layers
python -m wwtp.sensitivity_analysis_scripts.sweep_validation references   # by hand
python -m wwtp.sensitivity_analysis_scripts.sweep_validation score
```
Reference files go to `data/validation_data/` as the laptop has them: `gbr_catchments/`
(`catchments_consolidated.zip`, `waterbase_catchment_lookup.csv`, `waterbase_consolidated.csv`),
`nld_cbs_rwzi/` (the CBS xlsx), `usa_epa_sewersheds/Current_Release.zip` (read zipped),
`jmp_household/jmp_household.xlsx`, `eu_eurostat/eurostat_env_ww_con.tsv`,
`oecd_water_treat/oecd_water_treat.csv`.

## Parameters (`sweep_validation` in config.yaml)
| Key | Default | Effect |
| --- | --- | --- |
| `reference_years` | gb 2024, nl 2024, us 2020, hw 2024, uwwtd 2024 | `<year>_zonal_sum` compared with each plant reference; gb also its raster |
| `national_year`, `national_max_year_gap`, `national_min_plants` | 2022, 5, 5 | countries: population year, reference years accepted, countries scored |
| `national_series`, `eurostat_geo_aliases` | JMP `san_sew_sm_t` / `san_sew_t`, Eurostat `URB_OTH_CS`, OECD `PUBTOTTR`; EL->GR, UK->GB | which national series are read |
| `decision_weights` | gb 0.25, nl 0.25, jmp 0.5 | components of the choice |
| `bootstrap_samples`, `bootstrap_seed`, `confidence` | 1000, 0, 0.95 | the paired bootstrap |
| `gb_match_distance_m`, `us_match_distance_m` | 1000, 500 | plant matching |
| `threshold`, `eu_reference_factor` | from `eu_comparison` | UWWTD matching and capacity factor |
| `percent_verification` | from `verification_script` | the `ver` group |
| `paths.*` | `data/validation_data/...` | reference files; `output_dir` `data/sensitivity/sweep_validation` |
