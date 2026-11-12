# Sweep validation and choice (2026-11-12)

Replaces the sweep ranking (`compare_pop_sweep_hw_eu`, `sweep_ver_ranking`). Code:
`wwtp/sensitivity_analysis_scripts/sweep_validation.py`, config `sweep_validation`, part of the
Snakemake target `sweep`. Method in the module docstring and the folder README.

## Why the old ranking could not choose
- It scored people per plant against HydroWASTE reported values and UWWTD capacity. With reported
  sizes these numbers set the circle and the Voronoi weight, so candidates with reported sizes were
  scored against their own input.
- z-scores within the grid and fixed weights (HydroWASTE 0.7, UWWTD 0.3): relative only, and they
  change when the grid changes.
- The ver / unver / single split was made on each candidate's own basin level, so levels 5, 6 and 7
  were scored on different plants.
- Nothing global: no country or regional comparison.

## Two levels
- **Plants:** GB (Hoffmann et al. catchments, population raster in the catchment: geometry only), NL
  (CBS inhabitants per RWZI), US (EPA sewersheds; the radius was calibrated on them), HydroWASTE
  reported and UWWTD (inputs of candidates with reported sizes). Groups ver / unver / single fixed on
  the configured variant's basins.
- **Countries and SDG regions:** people in service areas / people of the country's raster (2022)
  against JMP sewered-and-treated (global), JMP sewer connection (upper bound), Eurostat and OECD.
  Countries with at least 5 plants.

## The choice
- Score: each component's error relative to the configured candidate, weighted GB 0.25, NL 0.25,
  JMP 0.5 (half plants, half countries). Plants: class-balanced error; countries: mean absolute
  error in percentage points.
- Paired bootstrap over plants and countries. The best replaces the configured candidate only if it
  is better overall and worse in no component, both at 95 %.
- Why both: the two levels pull apart. A larger radius fits the national shares by giving our plants
  the people of plants missing from the list (connection-radius note): better for countries, worse
  per plant in GB. The rule only switches when a candidate does not pay for one with the other.
- Part of every candidate's national gap is plants missing from our list; it is common to all
  candidates, so it moves the level of the national error, not the comparison.

## Check on real data (scratch runs of 6 Nov, GB and NL only, 1 km WorldPop 2020 as counts)
Configured (level 6, size weighting, reported radius) against the same with a rigid 50 km buffer:

| candidate | GB class-bal. | GB typical | GB total | GB capture | NL class-bal. | NL total | JMP error (GB, NL) |
|---|---|---|---|---|---|---|---|
| configured | x1.49 | x1.68 | 1.15 | 0.69 | x1.57 | 0.82 | 14.0 pp |
| rigid 50 km | x1.68 | x1.83 | 1.36 | 0.73 | x1.34 | 0.99 | 2.1 pp |

The rigid buffer is better for NL and the national shares, but significantly worse in GB (plant loads
x1.36), so the configured candidate stays. On the cluster the national level covers every country
with plants and a JMP value (about 125), not two.

## Running it
Copy `data/validation_data/` (gbr_catchments, nld_cbs_rwzi, usa_epa_sewersheds, jmp_household,
eu_eurostat, oecd_water_treat) from the laptop to the cluster's `data/validation_data/`;
`pip install -e .` once (openpyxl). Then
`snakemake --profile workflow/profiles/slurm sweep --jobs 10 --keep-going`; result in
`data/sensitivity/sweep_validation/summary.md`.
