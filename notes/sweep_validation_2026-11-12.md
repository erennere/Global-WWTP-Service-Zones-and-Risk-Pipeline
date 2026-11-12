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

## What decides now
- GB (Hoffmann et al. catchments, population raster in the catchment: geometry only) and NL (CBS
  inhabitants per RWZI). Neither was used to fit anything.
- Score: mean over GB and NL of the class-balanced error. Paired bootstrap over the reference units.
- The best candidate replaces the configured one only if it is better in GB and in NL at 95 %. This
  is the rule of the NL test (6 Nov): change a setting only if both countries agree.
- US sewersheds (calibration data), HydroWASTE and UWWTD are reported with their role, per group, but
  do not decide.
- The groups (ver / unver / single) are fixed once on the configured variant's basins.
- `served_by_country.csv` (people in service areas per country and candidate) is there for the
  country check against JMP.

## Check on real data (scratch runs of 6 Nov, 1 km WorldPop 2020 as counts)
Configured (level 6, size weighting, reported radius) against the same with a rigid 50 km buffer:

| candidate | GB class-bal. | GB typical | GB within x2 | GB total | GB capture | NL class-bal. | NL typical | NL within x2 | NL total |
|---|---|---|---|---|---|---|---|---|---|
| configured | x1.49 | x1.68 | 64 % | 1.15 | 0.69 | x1.57 | x1.50 | 73 % | 0.82 |
| rigid 50 km | x1.68 | x1.83 | 57 % | 1.36 | 0.73 | x1.34 | x1.36 | 81 % | 0.99 |

Mean score: rigid 50 km x1.50, configured x1.53, so the rigid buffer is "best"; it is better in NL
only, so the configured one stays. The numbers agree with the earlier GB and NL tests (NL: x1.47,
74 %, 0.84 with the raster scaled to 2024; GB: x1.56, 68 %, 1.11, capture 0.69 with cell centres).

## Running it
Copy `data/validation_data/gbr_catchments/`, `nld_cbs_rwzi/` and `usa_epa_sewersheds/` from the
laptop to the cluster's `data/validation_data/`; `pip install -e .` once (openpyxl). Then
`snakemake --profile workflow/profiles/slurm sweep --jobs 10 --keep-going`; result in
`data/sensitivity/sweep_validation/summary.md`.
