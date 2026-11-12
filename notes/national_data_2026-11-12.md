# National reference data and registers (2026-11-12)

Code: `wwtp/pop_validation_scripts/national_data.py`, config `national_data`, Snakemake target
`national_data` (`national_data_fetch` local, `national_registers`). The sweep validation takes its
reference files from the same download.

## Why
- The sandboxes cannot reach the hosts of the national datasets; the cluster can. One command there
  downloads everything, describes it and lists the plants our list lacks.
- A global validation needs four kinds of truth (inventory with links and status in the project file
  `claude/validation_datasets_inventory.md`): catchment polygons (geometry), plant registers (size,
  completeness, new plants), census sewer connection by district, national totals.

## What the first run gives
- `data/validation_data/manifest.json`: which addresses worked.
- `data/validation_data/national_data/new_plants.gpkg` (+ `.csv`): register plants farther than 1 km
  from every plant of ours, all countries in one file. Registers in this run: Brazil (ANA Atlas Esgotos
  2019), Canada (ECCC WSER), Australia (Geoscience Australia), Japan (MLIT P22; the zip may hold
  pumping stations as a second layer, column `layer` tells them apart).
- `peek.md`: structure of every new file (SDG 6.3.1 summary, UNSD, Eurostat NUTS 2, NL treatment
  units, Ireland and Spain agglomerations, ANA tables, INEGI census by locality). The parsers for the
  next validation levels are written from it.
- `results.zip`: the small files to send back.

## Not checked when added
The Japan P22 and INEGI ITER addresses, and the UNSD table, were not opened; `manifest.json` shows
whether they worked. US (CWNS 2022) and France (SANDRE) registers have no stable direct address yet.

## Next, from the first results
1. Sweep choice from `sweep_validation/summary.md`; set the recommended candidate in `create_voronoi`.
2. Plant level: add Ireland and the NL treatment units (capture) once their fields are known.
3. District level: Brazil (ANA municipal table, census), Mexico (INEGI by locality), Eurostat NUTS 2.
4. National: SDG 6.3.1 and UNSD next to JMP.
5. Register sizes (Brazil, Japan) as held-out plant tests.
