# src

## Sections
| Section | Purpose |
| --- | --- |
| `Module Overview` | Explains what lives in `src` and how the package is meant to be used |
| `Code Entry Files In This Package` | Points to the code/config files most readers should open first |
| `Required Starter Data Files` | Lists data artifacts the pipeline relies on before execution |
| `Configuration at a Glance` | Highlights the shared settings that shape most runs |
| `Shared Override Convention` | Shows how the shell wrappers pass runtime overrides |
| `Core Root-Level Pipeline Scripts` | Explains the key top-level scripts in more detail |
| `Section Guide` | Directs the reader to the stage-specific READMEs |

## Module Overview
`src/` is the executable heart of the project. Shell wrappers provide reproducible entry points and scheduler compatibility, while the Python modules hold the actual processing logic. Most scripts read `config.yaml` through `starter.py`, so the normal way to change paths or parameters is to edit config rather than code.

> **Run every `.sh` wrapper from `src/`, never from anywhere else** — for both `bash` and `sbatch`. Each wrapper sets `PROJECT_ROOT="."` and resolves `lib/utils.sh`, `config.yaml`, and `logs/` relative to it.
>
> ```bash
> cd src
> bash pop_at_risk_river_calculations/create_rasters.sh --level 8
> sbatch pop_at_risk_river_calculations/create_rasters.sh --level 8
> ```

## Code Entry Files In This Package

| File | Role | When you use it |
| --- | --- | --- |
| `config.yaml` | Canonical workflow configuration | before any run |
| `starter.py` | Shared config resolver and named-override parser | when tracing how values are inherited |
| `pipelines.py` | Shared output-path and helper orchestration utilities | when tracing cross-stage path creation |
| `geo_utils.py` | Shared spatial helpers | before writing any new CRS, geometry-repair, or nearest-neighbour code |
| `utils.py` | Shared domain-agnostic helpers | before writing any new logging, filesystem, DuckDB, or column-resolution code |

### Shared helper modules

`geo_utils.py` and `utils.py` are the two consolidation targets. When the same
logic is needed in more than one stage it belongs in one of them, not in a third
copy — most of the duplication the repository has accumulated came from not
knowing they exist.

| Module | Scope | Representative helpers |
| --- | --- | --- |
| `geo_utils.py` | anything that depends on geometry or CRS | `estimate_utm_crs`, `estimate_utm_epsg_for_geom`, `repair_geometry`, `nearest_neighbor_distances`, `buffer_geometry`, `load_eu_reference_layer`, `ensure_duckdb_spatial` |
| `utils.py` | filesystem, logging, SQL, column conventions — no geometry | `configure_logging`, `ensure_output_dir_for_file`, `duckdb_connection`, `resolve_latest_zonal_sum_column`, `industrial_category_mask`, `robust_bounds`, `clip_to_robust_bounds` |

`figures_scripts/_shared.py` is a third, narrower target: figure-layer column
conventions shared by the two piechart scripts only.

Shell launchers are execution wrappers, not starter dependencies. They are documented in the stage READMEs and in the root README canonical run order.

## Required Starter Data Files

These are the input data artifacts resolved from `config.yaml` that the repository depends on for end-to-end execution.

| Data file or directory | Why it is required | Default path status in this workspace |
| --- | --- | --- |
| `../data/Enhanced_HW_WWTP__jun20_2025.geojson` | seed corrected WWTP source for merge chain | present |
| `../data/wastewater_plant.geojson` | OSM correction reference layer | present |
| `../data/DL_results/csvv2-2.zip` and `../data/DL_results/aftersort.geojson` | legacy segmentation merge inputs | present |
| `../data/extra_points/Canada_14_03_2025.csv` | final merge enrichment | present |
| `../data/extra_points/Germany_Hydra_waste_geospatial_corrected.geojson` | final merge enrichment | present |
| `../data/final_data_source/final_W_europe_WWT_dec3.geojson` | final merge enrichment | present |
| `../data/final_data_source/Thailand_500m_merged.geojson` | final merge enrichment | present |
| `../data/final_data_source/final_USA_WWT_dec3.geojson` (or a replacement set in config) | final merge enrichment | missing at default path |
| `../data/hydroshed_river_levels/lvl{level}/` | input zip directory for `combine_watersheds.py` | needed for the default `level=6` and for every level the sweep runs (5, 6, 7) |
| `../data/cleaned_hydrowaste.csv` | core Voronoi source table | present |
| `../data/bboxes.csv` | Voronoi support table | present |
| `../data/cities.csv` | city-based Voronoi approach input | present |
| `../data/hydrorivers.gpkg` | river assignment and downstream-risk calculations | missing |
| `../data/extra_points/UWWTD_TreatmentPlants.gpkg` | EU validation reference | present |
| `../data/boundaries/ne_110m_admin_0_countries.shp` (+ sidecars) | country overlays for figure scripts | present |

Defaults for several annotation/segmentation paths are cluster-specific and must usually be overridden locally in `config.yaml`, especially:
- `merge_seg_results.paths.seg_results_filepath`
- `download_bing_annotate.paths.annotations_images_dir`
- `merge_annotations.paths.annotations_results_filepath`

## Configuration at a Glance
The full parameter tables now live in the relevant module READMEs. This file keeps the global, cross-stage settings that matter before you run anything.

| Setting | Default | Where it matters |
| --- | --- | --- |
| `correct_locations_w_OSM.rad` | `5000` | OSM fallback correction radius for the first merge stage |
| `merge_seg_results.legacy_merge` | `true` | Whether the legacy segmentation branch runs during merge |
| `create_voronoi.weight_method` | `size` | Global weighting transform for Voronoi stages (`size`: sqrt of plant size in people) |
| `create_voronoi.weight_func` | `mult` | Global weighted-distance mode for Voronoi stages |
| `create_voronoi.dynamic_buffering` | `true` | Enables per-site dynamic buffers |
| `create_voronoi.size_measure` | `reported` | Plant size in people for the service radius and `size` weights: `reported` (reported size, footprint estimate where missing) or `footprint` |
| `create_voronoi.execution.mode` | `sequential` | Launcher mode for the Voronoi stage |
| `create_rasters.annotations.default_mode` | `sequential` | Launcher mode for raster preparation |
| `create_rasters.zoom_level` | `8` | Tile zoom used in raster/risk stages |
| `find_unserved_pop.threshold_value` | `1000` | Threshold for non-served population filtering |
| `find_intersection_river.x_distance` | `5000` | River proximity search distance for non-served polygons |
| `impact_polygons_pop.impact_polygons_pop_params.c_limit` | `5.0` | Downstream concentration threshold |
| `impact_polygons_pop.impact_polygons_pop_params.org_per_pop` | `60.0` | Organic load per person |
| `piechart_figure.min_total_size` | `50000000` | Minimum size threshold for pie-chart country inclusion |
| `piechart_interactive.min_total_size` | `50000000` | Minimum size threshold for interactive pie-map country inclusion |
| `download_and_vectorize.industrial_zenodo_url` | Zenodo URL | Industrial raster source |

Module-specific settings are documented in the corresponding README files under `src/`.

## Shared Override Convention

Every entry point accepts the same seven named override flags, added by
`starter.add_standard_override_arguments` and read back by
`starter.parse_config_overrides`:

```bash
--level --version --buffer --weight-method --weight-func --dynamic-buffering --size-measure
```

The shell wrappers collect whatever subset you pass and forward it verbatim as
`"${OVERRIDE_ARGS[@]}"`:

```bash
cd src
bash create_voronoi.sh --level 8 --buffer 9000 --size-measure footprint
```

The older positional form (`level version buffer …`) has been removed: a stray
argv token used to be silently accepted as `level`. Unknown flags now exit 2
instead of being shifted away.

`starter.py` resolves these overrides after YAML inheritance. Earlier sections in `config.yaml` define shared values; later sections inherit them with `null`; explicit CLI values win over both.

## Core Root-Level Pipeline Scripts

Four production scripts sit directly under `src/` rather than inside a subpackage. They are documented here because they are central to the workflow and are often the first scripts a reader looks for.

| Script | Python module | What it does | Main inputs | Main outputs | Notes |
| --- | --- | --- | --- | --- | --- |
| `download_pop.sh` | `src.download_pop` | Downloads and prepares WorldPOP population inputs used later by `add_pop.py` and the risk workflow | `download_pop.paths.pop_dir` and shared config overrides | population rasters under `data/population` | Installs the package in editable mode, logs to `logs/pop_run.log`, and supports the shared named override flags |
| `create_voronoi.sh` | `src.create_voronoi` | Runs the weighted Voronoi service-area stage across the configured approaches | canonical merged WWTP layer, watershed inputs, weighting settings, buffering settings | Voronoi GeoPackages under `create_voronoi.paths.voronoi_dir` and buffer outputs under `create_voronoi.paths.buffers_dir` | Supports `array`, `sequential`, and `parallel` execution modes via `create_voronoi.execution.mode` |
| `add_pop.sh` | `src.add_pop` | Intersects Voronoi outputs with population rasters and writes population-enriched service areas | one Voronoi file index, Voronoi outputs, population rasters | population-enriched GeoPackages under `add_pop.paths.pop_output_dir` | Runs either as a local single-index job or as a SLURM array task |
| `combine_watersheds.sh` | `src.combine_watersheds` | Unzips and merges the per-level HydroBASINS archives into one watershed layer per level | `../data/hydroshed_river_levels/lvl{level}/*.zip` | combined watershed GeoPackages consumed by `create_voronoi` and the risk stage | Iterates over the configured levels; raises naming any zip with no readable layer rather than skipping it |

### `download_pop`
`download_pop.sh` is the population-ingest entry point. Run it after the source WWTP and watershed inputs are in place and before `add_pop.sh`. Its job is to build the population raster store consumed by later stages.

Workflow:
- Build WorldPop download URLs for all countries and years used by the project.
- Download archives or GeoTIFF files into the configured population directory.
- Extract compressed assets into per-country folders.
- If a country is delivered as CSV rather than TIFF, rasterize the point data to GeoTIFF.
- Mosaic multiple tiles into one country-level raster when needed.

Important outputs:
- `download_pop.paths.pop_dir/zipped`
- `download_pop.paths.pop_dir/unzipped`
- `download_pop.paths.pop_dir/rasterized`
- `download_pop.paths.pop_dir/merged`

Important behavior:
- The module uses `get_urls()` as the default URL source and keeps the HDX-based helper as a fallback path rather than the main workflow.
- CSV-based country inputs are projected to an estimated local UTM CRS for rasterization and then reprojected back to `EPSG:4326`.
- The shell wrapper reinstalls the package in editable mode before execution and writes a consolidated run log to `logs/pop_run.log`.

Key config:
- `download_pop.paths.pop_dir`

Typical run:
```bash
bash download_pop.sh
```

### `create_voronoi`
`create_voronoi.sh` is the central service-area launcher in the repository. It resolves `create_voronoi.execution.mode` from config and then executes one or more of the three supported approaches: WWTP-only Voronoi, watershed-aware Voronoi, and the city-based branch.

Approaches:
- `0`: WWTP Voronoi without watershed grouping.
- `1`: WWTP Voronoi with watershed-aware grouping and clipping.
- `2`: city-based Voronoi.

Workflow:
- Load the resolved configuration through `starter.py`.
- Build output paths from the current parameter combination.
- Select the requested approaches or default to all three.
- Skip completed approaches when outputs already exist and overwrite is disabled.
- Prepare grouped inputs, compute weights, generate weighted Voronoi regions, resolve overlaps, and clip results to the relevant basin or country boundary.

Execution modes:
- `sequential`: run all selected approaches one after another.
- `array`: run one approach per SLURM array task.
- `parallel`: run the approaches concurrently in the background.

Key controls:
- `create_voronoi.level`
- `create_voronoi.buffer`
- `create_voronoi.weight_method`
- `create_voronoi.weight_func`
- `create_voronoi.dynamic_buffering`
- `create_voronoi.size_measure`
- `create_voronoi.execution.mode`
- `create_voronoi.n_points`
- `create_voronoi.threshold`
- `create_voronoi.calculate_area_fn`
- `create_voronoi.calculate_buffer_fn`
- `create_voronoi.prepare_data_fn`

CLI details:
- Supports `--approach` to run only selected approaches.
- Supports `--only_round` to switch to the round-area weighting path.
- Supports `--verbose` for more detailed logging.

Typical runs:
```bash
bash create_voronoi.sh
python -m src.create_voronoi --approach 1 --verbose
python -m src.create_voronoi 8 2 15000 size mult true reported
```

### `add_pop`
`add_pop.sh` is the population-attachment bridge between geometric service areas and downstream analysis. It takes a Voronoi layer index, opens the configured rasters, and writes population-enriched service-area outputs to `add_pop.paths.pop_output_dir`.

Workflow:
- Select one Voronoi GeoPackage by zero-based index.
- Load the layer and read the country code column configured for the project.
- Resolve the available population rasters for each country represented in that file.
- Use `exactextract` to compute zonal statistics for every polygon.
- Add year-specific `*_zonal_sum` and `*_zonal_std` columns to the GeoPackage.
- Write the enriched result as `pop_added_<original_filename>.gpkg`.

Important behavior:
- The script takes the Voronoi file index as the required `--index` flag, alongside the shared override flags.
- It can run locally for a single index or in SLURM array mode via `SLURM_ARRAY_TASK_ID`.
- It processes all discovered raster years by default and restores the original CRS before writing outputs.
- When `overwrite_existing` is false, existing outputs are skipped rather than regenerated.

Key config:
- `add_pop.add_pop_max_workers`
- `add_pop.country_output_column`
- `add_pop.overwrite_existing`
- `add_pop.paths.voronoi_dir`
- `add_pop.paths.pop_tif_dir`
- `add_pop.paths.pop_output_dir`

Typical runs:
```bash
bash add_pop.sh 0
sbatch add_pop.sh
```

These outputs are the direct inputs for validation, non-served population calculation, figure generation, and downstream risk propagation.

## Section Guide
The stage-specific READMEs carry the detailed script descriptions, execution diagrams, configuration tables, and practical notes for each part of the pipeline.

| Stage | What it covers | Canonical launcher(s) | README |
| --- | --- | --- | --- |
| Data merge | Builds the canonical WWTP dataset | `data_merge/combine_locations.sh` | `src/data_merge/README.md` |
| Annotation | Builds grid, OSM, image, and label context | `annotation_scripts/grid_generation_and_osm_extract.sh`, `annotation_scripts/run_download_bing_annotate_array.sh`, `annotation_scripts/merge_annotations.sh` | `src/annotation_scripts/README.md` |
| Population and risk | Creates risk intermediates and population-at-risk outputs | `pop_at_risk_river_calculations/create_rasters.sh`, `pop_at_risk_river_calculations/pop_differences_and_impact_polygons.sh`, `pop_at_risk_river_calculations/find_pop_in_danger_pop.sh` | `src/pop_at_risk_river_calculations/README.md` |
| Figures and exports | Produces GeoJSON, plots, and report assets | `figures_scripts/piechart.sh`, `figures_scripts/composite_area_population_plots.sh`, `figures_scripts/pop_at_risk_figures.sh` | `src/figures_scripts/README.md` |
| Validation | Compares outputs to HydroWASTE and EU references | `pop_validation_scripts/comparison.sh` | `src/pop_validation_scripts/README.md` |
| Sensitivity analysis | Runs parameter sweeps and comparison summaries | `sensitivity_analysis_scripts/create_voronoi_param_sweep.sh`, `sensitivity_analysis_scripts/add_pop_param_sweep.sh`, `sensitivity_analysis_scripts/compare_pop_sweep_hw_eu.sh` | `src/sensitivity_analysis_scripts/README.md` |
| Industrial analysis | Quantifies industrial land not covered by eligible WWTPs | `industrial_analysis/industrial_analysis.sh` | `src/industrial_analysis/README.md` |
