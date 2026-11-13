# wwtp

## Sections
| Section | Purpose |
| --- | --- |
| `Module Overview` | Explains what lives in `wwtp` and how the package is meant to be used |
| `Code Entry Files In This Package` | Points to the code/config files most readers should open first |
| `Required Starter Data Files` | Lists data artifacts the pipeline relies on before execution |
| `Configuration at a Glance` | Highlights the shared settings that shape most runs |
| `Shared Override Convention` | Shows how every stage takes runtime overrides |
| `Core Root-Level Pipeline Scripts` | Explains the key top-level scripts in more detail |
| `Section Guide` | Directs the reader to the stage-specific READMEs |

## Module Overview
`wwtp/` is the executable heart of the project. Every stage is a module run as `python -m wwtp.<stage>`; the Snakemake workflow (`workflow/Snakefile`, see the root README) runs them in order, locally or as SLURM jobs, and `wwtp/workflow.py` tells it which files each stage reads and writes. Most scripts read `config.yaml` through `starter.py`, so the normal way to change paths or parameters is to edit config (or an overlay) rather than code.

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

`workflow.py` lists each stage's inputs and outputs for Snakemake from the same config and path helpers the stages use; `settings.py` holds the typed model of every config section.

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
| `create_rasters.annotations.default_mode` | `sequential` | Launcher mode for raster preparation |
| `create_rasters.zoom_level` | `8` | Tile zoom used in raster/risk stages |
| `find_unserved_pop.threshold_value` | `1000` | Threshold for non-served population filtering |
| `find_intersection_river.x_distance` | `5000` | River proximity search distance for non-served polygons |
| `impact_polygons_pop.impact_polygons_pop_params.c_limit` | `5.0` | Downstream concentration threshold |
| `impact_polygons_pop.impact_polygons_pop_params.org_per_pop` | `60.0` | Organic load per person |
| `piechart_figure.min_total_size` | `50000000` | Minimum size threshold for pie-chart country inclusion |
| `piechart_interactive.min_total_size` | `50000000` | Minimum size threshold for interactive pie-map country inclusion |
| `download_and_vectorize.industrial_zenodo_url` | Zenodo URL | Industrial raster source |

Module-specific settings are documented in the corresponding README files under `wwtp/`.

## Shared Override Convention

Every entry point accepts the same seven named override flags, added by
`starter.add_standard_override_arguments` and read back by
`starter.parse_config_overrides`:

```bash
--level --version --buffer --weight-method --weight-func --dynamic-buffering --size-measure
```

Pass any subset; the workflow passes all seven for each variant:

```bash
python -m wwtp.create_voronoi --approach 1 --level 8 --buffer 9000 --size-measure footprint
```

The older positional form (`level version buffer …`) has been removed: a stray
argv token used to be silently accepted as `level`. Unknown flags now exit 2
instead of being shifted away.

`starter.py` resolves these overrides after YAML inheritance. Earlier sections in `config.yaml` define shared values; later sections inherit them with `null`; explicit CLI values win over both.

## Core Root-Level Pipeline Scripts

Four production scripts sit directly under `wwtp/` rather than inside a subpackage. They are documented here because they are central to the workflow and are often the first scripts a reader looks for.

| Script | Python module | What it does | Main inputs | Main outputs | Notes |
| --- | --- | --- | --- | --- | --- |
| `download_pop.py` | `wwtp.download_pop` | Downloads and prepares WorldPOP population inputs used later by `add_pop.py` and the risk workflow | `download_pop.paths.pop_dir` and shared config overrides | population rasters under `data/population` | Snakemake rule `download_pop` (leaves a stamp file, since its outputs are one folder per country) |
| `create_voronoi.py` | `wwtp.create_voronoi` | Runs the weighted Voronoi service-area stage across the configured approaches | canonical merged WWTP layer, watershed inputs, weighting settings, buffering settings | Voronoi GeoPackages under `create_voronoi.paths.voronoi_dir` and buffer outputs under `create_voronoi.paths.buffers_dir` | One Snakemake job per approach and variant (`create_voronoi__<tag>__a<approach>`) |
| `add_pop.py` | `wwtp.add_pop` | Intersects Voronoi outputs with population rasters and writes population-enriched service areas | the service areas of one approach (`--approach`), population rasters | population-enriched GeoPackages under `add_pop.paths.pop_output_dir` | One Snakemake job per approach and variant |
| `combine_watersheds.py` | `wwtp.combine_watersheds` | Unzips and merges the per-level HydroBASINS archives into one watershed layer per level | `../data/hydroshed_river_levels/lvl{level}/*.zip` | combined watershed GeoPackages consumed by `create_voronoi` and the risk stage | Iterates over the configured levels; raises naming any zip with no readable layer rather than skipping it |

### `download_pop`
`download_pop` is the population-ingest stage. It runs before `add_pop`. Its job is to build the population raster store consumed by later stages.

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

Key config:
- `download_pop.paths.pop_dir`

Typical run:
```bash
snakemake --profile workflow/profiles/slurm download_pop
python -m wwtp.download_pop
```

### `create_voronoi`
`create_voronoi` is the central service-area stage. It runs one or more of the three supported approaches: WWTP-only Voronoi, watershed-aware Voronoi, and the city-based branch (`city_voronoi`); the workflow runs each approach as its own job.

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

Key controls:
- `create_voronoi.level`
- `create_voronoi.buffer`
- `create_voronoi.weight_method`
- `create_voronoi.weight_func`
- `create_voronoi.dynamic_buffering`
- `create_voronoi.size_measure`
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
python -m wwtp.create_voronoi --approach 1 --verbose
python -m wwtp.create_voronoi --approach 0 1 --level 8 --buffer 15000 --dynamic-buffering false
```

### `add_pop`
`add_pop` is the population-attachment bridge between geometric service areas and downstream analysis. It takes an approach, reads that approach's service areas, opens the configured rasters, and writes population-enriched service-area outputs to `add_pop.paths.pop_output_dir`.

Workflow:
- Read the service areas of the requested approach (`create_output_paths(cfg)['voronoi'][approach]`; missing: it fails).
- Load the layer and read the country code column configured for the project.
- Resolve the available population rasters for each country represented in that file.
- Use `exactextract` to compute zonal statistics for every polygon.
- Add year-specific `*_zonal_sum` and `*_zonal_std` columns to the GeoPackage.
- Write the enriched result as `pop_added_<original_filename>.gpkg`.

Important behavior:
- The script takes the approach as the required `--approach` flag (0, 0_only_round, 1, 1_only_round, 2), alongside the shared override flags.
- It processes all discovered raster years by default and restores the original CRS before writing outputs.
- When `overwrite_existing` is false, existing outputs are skipped rather than regenerated.

Key config:
- `add_pop.add_pop_max_workers`
- `add_pop.country_output_column`
- `add_pop.overwrite_existing`
- `add_pop.paths.voronoi_dir`, `add_pop.paths.buffers_dir`
- `add_pop.paths.pop_tif_dir`
- `add_pop.paths.pop_output_dir`

Typical runs:
```bash
python -m wwtp.add_pop --approach 1
```

These outputs are the direct inputs for validation, non-served population calculation, figure generation, and downstream risk propagation.

## Section Guide
The stage-specific READMEs carry the detailed script descriptions, execution diagrams, configuration tables, and practical notes for each part of the pipeline.

| Stage | What it covers | Snakemake rules / targets | README |
| --- | --- | --- | --- |
| Data merge | Builds the canonical WWTP dataset | `combine_locations` | `wwtp/data_merge/README.md` |
| Annotation | Builds grid, OSM, image, and label context | `annotation_grids`, `wastewater_join`, `download_bing_annotate`, `merge_annotations`; target `annotations` | `wwtp/annotation_scripts/README.md` |
| Population and risk | Creates risk intermediates and population-at-risk outputs | `create_rasters__<tag>` (+ shards), `find_unserved_pop__<tag>`, `assign_rivers_to_basin__<tag>`, `find_intersection_river__<tag>`, `impact_polygons_pop__<tag>`, `find_pop_in_danger_pop__<tag>`, `find_diff_pop__<tag>` | `wwtp/pop_at_risk_river_calculations/README.md` |
| Figures and exports | Produces GeoJSON, plots, and report assets | `<figure>__<tag>`, `figures__<tag>` | `wwtp/figures_scripts/README.md` |
| Validation | Compares outputs to HydroWASTE and EU references | `verification_script__<tag>`, `hw_comparison__<tag>`, `eu_comparison__<tag>` | `wwtp/pop_validation_scripts/README.md` |
| Reference data | Downloads the validation references and national plant registers; lists the register plants our list lacks | target `national_data` (`national_data_fetch`, `national_registers`) | `wwtp/pop_validation_scripts/national_data.py` docstring |
| Sensitivity analysis | Runs the parameter sweep, validates it per plant (real catchments) and per country (JMP, Eurostat, OECD) and recommends a candidate | targets `sweep`, `sweep_layers` | `wwtp/sensitivity_analysis_scripts/README.md` |
| Calibration | Plant size: footprint fit with country offsets and country values for plants without a usable footprint, and a national scale so that the sizes of a country add up to its connected people (`create_voronoi.footprint_to_people`, `size_scale`). Radius: exponent against census (`calibration/service_radius.py`), scale by country against national connection rates in 126 countries, tested by leaving regions out (`calibration/connection_radius.py`; rule in `create_voronoi.radius_calibration.*.by_connection`) | target `calibration` (national_data, curves, `calibrate`, `plant_check` on the real catchments of US, GB, NL, IE, ES and Mexico's census, one results zip); `connection_radius` for the tables only | module docstrings, `notes/size_calibration_2026-11-12.md`, `notes/radius_calibration_2026-11-12.md` |
| Industrial analysis | Quantifies industrial land not covered by eligible WWTPs | `download_and_vectorize_lvl<level>`, `find_unconnected_industrial_areas__<tag>` | `wwtp/industrial_analysis/README.md` |
