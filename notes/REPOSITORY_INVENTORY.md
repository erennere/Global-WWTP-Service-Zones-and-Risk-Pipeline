# Repository Inventory

## 1. Main Pipeline Order

- Recommended end-to-end order, explicitly documented in `README.md` and `src/README.md`:
- `src/data_merge/combine_locations.sh`: canonical WWTP point build chain.
- `src/combine_watersheds.sh`: combined watershed layer build.
- `src/annotation_scripts/grid_generation_and_osm_extract.sh`: annotation grids plus OSM context.
- `src/annotation_scripts/run_download_bing_annotate_array.sh`: imagery and annotation asset generation.
- `src/annotation_scripts/merge_annotations.sh`: annotation merge back into the WWTP dataset.
- `src/download_pop.sh`: population raster download and preparation.
- `src/create_voronoi.sh`: Voronoi service-area generation.
- `src/add_pop.sh`: population attachment to one Voronoi file index per run.
- `src/pop_at_risk_river_calculations/create_rasters.sh`: raster preparation for non-served analysis.
- `src/pop_at_risk_river_calculations/pop_differences_and_impact_polygons.sh`: non-served, difference, river-linking, and impact-polygon stages.
- `src/pop_at_risk_river_calculations/find_pop_in_danger_pop.sh`: final population-at-risk aggregation.
- `src/industrial_analysis/industrial_analysis.sh`: industrial-land branch.
- `src/pop_validation_scripts/comparison.sh` and the figure scripts under `src/figures_scripts`: validation and visualization after generation stages.
- Adjacent but not part of the documented main order:
- `src/annotation_scripts/annotations_inspection.sh`: annotation QA sampling.
- `src/annotation_scripts/copy_falsy_images.sh`: annotation QA helper.
- `src/figures_scripts/pop_at_risk_figures.sh`: risk-result visualization.
- The sensitivity-analysis branch is separate from the baseline production order and is documented in `src/sensitivity_analysis_scripts/README.md`.

## 2. Major Directories And Purpose

- Top-level `data/`: stores raw inputs, intermediates, and generated outputs, as described in `README.md`.
- `data/boundaries/`: Natural Earth country-boundary shapefile components.
- `data/DL_results/`: DL or segmentation result artifacts referenced by config keys `dl_dir`, `dl_zipfile`, and `dl_mapfile`.
- `data/extra_points/`: extra country-specific WWTP source inputs referenced by `src/config.yaml`.
- `data/final_data_source/`: final or base country-specific WWTP source datasets referenced by `src/config.yaml`.
- `data/figures/`: generated figure outputs and interactive HTML map artifacts.
- Top-level `src/`: executable codebase, shell wrappers, shared orchestration, and configuration.
- `src/data_merge/`: canonical WWTP point dataset construction and harmonization.
- `src/annotation_scripts/`: annotation-grid generation, OSM extraction, imagery rendering, annotation merging, and QA.
- `src/industrial_analysis/`: industrial raster download/vectorization and uncovered-industrial-area analysis.
- `src/figures_scripts/`: static and interactive communication outputs.
- `src/pop_at_risk_river_calculations/`: non-served population, river linkage, downstream impact, and final population-at-risk outputs.
- `src/pop_validation_scripts/`: validation and comparison against HydroWaste and EU references.
- `src/sensitivity_analysis_scripts/`: parameter sweeps and sweep-result comparison tooling.

## 3. Python Entrypoints

### Top-level

- `src/download_pop.py`: population download and raster preparation.
- `src/combine_watersheds.py`: watershed archive merge.
- `src/create_voronoi.py`: main Voronoi generation entrypoint.
- `src/add_pop.py`: attach population to a selected Voronoi output.

### Data merge

- `src/data_merge/correct_locations_w_OSM.py`
- `src/data_merge/merge_seg_results.py`
- `src/data_merge/final_data_merge.py`

### Annotation

- `src/annotation_scripts/NEW_01_GENERATEGRIDS.py`
- `src/annotation_scripts/NEW_02_EXTRACTOSMDATAFULL_GEOJSON.py`
- `src/annotation_scripts/NEW_03_WASTEWATERJOIN_GEOJSON.py`
- `src/annotation_scripts/NEW_04_EXPORTGEOTIFF.py`
- `src/annotation_scripts/download_bing_annotate.py`
- `src/annotation_scripts/merge_annotations.py`
- `src/annotation_scripts/annotations_inspection.py`
- `src/annotation_scripts/copy_falsy_images.py`
- `src/annotation_scripts/sample_annotations_by_class.py`: ad-hoc class-check helper; the only `config.yaml` section that does not resolve, and not wired into any chain (see `src/annotation_scripts/README.md`).
- `src/annotation_scripts/README.md` explicitly says `NEW_03_WASTEWATERJOIN_GEOJSON.py` is currently not used, and `NEW_04_EXPORTGEOTIFF.py` is not used and not tested.

### Industrial analysis

- `src/industrial_analysis/download_and_vectorize.py`
- `src/industrial_analysis/find_unconnected_industrial_areas.py`

### Figures

- `src/figures_scripts/composite_area_population_plots.py`
- `src/figures_scripts/piechart_figure.py`
- `src/figures_scripts/piechart_interactive.py`
- `src/figures_scripts/sizes_interactive_map.py`
- `src/figures_scripts/interactive_unconnected_industrial_map.py`
- `src/figures_scripts/pop_at_risk_figures.py`
- `src/figures_scripts/_shared.py`: not an entrypoint; shared figure-layer column conventions for the two piechart scripts.
- `convert_voronoi_to_geojson_for_map.py`/`.sh` were deleted; their function moved into the other figure scripts.

### Population-at-risk and river calculations

- `src/pop_at_risk_river_calculations/create_rasters.py`
- `src/pop_at_risk_river_calculations/find_unserved_pop.py`
- `src/pop_at_risk_river_calculations/find_diff_pop.py`
- `src/pop_at_risk_river_calculations/assign_rivers_to_basin.py`
- `src/pop_at_risk_river_calculations/find_intersection_river.py`
- `src/pop_at_risk_river_calculations/impact_polygons_pop.py`
- `src/pop_at_risk_river_calculations/find_pop_in_danger_pop.py`

### Validation

- `src/pop_validation_scripts/verification_script.py`
- `src/pop_validation_scripts/hw_comparison.py`
- `src/pop_validation_scripts/eu_comparison.py`

### Sensitivity analysis

- `src/sensitivity_analysis_scripts/create_voronoi_parallel_sweep.py`
- `src/sensitivity_analysis_scripts/compare_pop_sweep_hw_eu.py`
- `src/sensitivity_analysis_scripts/sweep_ver_ranking.py`

## 4. Bash Scripts And Which Python Files They Execute

### Top-level

- `src/download_pop.sh` -> `src/download_pop.py`
- `src/combine_watersheds.sh` -> `src/combine_watersheds.py`
- `src/create_voronoi.sh` -> `src/create_voronoi.py`, either once for all approaches or once per selected approach depending on `execution.mode`
- `src/add_pop.sh` -> `src/add_pop.py`

### Data merge

- `src/data_merge/combine_locations.sh` -> `src/data_merge/correct_locations_w_OSM.py` -> `src/data_merge/merge_seg_results.py` with `--variant old` -> `src/data_merge/final_data_merge.py` -> `src/data_merge/merge_seg_results.py` with `--variant new`

### Annotation

- `src/annotation_scripts/grid_generation_and_osm_extract.sh` -> `src/annotation_scripts/NEW_01_GENERATEGRIDS.py` -> `src/annotation_scripts/NEW_02_EXTRACTOSMDATAFULL_GEOJSON.py`
- `src/annotation_scripts/run_download_bing_annotate_array.sh` -> `src/annotation_scripts/download_bing_annotate.py` with `instance_id`, `--num-instances`, and `--split-seed`
- `src/annotation_scripts/merge_annotations.sh` -> `src/annotation_scripts/merge_annotations.py`
- `src/annotation_scripts/annotations_inspection.sh` -> `src/annotation_scripts/annotations_inspection.py`
- `src/annotation_scripts/copy_falsy_images.sh` -> `src/annotation_scripts/copy_falsy_images.py`

### Industrial

- `src/industrial_analysis/industrial_analysis.sh` -> `src/industrial_analysis/download_and_vectorize.py` -> `src/industrial_analysis/find_unconnected_industrial_areas.py`

### Figures

- `src/figures_scripts/composite_area_population_plots.sh` -> `src/figures_scripts/composite_area_population_plots.py`
- `src/figures_scripts/pop_at_risk_figures.sh` -> `src/figures_scripts/pop_at_risk_figures.py`
- `src/figures_scripts/piechart.sh` -> `src/figures_scripts/piechart_figure.py`, `piechart_interactive.py`, `sizes_interactive_map.py`
- `src/figures_scripts/interactive_unconnected_industrial_map.sh` -> `src/figures_scripts/interactive_unconnected_industrial_map.py`

### Population-at-risk and river calculations

- `src/pop_at_risk_river_calculations/create_rasters.sh` -> `src/pop_at_risk_river_calculations/create_rasters.py`
- `src/pop_at_risk_river_calculations/find_unserved_pop.sh` -> `src/pop_at_risk_river_calculations/find_unserved_pop.py`
- `src/pop_at_risk_river_calculations/assign_rivers_to_basin.sh` -> `src/pop_at_risk_river_calculations/assign_rivers_to_basin.py`, with `--max-workers 2`
- `src/pop_at_risk_river_calculations/find_intersection_river.sh` -> `src/pop_at_risk_river_calculations/find_intersection_river.py`, with `--max-workers 32`
- `src/pop_at_risk_river_calculations/pop_differences_and_impact_polygons.sh` -> `find_unserved_pop.py` -> `find_diff_pop.py --index 0 --is-parallel true` -> `assign_rivers_to_basin.py --max-workers 2` -> `find_intersection_river.py --max-workers 32` -> `impact_polygons_pop.py --max-workers 64`
- `src/pop_at_risk_river_calculations/find_pop_in_danger_pop.sh` -> `src/pop_at_risk_river_calculations/find_pop_in_danger_pop.py`

### Validation

- `src/pop_validation_scripts/comparison.sh` -> `src/pop_validation_scripts/verification_script.py` -> `src/pop_validation_scripts/hw_comparison.py` -> `src/pop_validation_scripts/eu_comparison.py`

### Sensitivity analysis

- `src/sensitivity_analysis_scripts/create_voronoi_param_sweep.sh` -> repeated runs of `src/create_voronoi.py` with `--approach 1`
- `src/sensitivity_analysis_scripts/add_pop_param_sweep.sh` -> repeated runs of `src/add_pop.py` across discovered file indexes
- `src/sensitivity_analysis_scripts/create_voronoi_param_sweep_parallel.sh` -> `src/sensitivity_analysis_scripts/create_voronoi_parallel_sweep.py` -> subprocess calls to `src/create_voronoi.py`
- `src/sensitivity_analysis_scripts/industrial_analysis_sweep.sh` -> `src/industrial_analysis/download_and_vectorize.py` -> `src/industrial_analysis/find_unconnected_industrial_areas.py`
- `src/sensitivity_analysis_scripts/compare_pop_sweep_hw_eu.sh` -> `src/sensitivity_analysis_scripts/compare_pop_sweep_hw_eu.py`
- `src/sensitivity_analysis_scripts/sweep_ver_ranking.sh` -> `src/sensitivity_analysis_scripts/sweep_ver_ranking.py`
- `src/sensitivity_analysis_scripts/create_voronoi_single_additive_test.sh` -> single-combination smoke run of the additive regime (additive is not in the default sweep grid)

## 5. Configuration Systems And Parameter Interfaces

- `src/config.yaml` is the single repository-wide default configuration source. The top-level and research-code READMEs both state that defaults live there rather than inside the Python modules.
- `src/config.yaml` contains these explicit configuration sections: `arguments`, `paths`, `s3`, `params`, `annotations`, `booleans`, `figures`, `credentials`, `execution`, and `impact_polygons_pop_params`.
- `src/starter.py` is the central config loader. It normalizes optional CLI values, parses optional integers, floats, and booleans, expands path templates, derives `weight_type` and `weight_func_suffix`, resolves dynamic-buffer path tokens, and returns the flattened runtime cfg dictionary.
- The shared override interface, added by `starter.add_standard_override_arguments` and parsed by `starter.parse_config_overrides`, is the seven named flags `--level`, `--version`, `--buffer`, `--weight-method`, `--weight-func`, `--dynamic-buffering`, `--dynamic-buffer-k`. The legacy positional form has been removed, and `lib/utils.sh parse_overrides` now exits 2 on an unknown flag instead of shifting it away.
- The config explicitly supports swappable callable names through `calculate_area_fn`, `calculate_buffer_fn`, and `prepare_data_fn` in `src/config.yaml`, with deferred resolution handled in `src/pipelines.py`.
- `src/create_voronoi.py` adds `--approach`, `--only-round`, and `--verbose` on top of the shared overrides.
- `src/industrial_analysis/find_unconnected_industrial_areas.py` adds `--approach`, `--only-round`, and `--verbose`, but only accepts approach `0` or `1`.
- `src/data_merge/merge_seg_results.py` adds `--variant` with choices `old` and `new`.
- `src/annotation_scripts/download_bing_annotate.py` adds `instance_id`, `--num-instances`, and `--split-seed`.
- `src/pop_at_risk_river_calculations/create_rasters.py` adds `--job-index` and `--total-jobs`.
- `src/pop_at_risk_river_calculations/find_diff_pop.py` adds `--index` and `--is-parallel`.
- `src/add_pop.py` adds a required `--index` (Voronoi file index).
- `src/figures_scripts/composite_area_population_plots.py` adds `--approach`, `--color-col`, `--zonal-col`, `--hist-lower-q`, and `--hist-upper-q`.
- `src/sensitivity_analysis_scripts/create_voronoi_parallel_sweep.py` adds `task_id`, `version`, `--approach`, `--num-jobs`, `--retry-failed-runs`, and `--shuffle-seed`. Its positional `dynamic_buffering` and `dynamic_buffer_k` arguments are explicitly accepted only for backward compatibility and ignored.
- `src/create_voronoi.sh` reads `execution.mode` from `src/config.yaml` to choose array, sequential, or parallel wrapper behavior.
- `src/pop_at_risk_river_calculations/create_rasters.sh` reads `annotations.default_mode` from `src/config.yaml` to choose array, sequential, or parallel wrapper behavior.
- The sweep wrappers use `SLURM_ARRAY_TASK_ID` and `SHUFFLE_SEED` as execution and sharding interfaces.
- Non-YAML config surface: the Bing Maps key is read from the required `BING_API_KEY` environment variable; it is no longer stored in `config.yaml` or in a module-level constant.

## 6. Model Architectures Available In The Repository

- No trainable ML architecture implementation is explicitly present in the repository code. The codebase contains no PyTorch, TensorFlow, Keras, XGBoost, LightGBM, CatBoost, or sklearn training imports, and no `train`, `fit`, `epoch`, `optimizer`, or `loss` workflow was found in the executable Python modules.
- External model outputs are consumed as inputs rather than implemented here:
- `src/data_merge/merge_seg_results.py`: merges segmentation outputs into geospatial datasets.
- `src/annotation_scripts/merge_annotations.py`: parses annotation model text responses.
- `src/data_merge/final_data_merge.py`: merges corrected points with country-specific datasets and model outputs.
- The explicit implemented algorithmic model families are:
- `src/create_voronoi.py`: Voronoi approach `0`, approach `1`, and approach `2`; optional `only_round` mode.
- `src/config.yaml`: `weight_method` choices `linear`, `logarithmic`, `square_root`, `sigmoid`.
- `src/config.yaml`: `weight_func` choices `mult`, `add`, or empty.
- `src/config.yaml`: rigid buffering and dynamic buffering through `dynamic_buffering` and `dynamic_buffer_k`.
- `src/pop_at_risk_river_calculations/impact_polygons_pop.py`: a configurable downstream impact model with `org_per_pop`, `width`, `c_limit`, `base_k`, `theta`, `step_m`, `least_discharge_cms`, and `impact_radii`.

## 7. Training, Evaluation, Merging, And Visualization Stages

- Training stage: no explicit training stage or model-training code was found.

### Merging stages

- `src/data_merge/combine_locations.sh`: OSM correction, segmentation merge, and final WWTP merge.
- `src/combine_watersheds.sh`: watershed merge.
- `src/annotation_scripts/merge_annotations.sh`: annotation merge back into the WWTP dataset.

### Evaluation and validation stages

- `src/pop_validation_scripts/comparison.sh`: verification split, HydroWaste comparison, EU comparison.
- `src/sensitivity_analysis_scripts/compare_pop_sweep_hw_eu.sh`: cross-file HW and EU sensitivity diagnostics over all population-enriched outputs.
- `src/annotation_scripts/annotations_inspection.sh`: annotation QA sampling and class-distribution inspection.

### Visualization stages

- `src/figures_scripts/composite_area_population_plots.sh`: composite histogram and scatter diagnostics.
- `src/figures_scripts/piechart_figure.py`: static world map with donut markers.
- `src/figures_scripts/piechart_interactive.py`: interactive Folium summary map.
- `src/figures_scripts/pop_at_risk_figures.sh`: risk-result visualization.

### Generation stages feeding those outputs

- `src/download_pop.sh`
- `src/create_voronoi.sh`
- `src/add_pop.sh`
- `src/pop_at_risk_river_calculations/create_rasters.sh`
- `src/pop_at_risk_river_calculations/pop_differences_and_impact_polygons.sh`
- `src/pop_at_risk_river_calculations/find_pop_in_danger_pop.sh`
- `src/industrial_analysis/industrial_analysis.sh`

## 8. Parameter Sweep Functionality And Hyperparameter Exploration

- Sweep-generation entrypoints:
- `src/sensitivity_analysis_scripts/create_voronoi_param_sweep.sh`
- `src/sensitivity_analysis_scripts/add_pop_param_sweep.sh`
- `src/sensitivity_analysis_scripts/create_voronoi_param_sweep_parallel.sh`
- `src/sensitivity_analysis_scripts/industrial_analysis_sweep.sh`
- Shared explicit sweep grid in the shell scripts:
- levels: `6`, `7`, `8`, `9`
- weight funcs: `mult`, `add`, empty
- weight methods: `linear`, `logarithmic`, `square_root`, `sigmoid`
- rigid buffers: `9000`, `11000`, `13000`, `15000`
- dynamic k values: `0.6`, `0.7`, `0.8`
- Shared explicit sweep logic:
- rigid-buffer regime adds combinations with `dynamic_buffering=false` and each buffer value
- dynamic-buffer regime adds combinations with `dynamic_buffering=true`, buffer fixed at `9000`, and each dynamic k value
- when `weight_func` is empty, only the canonical `weight_method=linear` is kept to avoid redundant runs
- combinations are deterministically shuffled with `random.Random(seed).shuffle(combos)`
- combinations are sharded by modulo across 10 SLURM array tasks using `idx % 10 == task_id`
- `src/sensitivity_analysis_scripts/create_voronoi_param_sweep.sh`: runs `src/create_voronoi.py` with approach fixed to `1`
- `src/sensitivity_analysis_scripts/add_pop_param_sweep.sh`: discovers available Voronoi output files at runtime, then runs `src/add_pop.py` once per discovered file index for each assigned parameter combination
- `src/sensitivity_analysis_scripts/create_voronoi_param_sweep_parallel.sh`: delegates to `src/sensitivity_analysis_scripts/create_voronoi_parallel_sweep.py`, which further splits one task's assigned combinations across internal jobs and retries failed runs with progressively fewer parallel jobs
- `src/sensitivity_analysis_scripts/industrial_analysis_sweep.sh`: reuses the same grid and runs the two industrial-analysis stages per combination
- Sweep-result evaluation:
- `src/sensitivity_analysis_scripts/compare_pop_sweep_hw_eu.sh` is not a generator sweep; it evaluates all parseable population-enriched GPKGs, deduplicates empty-weight-func duplicates, computes HW and EU metrics, writes alias maps and ranking CSVs, and saves sensitivity-score figures

## 9. Dependencies Between Scripts

### Output and stage dependencies

- `src/data_merge/combine_locations.sh` produces the canonical WWTP datasets consumed by later WWTP-based stages, including `src/create_voronoi.py`, `src/annotation_scripts/grid_generation_and_osm_extract.sh`, and the industrial-analysis branch.
- `src/combine_watersheds.sh` produces the combined watershed dataset consumed by approach-1 Voronoi, industrial boundary enrichment, and river-risk stages.
- `src/annotation_scripts/merge_annotations.sh` writes annotation-derived fields back into the main corrected points layer that later WWTP stages use.
- `src/download_pop.sh` produces country TIFF inputs consumed by `src/add_pop.py` and `src/pop_at_risk_river_calculations/create_rasters.py`.
- `src/create_voronoi.sh` produces Voronoi layers consumed by `src/add_pop.sh`.
- `src/add_pop.sh` produces population-enriched Voronoi outputs consumed by `src/pop_at_risk_river_calculations/create_rasters.sh`, `src/pop_validation_scripts/comparison.sh`, the figure scripts, and `src/sensitivity_analysis_scripts/compare_pop_sweep_hw_eu.py`.
- `src/pop_at_risk_river_calculations/create_rasters.sh` feeds `src/pop_at_risk_river_calculations/find_unserved_pop.py`.
- `src/pop_at_risk_river_calculations/pop_differences_and_impact_polygons.sh` encodes an explicit serial dependency chain: `find_unserved_pop` -> `find_diff_pop` -> `assign_rivers_to_basin` -> `find_intersection_river` -> `impact_polygons_pop`.
- `src/pop_at_risk_river_calculations/find_pop_in_danger_pop.sh` depends on the impact outputs produced by the previous chain.
- `src/industrial_analysis/download_and_vectorize.py` writes the merged industrial layer consumed by `src/industrial_analysis/find_unconnected_industrial_areas.py`.

### Shared code dependencies

- `src/starter.py` is the central dependency for runtime configuration; nearly every executable module imports `load_config` and `parse_config_overrides`.
- `src/geo_utils.py` (spatial) and `src/utils.py` (domain-agnostic) are the canonical shared-helper modules. New shared logic belongs in one of them; `src/figures_scripts/_shared.py` is a narrower target for figure-layer column conventions.
- `src/pipelines.py` is the shared orchestration dependency for output-path generation, data preparation, and Voronoi execution; it is reused by `src/create_voronoi.py`, `src/industrial_analysis/find_unconnected_industrial_areas.py`, `src/figures_scripts/composite_area_population_plots.py`, `src/figures_scripts/piechart_figure.py`, `src/figures_scripts/piechart_interactive.py`, `src/pop_at_risk_river_calculations/create_rasters.py`, and `src/sensitivity_analysis_scripts/create_voronoi_parallel_sweep.py`.
- `src/create_voronoi.py` also acts as a shared utility module; helpers from it are imported by `src/download_pop.py`, `src/combine_watersheds.py`, the data-merge scripts, annotation scripts, validation scripts, industrial-analysis scripts, and population-at-risk scripts.
- `src/annotation_scripts/merge_annotations.py` provides `decode_gen_text`, which is reused by `src/annotation_scripts/annotations_inspection.py`.
- `src/pop_validation_scripts/hw_comparison.py` provides `ndvi`, `multiples`, and `replace_inf`, which are reused by `src/pop_validation_scripts/eu_comparison.py` and `src/sensitivity_analysis_scripts/compare_pop_sweep_hw_eu.py`.
- `src/data_merge/merge_seg_results.py` provides `assign_to_nearest`, which is reused by `src/pop_validation_scripts/eu_comparison.py` and `src/sensitivity_analysis_scripts/compare_pop_sweep_hw_eu.py`.