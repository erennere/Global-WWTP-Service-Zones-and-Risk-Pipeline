# Test Expansion Prompt

Use this prompt when continuing test development for `wwtp-service-pipeline`.

## Prompt

You are extending the automated test suite for the `wwtp-service-pipeline` repository.

Work from the repository as it exists now. Do not invent architecture that is not present in the code. Keep the test strategy aligned with the current synthetic-fixture approach already established under `src/tests`.

### Current test harness

- Root pytest config lives in `pytest.ini`.
- Tests live under `src/tests`.
- Current markers are `unit`, `integration`, and `slow`, and `--strict-markers` is enabled.
- Shared fixtures live in `src/tests/conftest.py`.
- The current suite passes with 159 tests (`pytest --tb=short -q`).
- Core shared fixtures currently include:
  - `tiny_points_gdf`
  - `sample_sites_gdf`
  - `tiny_watershed_gdf`
  - `tiny_country_gdf`
  - `tiny_population_array`
  - `mock_cfg`
  - `mock_rivershed_gdf`
- Current design principle: use tiny synthetic geometries and mocked I/O for unit tests; use real small geometry operations and `rasterio.MemoryFile` rasters for integration tests.

### Current tested surface

The existing suite already covers a meaningful amount of real repository behavior:

- `starter.py`
  - CLI override normalization
  - path normalization
  - config loading and alias expansion
  - dynamic buffer token handling
  - invalid `weight_method`
- `pipelines.py`
  - configured callable resolution
  - output-path builders
  - mean-2-nearest-neighbor helper
  - `run_voronoi_approach` skip/success/write/error branches
- `create_voronoi.py`
  - `calculate_area`
  - default distance functions
  - `cluster_point_indices`
  - `calculate_buffer`
  - `weighted_voronoi` empty-input, missing-site-id, UTM-estimation-failure, buffer-length-mismatch, and single-site clipping branches
  - `initialize_voronoi_weights`
  - `create_weights`
  - grouped normalization inside `orchestrate_voronoi_weights`
  - output/checkpoint resume and overwrite behavior in `orchestrate_voronoi_weights`
  - `_filter_requested_approaches` and `city_voronoi` gating
- `data_merge`
  - `final_data_merge.cluster_point_indices`
  - `final_data_merge.cluster_points`
  - `final_data_merge.find_unmatched_targets`
  - `final_data_merge.get_best_points`
  - `final_data_merge.find_safe_epsg`
  - `final_data_merge.find_meter_coordinates`
  - `final_data_merge.main` orchestration wiring and output writing
  - `merge_seg_results.assign_to_nearest`
  - `merge_seg_results.merge_new`
  - `merge_seg_results.merge_old`
  - `merge_seg_results.main` variant dispatch (`old`/`new`) with legacy guard behavior
  - `correct_locations_w_OSM.coordinate_corr_locations_wOSM`
  - `correct_locations_w_OSM.main` output-writing and country-enrichment wiring branches
- `industrial_analysis.download_and_vectorize`
  - `_find_raster_dirs`
  - `vectorize_rasters_parallel` no-raster branch
  - `_vectorize_and_merge` no-dir and empty-vectorization failure branches
  - `main` cached-vector branch and enrichment-failure branch
- `add_pop.py`
  - `find_country_tif_files`
  - `find_newest_country_tif_files`
  - `intersect_single_file` (negative-stat clipping and no-year short-circuit)
  - `intersect_all_files` (country dispatch and empty-result branch)
  - `orchestrate_intersections` (index validation and output-writing flow)
  - `main` (config wiring and non-integer index handling)
- `pop_at_risk_river_calculations`
  - `find_diff_pop.find_difference`
  - `find_diff_pop.parse_args`
  - `find_diff_pop.find_differences` serial and parallel branches
  - `find_diff_pop.process_epsg_group`
  - `find_diff_pop.main` happy-path wiring and input/index error branches
  - `find_unserved_pop.create_unserved_pop` happy, missing-file, empty-result, and missing-geometry branches
  - `find_unserved_pop.main`
  - `find_diff_pop.parse_bool`
  - `find_pop_in_danger_pop.assign_tile_to_df`
  - `find_pop_in_danger_pop.group_tile_population_sums`
  - `find_pop_in_danger_pop.rename_cols`
  - `find_pop_in_danger_pop.find_tiles_in_countries`
  - `find_pop_in_danger_pop.main` geometry-preservation and empty-input-output branches
  - `impact_polygons_pop.calculate_load_ratio`
  - `impact_polygons_pop.generate_single_segment_plume`
  - `impact_polygons_pop.create_impact_polygons`
  - `impact_polygons_pop.get_runtime_params`
  - `impact_polygons_pop.create_dicts`
  - `impact_polygons_pop.orchestrate_logic`
- `industrial_analysis.find_unconnected_industrial_areas`
  - `load_industrial_areas`
  - `load_wwtps` basin-enrichment branch
  - `filter_industrial_wwtps`
  - `run_voronoi_for_wwtps` approach `0` and `1`
  - invalid-approach handling
  - `find_unconnected_areas`
  - `main()` skip/no-industrial/failure branches
- small synthetic integration coverage for:
  - config/path alignment
  - population enrichment with `MemoryFile`
  - weighted Voronoi geometry validity and clipping
  - industrial raster vectorization
- low-priority helper coverage started for:
  - `pop_validation_scripts.verification_script.find_verification_watersheds`
  - `pop_validation_scripts.hw_comparison` (`ndvi`, `multiples`, `replace_inf`, `extract_voronoi_parameters`)
  - `pop_validation_scripts.verification_script.main` split-and-write orchestration branches
  - `pop_validation_scripts.hw_comparison.main` gpkg discovery and dispatch wiring
  - `pop_validation_scripts.eu_comparison.main` nearest-assignment/filter/dispatch wiring
  - `figures_scripts.composite_area_population_plots` helper utilities
  - `annotation_scripts.merge_annotations` (`decode_gen_text`, `parse_idx_from_image_name`, and `main` wiring/missing-column guard)
  - `annotation_scripts.annotations_inspection` helper + main wiring/missing-column branches
  - `annotation_scripts.copy_falsy_images.main` selection/copy and missing-file branches

Do not spend time re-testing those exact paths unless a failing behavior or uncovered branch makes it necessary.

### Priority gaps to cover next

Focus next on uncovered Python logic with the best value-to-complexity ratio, in this order:

1. `src/create_voronoi.py`
   - `weighted_voronoi` worker-path branches not covered by the integration tests
   - clipping/buffering edge cases
   - contour-extraction fallbacks
   - direct CLI/main execution gating only after helper branches are exhausted
2. `src/add_pop.py`
  - any still-uncovered warning/error branches in `main` or `orchestrate_intersections` after validating with `--cov-branch`
3. `src/pop_at_risk_river_calculations/find_diff_pop.py`
  - any residual failure/warning branches not exercised by current unit coverage
4. `src/pipelines.py`
  - low-priority cleanup only: residual guard branches in `prepare_data` / `run_voronoi_approach`
  - note: this module currently has no top-level `main()` orchestration entrypoint
5. `src/data_merge/correct_locations_w_OSM.py`
  - remaining main-path wiring branches not yet exercised directly

These stay lower priority during the first waves, but they are still in-scope for completion once the higher-priority gaps are stable:

- `annotation_scripts/*`
- `figures_scripts/*`
- `pop_validation_scripts/*`
- one-off wrapper scripts whose main value is shell orchestration rather than reusable logic

### Completion phase (includes low-priority modules)

After the priority list is handled, continue until the remaining low-priority Python logic is covered with lightweight, behavior-focused tests.

Order for final completion:

1. `src/pop_validation_scripts/*`
  - prefer parser/transform helpers and deterministic comparison logic
  - avoid image-rendering or report-format assertions unless they encode business rules
2. `src/figures_scripts/*`
  - test data-shaping helpers and guard branches
  - do not overfit tests to plot style internals
3. `src/annotation_scripts/*`
  - prioritize pure Python validation/merge/filter logic
  - keep network, download, and subprocess paths mocked
4. wrapper-like script entry points
  - only add tests where CLI argument handling, config wiring, or branch gating has meaningful logic
  - skip trivial wrappers that only call one function with no branching

Stop only when all remaining uncovered branches are either:
- tested; or
- explicitly documented as intentionally out-of-scope (with reason) in `src/tests/README.md`.

### Important repo-specific note

The config key `city_voronoi` is now enforced in the approach-selection path of `create_voronoi.py`.

That means:

- approach `2` should be skipped when `city_voronoi=False`;
- any change to approach filtering should keep that behavior covered by tests.

### Fixture rules

- Do not read from `data/` in tests.
- Prefer extending `src/tests/conftest.py` only when a fixture will be reused by multiple modules.
- Keep fixtures synthetic and tiny.
- Use `mock_cfg` instead of hardcoding any path from `config.yaml`.
- If you need new raster coverage, use `rasterio.MemoryFile` or small temporary rasters under `tmp_path`.
- If you need new tabular inputs, build them in memory with `pandas`.

### Mocking rules

- Mock all network calls:
  - `requests`
  - S3/Overture access
  - remote DuckDB access
- Mock all subprocess or SLURM behavior.
- Mock file reads and writes in unit tests:
  - `geopandas.read_file`
  - `GeoDataFrame.to_file`
  - `pandas.read_csv`
  - `rasterio.open`
  - `duckdb.sql` / `duckdb.connect` when the test is not explicitly about those internals
- Do not mock core `geopandas`/`shapely` spatial operations if the behavior under test is geometric.
- Integration tests may use real small geometry operations and `MemoryFile` rasters.

### Test style rules

- Keep tests branch-focused and behavior-focused.
- Use `pytest.mark.unit` for isolated, mocked tests.
- Use `pytest.mark.integration` for multi-function flows with real small geometries or in-memory rasters.
- Only use `pytest.mark.slow` if the test is materially heavier than the current suite.
- Prefer one clear assertion cluster per behavior.
- Do not add trivial tests.
- Do not test shell wrappers (`.sh`).
- Do not rely on external downloads or large repo assets.

### Validation rules

After each edit slice:

1. run the narrowest affected tests first;
2. if they pass, run the full suite with:

```bash
pytest --tb=short -q
```

3. clean generated artifacts if the run created any, especially:
   - `add_pop.log`
   - `.pytest_cache`
   - `src/tests/**/__pycache__`
   - any temporary output written by new tests

### Deliverables

When extending the suite:

- add only the tests needed for the current target slice;
- update `src/tests/README.md` if you introduce a reusable fixture, a new testing pattern, or a new category of integration test;
- keep changes minimal and local;
- if a test reveals a real code defect, fix the code at the root cause and keep the test;
- if a requested behavior is not implemented in the repo, call that out explicitly instead of pretending it exists.

### Suggested immediate next slice

Start with `create_voronoi.py` worker-path and fallback branches.

Concrete next targets:

- `weighted_voronoi`
  - worker branch behavior with mocked pool/executor boundaries
  - failure handling when one worker yields invalid or empty geometry
- contour and clipping fallbacks
  - branch behavior when contour extraction returns no valid polygons
  - clipping fallback when clipping geometry is empty or invalid
- `_filter_requested_approaches` / approach dispatch interplay
  - keep `city_voronoi=False` skip behavior locked while testing mixed requested-approach inputs

For that slice, keep the same strategy: synthetic geometries, mocked file discovery and write calls, no `data/` dependencies, and no external downloads.

### Suggested follow-through slices after that

1. `create_voronoi.weighted_voronoi` worker-path and fallback branches
2. remaining `data_merge` orchestration/main flows
3. `industrial_analysis/download_and_vectorize.py` beyond current raster-vectorization integration coverage
4. completion-phase low-priority modules (`pop_validation_scripts`, `figures_scripts`, `annotation_scripts`) using tiny synthetic fixtures and mocked I/O