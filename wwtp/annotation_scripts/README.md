# annotation_scripts

## Sections
| Section | Purpose |
| --- | --- |
| `What This Module Does` | Explains the aim of the annotation stage |
| `How It Fits In` | Shows where this stage sits in the wider workflow |
| `Scripts in This Folder` | Summarises the role, inputs, and outputs of each script |
| `Execution Flow` | Gives a compact visual view of the stage |
| `Run Instructions` | Lists the commands most users will actually run |
| `Smart Behaviors` | Notes implementation details that matter in practice |
| `Parameters` | Collects the configuration settings specific to this stage |
| `Known Issues / TODOs` | Flags current caveats and limitations |

## What This Module Does
This module prepares and merges annotation context around WWTP points. It creates grids, extracts OSM features, renders annotation imagery, and then merges the annotation results back into the corrected WWTP dataset.

## How It Fits In
It runs after the main WWTP merge and before the Voronoi and population stages when annotation-derived context is needed. Its outputs are used for QA, downstream filtering, and explanation of WWTP type labels.

## Required Starter Data Files

Before running this stage, these inputs must be available.

| Data file | Why needed | Default path status |
| --- | --- | --- |
| Corrected and merged WWTP layer | points for grid generation | produced by the `combine_locations` rule (`wwtp/data_merge/`) |
| Satellite imagery directory | source for annotation download | `download_bing_annotate.paths.annotations_images_dir` (cluster-specific path by default) |
| Annotation inference results | CSV with AI labels for merge-back | `merge_annotations.paths.annotations_results_filepath` (cluster-specific path by default) |

Cluster-specific defaults for imagery and annotation outputs must be overridden in `wwtp/config.yaml` for local execution. See `download_bing_annotate.paths.*` and `merge_annotations.paths.*` in config.

## Scripts in This Folder
| Script | Role | What it does | Key inputs | Key outputs |
| --- | --- | --- | --- | --- |
| `NEW_01_GENERATEGRIDS.py` | Python worker | Builds annotation grids around corrected WWTP points | corrected WWTP layer, grid parameters | grid GeoPackages |
| `NEW_02_EXTRACTOSMDATAFULL_GEOJSON.py` | Python worker | Extracts OSM context for each grid | grid files, OSM settings, retries | per-grid OSM GeoJSON |
| `download_bing_annotate.py` | Python worker | Draws annotation overlays on pre-downloaded imagery | imagery directory, OSM context, tile ids | annotated image assets |
| `merge_annotations.py` | Python worker | Parses annotation text and merges labels into WWTP points | annotation CSV, annotated images, corrected layer | updated corrected-all GeoPackage |
| `annotations_inspection.py` | Python worker | Builds class summaries and inspection samples | annotated images and annotation results | QA sample set |
| `copy_falsy_images.py` | Python worker | Copies a filtered image subset for QA | annotated images and selection criteria | copied image subset |
| `NEW_03_WASTEWATERJOIN_GEOJSON.py` | Python worker | Optional join helper for GeoJSON/parquet intermediates | OSM outputs and temp parquet files | joined geodata |
| `NEW_04_EXPORTGEOTIFF.py` | Python worker | Optional helper: renders `export_layer_filepath` into one image + `.wld` world file per grid cell (`python -m wwtp.annotation_scripts.NEW_04_EXPORTGEOTIFF`) | NEW_01 grid, one vector layer | images in `export_patch_dir` |
| `sample_annotations_by_class.py` | Python worker | Ad-hoc class-balanced sampling of annotated images | annotation results and annotated image directory | sampled image list (not yet wired into the pipeline - see Known Issues) |

## Execution Flow
```mermaid
graph TD
  A[(merged plants)] --> B[NEW_01_GENERATEGRIDS]
  B --> C[NEW_02_EXTRACTOSMDATAFULL_GEOJSON]
  C --> D[NEW_03_WASTEWATERJOIN_GEOJSON]
  D --> E[download_bing_annotate x workflow.bing_annotate_shards]
  E --> F[(annotated images)]
  F -. external inference .-> G[merge_annotations]
  G --> H[(annotation-enriched WWTP layer)]
```
Snakemake rules: `annotation_grids` (NEW_01 + NEW_02), `wastewater_join` (NEW_03), `download_bing_annotate` (one job per shard), `merge_annotations`; target `annotations`.

## Run Instructions
```bash
snakemake --profile workflow/profiles/slurm annotations     # the whole chain, shards as separate jobs
python -m wwtp.annotation_scripts.merge_annotations           # one stage on its own
python -m wwtp.annotation_scripts.annotations_inspection      # QA sampling (not part of the workflow)
python -m wwtp.annotation_scripts.copy_falsy_images
```

Job logs are in `{data_dir}/workflow/logs/`.

## Smart Behaviors
- `annotations_inspection.py` uses `annotations.n_sample_size` with a default fallback of `1000` and saves figures in headless runs instead of relying on an interactive display.
- `merge_annotations.py` parses the image index defensively from filenames and drops any existing annotation columns before re-merging to avoid duplicate `_x`/`_y` columns.
- `download_bing_annotate.py` reads `annotations_images_dir` and `annotated_images_output_dir` from config, so local imagery paths can be swapped without touching code.
- `download_bing_annotate` runs as `workflow.bing_annotate_shards` jobs; each takes its share of the tiles (seeded shuffle, `annotations.random_seed`).

Delete the grid, annotation, or merged outputs to force a rerun of the corresponding stage.

## Parameters
| Config key | Default | Effect |
| --- | --- | --- |
| `NEW_01_GENERATEGRIDS.annotations.cell_size` | `3072` | Grid cell size |
| `NEW_01_GENERATEGRIDS.annotations.factor` | `1.194` | Grid expansion factor |
| `NEW_02_EXTRACTOSMDATAFULL_GEOJSON.annotations.max_workers` | `8` | OSM query parallelism |
| `NEW_02_EXTRACTOSMDATAFULL_GEOJSON.annotations.overwrite` | `false` | Overwrite existing OSM outputs |
| `NEW_02_EXTRACTOSMDATAFULL_GEOJSON.annotations.retries` | `5` | OSM retry count |
| `NEW_02_EXTRACTOSMDATAFULL_GEOJSON.annotations.overpass_urls` | list | Rotating Overpass endpoint list |
| `NEW_02_EXTRACTOSMDATAFULL_GEOJSON.annotations.overpass_pause_seconds` | `0.1` | Pause between task-batch submissions |
| `annotations_inspection.annotations.n_sample_size` | `1000` | QA sample size |
| `download_bing_annotate.annotations.random_seed` | `42` | Sampling/splitting seed, declared here because this is its earliest consumer |
| `annotations_inspection.annotations.random_seed` | null | Inherits the seed above so QA sampling and the imagery split stay aligned |
| `download_bing_annotate.annotations.zoom_level` | `17` | Imagery zoom level |
| `download_bing_annotate.annotations.image_size_px` | `3072` | Requested imagery width/height in pixels |
| `download_bing_annotate.annotations.max_workers` | `64` | Parallel bbox annotation worker count |
| `download_bing_annotate.annotations.georeferenced` | `false` | Write georeferenced GeoTIFF output instead of resized PNG |
| `download_bing_annotate.annotations.fontsize` | `24` | Label font size in rendered outputs |
| `download_bing_annotate.annotations.dpi` | `72` | PNG save DPI |
| `download_bing_annotate.annotations.target_size_px` | `1024` | Final PNG output width/height |
| `download_bing_annotate.annotations.earth_radius_m` | `6378137` | Earth radius used for Mercator math |
| `download_bing_annotate.annotations.mercator_tile_size_px` | `256` | Web-Mercator base tile size |
| `download_bing_annotate.annotations.imagery_reference_tile_size_px` | `512` | Imagery provider native tile size used in pixel scaling |
| `download_bing_annotate.paths.annotations_images_dir` | ⚠️ path | Source imagery directory |
| `download_bing_annotate.paths.annotated_images_output_dir` | ⚠️ path | Annotated image output directory |
| `merge_annotations.paths.annotations_results_filepath` | ⚠️ path | Annotation results CSV |
| `annotations_inspection.paths.annotations_verf_image_outpath_dir` | ⚠️ path | QA image output directory |

## Known Issues / TODOs
- `NEW_03_WASTEWATERJOIN_GEOJSON.py` is part of the workflow (rule `wastewater_join`; `download_bing_annotate` reads its merged layers). `NEW_04_EXPORTGEOTIFF.py` is an optional helper outside it.
- `sample_annotations_by_class.py` (QA sampling) and `annotations_inspection.py` / `copy_falsy_images.py` run by hand; they are not rules of the workflow.
- No explicit `TODO` or `FIXME` markers were found in this module.
