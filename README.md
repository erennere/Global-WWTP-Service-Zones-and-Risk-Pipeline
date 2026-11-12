# Global-WWTP-Service-Zones-and-Risk-Pipeline

The repository is a pipeline developed to explore whether AI can fill data gaps regarding Wastewater Treatment Plants (WWTPs) globally, for example the population served and the type of WWTP (residential, industrial, etc.). The core idea is to approximate WWTP service areas via Voronoi tessellations, provided that WWTP coverage is adequate enough that cells do not erroneously extend into areas served by uncovered WWTPs absent from the dataset.

WWTPs are not just point infrastructure in this project: they are spatial service units whose catchments can be approximated, refined, and compared against reference datasets. This pipeline is designed to run at multi-country scale and combines correction, harmonization, Voronoi allocation, population attribution, river-linked risk propagation, validation, industrial coverage analysis, and reporting. Most runtime behavior is controlled from `wwtp/config.yaml`; a Snakemake workflow (`workflow/Snakefile`) runs every stage on one machine or as SLURM jobs.

## Table of Contents

- [Data Sources](#data-sources)
- [Approaches](#approaches)
- [Weighted Voronoi Tessellation](#weighted-voronoi-tessellation)
- [Clipping / Buffering](#clipping--buffering)
- [Analyses](#analyses)
- [What This Repository Contains](#what-this-repository-contains)
- [Running the Pipeline](#running-the-pipeline)
- [Project Data Flow](#project-data-flow)
- [Environment Creation](#environment-creation)
- [Starter Data Files You Need Before Running](#starter-data-files-you-need-before-running)
- [Checks: pre-commit and CI](#checks-pre-commit-and-ci)
- [Configuration Parameters You Usually Need To Change](#configuration-parameters-you-usually-need-to-change)
- [What Is Missing Or External To The Repository](#what-is-missing-or-external-to-the-repository)
- [Section Documentation](#section-documentation)

## Data Sources

### Primary WWTP Dataset: HydroWASTE
HydroWASTE is a spatially explicit global database of 58,502 WWTPs and their characteristics, developed by combining national and regional datasets with auxiliary information to derive or complete missing attributes including population served, effluent flow rate, and treatment level.

Reference:
Ehalt Macedo, H., Lehner, B., Nicell, J., Grill, G., Li, J., Limtong, A., Shakya, R. (2022). Distribution and characteristics of wastewater treatment plants within the global river network. Earth System Science Data, 14(2): 559-577. https://doi.org/10.5194/essd-14-559-2022

A drawback of HydroWASTE is that some points represent the discharge location rather than the WWTP itself. To address this, the repository uses a pre-corrected file configured as `correct_locations_w_OSM.paths.paul_corrected_filepath` in `wwtp/config.yaml`.

The remaining locations are scanned in OpenStreetMap within a 5 km radius (`correct_locations_w_OSM.rad = 5000`). OSM geometry is assigned if found, otherwise the location is discarded.

### European Dataset: Waterbase-UWWTD
A similar location-correction approach is applied to the European dataset. The Waterbase-UWWTD (Urban Waste Water Treatment Directive) dataset, provided by the European Environment Agency (EEA), includes data on individual WWTPs and collecting systems: their localisation, capacity and actual load treated, type of treatment, and aggregated performance data.

Reference:
European Environment Agency. Waterbase - UWWTD: Urban Waste Water Treatment Directive - reported data. https://www.eea.europa.eu/data-and-maps/data/waterbase-uwwtd-urban-waste-water-treatment-directive

Extra points were also added for the United States, Canada, Germany, and Thailand.

### Watershed Basins: HydroBASINS / HydroSHEDS
Using Voronoi tessellation alone does not account for river topology. It is costly to transport water across different watersheds, so WWTPs are assigned to watershed basins before analysis.

HydroSHEDS is a mapping product providing hydrographic information for regional and global-scale applications in a consistent format, offering geo-referenced datasets at various scales including river networks, watershed boundaries, drainage directions, and flow accumulations; it is based on elevation data obtained in 2000 by NASA's Shuttle Radar Topography Mission (SRTM). HydroBASINS provides polygons of nested, hierarchical watersheds ranging from level 1 to level 12, using Pfafstetter codes.

Reference:
Lehner, B., Grill, G. (2013). Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems. Hydrological Processes, 27(15): 2171-2186. https://doi.org/10.1002/hyp.9740

HydroBASINS Technical Documentation:
https://data.hydrosheds.org/file/technical-documentation/HydroBASINS_TechDoc_v1c.pdf

HydroBASIN zip files for a level `X` should be placed under `data/hydroshed_river_levels/lvl{X}`. The level is a free parameter: higher levels group more WWTPs together, while lower levels increase individual differences and reduce inter-WWTP interaction.

### Population Data: WorldPOP
WorldPOP 100 m rasters from 2014 to 2024 are downloaded and saved per country. Each Voronoi cell is intersected with the raster to obtain population estimates.

## Approaches

Three approaches are implemented. In all cases, Voronoi cells are created for each buffer and set of points individually.

- Approach 1 builds buffers around WWTPs, then groups and dissolves intersecting buffers. The dissolved buffer serves as the basin for the WWTPs therein.
- Approach 2 groups WWTPs based on a buffer layer, in this repository watershed boundaries from HydroBASINS.
- Approach 3 groups WWTPs based on city identification. Cities are buffered; the population inside is divided among WWTPs within using Voronoi cells. It exists in the codebase but is not the main production path.

Although the present application is wastewater infrastructure, the methodological structure is not specific to WWTPs. The framework can be transferred to other infrastructure domains whenever the analytical problem is to approximate a service area, allocate an exposed or served population, and compare competing facilities under spatial constraints. In practical terms, such a transfer usually requires changing five elements rather than redesigning the whole pipeline: the point layer representing service-providing sites, the boundary or grouping layer that constrains plausible service exchange, the weighting proxy that represents relative service capacity or attractiveness, the population or exposure surface being allocated, and the validation dataset used to assess realism.

For example, the same workflow could be adapted to hospitals, schools, health posts, fire stations, solid-waste facilities, warehouses, or other distributed service infrastructure. In a healthcare setting, WWTP points would be replaced by hospital or clinic locations, watershed boundaries could be replaced by administrative regions, catchments, travel sheds, or road-constrained service zones, and the weighting term could be derived from bed count, staffing, floor area, or treatment capacity instead of lagoon area or pond count. In an education setting, facilities could be schools, the population surface could be school-age population rather than total population, and validation could be performed against enrolment or district planning data. The same logic applies to emergency response, logistics, and other networked public-service systems: the geometry engine remains similar, while the domain-specific meaning of sites, constraints, and weights changes.

The important methodological point is that the repository separates the generic allocation machinery from the domain-specific inputs. The Voronoi engine, buffering logic, weighting transforms, and downstream population intersection are reusable components; what changes from one application to another is the interpretation of the inputs and the choice of constraint layer. This is why the repository exposes configurable hooks such as `create_voronoi.calculate_area_fn`, `create_voronoi.calculate_buffer_fn`, and `create_voronoi.prepare_data_fn`: they allow the same spatial allocation framework to be retuned for alternative service systems without rewriting the core geometry pipeline.

## Weighted Voronoi Tessellation

Standard Voronoi tessellation treats all locations equally, which does not reflect real differences between WWTPs. A custom weighted distance function is implemented. The code creates an xy-plane with resolution `n_steps` meters between points and checks which area of influence each point falls into.

Three distance functions are available:
- Euclidean distance.
- Multiplicative Euclidean.
- Additive Euclidean.

Weights are normalized within each buffer and computed from ML-derived tags such as the number of ponds and total treatment pond area, combined as `total_area * sqrt(number_of_ponds)`.

Weight transformation functions are controlled by `create_voronoi.weight_method` and currently supported values are:
- `size` (default): sqrt of the plant size in people (`create_voronoi.size_measure`); needs `weight_func: mult`
- `linear`
- `square_root`
- `logarithmic`
- `sigmoid`

## Clipping / Buffering

To avoid disproportionately large Voronoi cells, especially in sparse areas, cells are clipped to a buffer around each location after creation.

- Static buffering: disable dynamic buffering with `create_voronoi.dynamic_buffering: false`. The `--buffer` flag overrides the config value at runtime.
- Dynamic buffering: enable with `create_voronoi.dynamic_buffering: true`. Each plant's radius grows with its size in people (`create_voronoi.size_measure`: `reported` or `footprint`; numbers in `create_voronoi.radius_calibration`).

The sweep analysis scripts support sensitivity analysis across combinations of model parameters. The configurable hooks that govern those runs are `create_voronoi.calculate_area_fn`, `create_voronoi.calculate_buffer_fn`, and `create_voronoi.prepare_data_fn`, which are resolved at runtime through `wwtp/pipelines.py`.

## Analyses

### Population Validation
Calculated population estimates are compared against HydroWASTE, using only locations where `QUAL_POP == 1` from official sources, and against European Waterbase-UWWTD data, which reports Population Equivalent rather than actual observed population.

Three validation tiers are differentiated:
- Basins with a single WWTP.
- Basins with multiple sources where at least a threshold fraction of plants appear in both datasets.
- Basins with multiple sources regardless of threshold coverage.

For both datasets, a Normalized Difference Index and a linear regression are produced.

### Population at Risk
This branch propagates organic material from unserved settlements downstream.

The workflow identifies non-served population outside WWTP service areas, assigns river segments to basins, links non-served polygons to nearby rivers, propagates organic load downstream until concentration falls below `impact_polygons_pop_params.c_limit`, and then intersects the resulting impact corridors with WorldPOP.

### Industrial Area Analysis
For this analysis, a 10 m industrial land dataset from Zenodo is downloaded and vectorized. Industrial areas are assigned to watersheds and compared against Voronoi coverage generated from industrial WWTPs selected through the annotation-derived category fields. Industrial areas not served by any such WWTP are identified and reported.

The remaining sections shift from method to use: they first show how the repository is organised, then outline the standard execution order and end-to-end data flow, and finally summarise the setup and configuration details needed to run the workflow.

## What This Repository Contains

The repository is organized around one canonical processing chain and several optional analysis or reporting branches.

| Area | Location | Purpose |
| --- | --- | --- |
| Runtime configuration | `wwtp/config.yaml` | Global parameter and path control |
| Config resolver | `wwtp/starter.py` | Resolves inherited config values and named `--level`/`--version`/... overrides |
| Main technical index | `wwtp/README.md` | Stage index and cross-stage settings |
| Merge stage | `wwtp/data_merge/` | Build the canonical WWTP dataset |
| Annotation stage | `wwtp/annotation_scripts/` | Generate grids, OSM context, and annotation merges |
| Workflow | `workflow/Snakefile`, `workflow/profiles/`, `wwtp/workflow.py` | Run the stages in order, locally or on SLURM; sweeps |
| Voronoi stage | `wwtp/create_voronoi.py` (command line), `wwtp/service_areas/` (size, weights, solver, contours, joins, overlaps, orchestrate) | Create service areas |
| Population stage | `wwtp/add_pop.py` | Attach WorldPOP counts to service areas |
| Risk stage | `wwtp/pop_at_risk_river_calculations/` | Compute non-served population and downstream risk |
| Validation stage | `wwtp/pop_validation_scripts/` | Compare outputs against HydroWASTE and Waterbase-UWWTD |
| Industrial stage | `wwtp/industrial_analysis/` | Measure industrial land not covered by eligible WWTPs |
| Figures stage | `wwtp/figures_scripts/` | Build maps, plots, and publication outputs |
| Sweep stage | `wwtp/sensitivity_analysis_scripts/` | Validate the parameter sweep (plants: real catchments; countries: JMP, Eurostat, OECD) and choose a candidate |

Each major processing directory has its own README that explains the role of the stage, the purpose of each script, the local data flow, and the configuration settings that matter for that part of the pipeline.

## Running the Pipeline

Snakemake runs every stage through its own command line (`python -m wwtp.<stage>`) and knows each stage's input and output files, so it runs only what is missing or out of date, in order, in parallel where it can. From the repository root, in the pipeline's environment (`pip install -e ".[workflow]"`):

A run does only the part it names; `snakemake` alone lists the targets and runs nothing. `--profile workflow/profiles/local` runs on this machine, `--profile workflow/profiles/slurm` as SLURM jobs (one job per stage and shard); `-n` only shows the plan.

```bash
snakemake --profile workflow/profiles/slurm service_areas  # configured variant: service areas, population layers, HydroWASTE/UWWTD comparisons
snakemake --profile workflow/profiles/slurm risk       # configured variant, the rest: non-served population, river risk, population differences, industrial areas
snakemake --profile workflow/profiles/slurm analysis   # service_areas + risk
snakemake --profile workflow/profiles/slurm figures    # its figures (plus whatever of the analysis they still need)
snakemake --profile workflow/profiles/slurm sweep      # whole sweep grid: service areas + population, validation and the recommended candidate
snakemake --profile workflow/profiles/slurm national_data   # downloads the validation references and national plant registers (login node); lists the register plants our list lacks
snakemake --profile workflow/profiles/slurm all        # analysis + figures + sweep
snakemake --profile workflow/profiles/slurm fig1       # Figure 1 from its prepared inputs
snakemake --profile workflow/profiles/slurm annotations --config run_once=annotations  # grids, OSM context, images, labels
snakemake --profile workflow/profiles/slurm connection_radius   # calibration inputs: national connection statistics + served-share curves per country
```

The sweep validation's reference files (GB catchments, CBS, EPA sewersheds, JMP, Eurostat, OECD) are downloaded by the local rule `national_data_fetch` (URLs in `national_data.sources`); it needs internet on the node that runs Snakemake and keeps files that are already in `data/validation_data/`. `national_data` also writes `data/validation_data/national_data/new_plants.gpkg` (plants of the national registers farther than `match_distance_m` from every plant of ours, one file for all countries), `peek.md` (structure of every downloaded file) and `results.zip`.

`sweep_layers` builds only the service areas + population of the sweep grid, without the validation (`sweep_validation`: plants against GB and NL catchments, countries and SDG regions against JMP, Eurostat and OECD; see `wwtp/sensitivity_analysis_scripts/README.md`; result in `data/sensitivity/sweep_validation/summary.md`). Approach 0 never reads the basins, so the sweep computes it once per weighting and extent (at the configured level) and hard-links those files to the other levels' names (`share_approach0_pop__*`): 144 instead of 216 `create_voronoi` and `add_pop` jobs. The linked layers keep the shared level's `HYBAS_ID`/`basin_area` columns; the sweep validation scores approach 0 once and takes the groups from the configured level.

- The configured variant is the one `create_voronoi` sets in `wwtp/config.yaml` (level, version, buffer, weighting, size measure); the sweep grid is the `sweep` section. Change either there or in an overlay, not on the command line, so the files Snakemake expects match what the stages write.
- One-time stages are never part of a run unless named: `combine_locations`, the image annotation (`annotation_grids`, `wastewater_join`, `download_bing_annotate`, `merge_annotations`, together `annotations`) and `download_pop`. A run takes their outputs as they are (`all_annotated_v2.gpkg`, the WorldPop rasters behind `{data_dir}/workflow/stamps/download_pop.done`) and stops before submitting anything if one is missing. Include them for one run with `--config run_once=annotations,download_pop` after the targets (`--config` takes every word that follows it, so `snakemake analysis --config run_once=merge_annotations`), or for good in `workflow.run_once`; forcing them (`-R annotation_grids`) also needs them named. If the WorldPop rasters are already in place without the stamp, `touch` it.
- `analysis` (= `service_areas` + `risk`) builds, for the configured variant: basins, service areas (approaches 0 and 1, and 2 with `city_voronoi`), population layers, the non-served population (`create_rasters` as `workflow`-configured country shards plus a merge), rivers, impact polygons, population at risk, the population differences, the HydroWASTE/UWWTD comparisons and unconnected industrial areas. `figures` draws the variant's figures; `fig1` stays out of `all` because its inputs are prepared by hand.
- SLURM threads, memory and runtime per stage are in `wwtp/config.yaml`, section `workflow.resources`; job logs go to `{data_dir}/workflow/logs/`. Stages whose outputs cannot be listed in advance (downloads, plots) leave a marker in `{data_dir}/workflow/stamps/`.
- On outputs made before the workflow existed (by the former shell wrappers), run `snakemake --touch` once: it marks them as up to date instead of recomputing them.
- A single stage still runs on its own, e.g. `python -m wwtp.create_voronoi --approach 1 --weight-func add` for a one-off additive run.

## Project Data Flow

```mermaid
graph TD
	A[(HydroWASTE and country additions)] --> B[combine_locations]
	B --> C[(Canonical merged WWTP layer)]

	D[(HydroBASINS zip archives)] --> E[combine_watersheds]
	E --> F[(Combined watershed layers)]

	C --> G[annotation_grids]
	G --> H[(annotation grids and OSM context)]
	H --> I[download_bing_annotate]
	I --> J[(annotated imagery)]
	J --> K[merge_annotations]
	K --> L[(annotation-enriched WWTP layer)]

	M[(WorldPOP rasters)] --> N[download_pop]
	L --> O[create_voronoi]
	F --> O
	O --> P[(Voronoi service areas)]
	N --> Q[(population raster store)]
	P --> R[add_pop]
	Q --> R
	R --> S[(population-enriched Voronoi layers)]

	S --> T[create_rasters]
	T --> T1[(raster country statistics)]
	T --> U1[find_unserved_pop]
	F --> U3
	V[(HydroRIVERS)] --> U3
	U1 --> U2[find_diff_pop]
	U1 --> U3[assign_rivers_to_basin]
	U3 --> U4[find_intersection_river]
	U4 --> U5[impact_polygons_pop]
	U5 --> W[find_pop_in_danger_pop]
	W --> X[(population-at-risk outputs)]

	S --> Y[verification_script, hw_comparison, eu_comparison]
	Y --> Z[(validation outputs)]

	AZ[(Zenodo industrial land-cover source)] --> AA[download_and_vectorize]
	AA --> AA1[(vectorized industrial areas)]
	L --> AA2[find_unconnected_industrial_areas]
	F --> AA2
	AA1 --> AA2
	AA2 --> AB[(industrial coverage outputs)]

	S --> AC[composite_area_population_plots]
	X --> AD[pop_at_risk_figures]
	T1 --> AG[piechart_figure, piechart_interactive]
	AG --> AH[(static and interactive piecharts)]
	S --> AI[sizes_interactive_map]
	AI --> AJ[(interactive size map)]
	AB --> AE[interactive_unconnected_industrial_map]
	AE --> AF[(unconnected-industrial HTML map)]
```

Notes on this diagram:
- `find_diff_pop` is a dead end: nothing downstream depends on its output.
- Only `find_unconnected_industrial_areas` reads the annotation-enriched WWTP layer; `download_and_vectorize` has no annotation dependency, so the two industrial-analysis stages are shown as gated separately.
- `composite_area_population_plots` reads the population-enriched Voronoi layers directly and does not consume validation or population-at-risk outputs.
- `piechart_figure, piechart_interactive` (static and interactive) reads the raster country statistics produced by `create_rasters`; `sizes_interactive_map` reads the population-enriched Voronoi layers directly, not the raster statistics.

## Environment Creation

The code is the `wwtp` package (folder `wwtp/`; it was called `src` until November 2026) and its metadata is `pyproject.toml` at the repository root. A minimal local environment can be created from the repository root. An environment installed before the rename needs `python -m pip install -e .` once more.

### PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### Bash
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To install the test dependencies as well:

```bash
python -m pip install -e ".[test]"
```

Practical prerequisites:
- Python 3.11 or newer.
- Snakemake for the workflow (`pip install -e ".[workflow]"`, which also brings the SLURM executor plugin).
- Access to the geospatial inputs under `data/`.
- SLURM access for `workflow/profiles/slurm`.

## Checks: pre-commit and CI

`pre-commit install` once per clone (pre-commit itself is best installed apart from the pipeline's environment, e.g. `pipx install pre-commit`) runs on every commit: ruff (lint and format), gitleaks (secrets), large files (> 5 MB), YAML/TOML syntax, line endings, nbstripout for notebooks and shellcheck for the remaining shell scripts. `pre-commit run --all-files` checks everything. The ruff reformatting commit is listed in `.git-blame-ignore-revs` (`git config blame.ignoreRevsFile .git-blame-ignore-revs`).

GitHub Actions (`.github/workflows/`): on every push and pull request the hooks and the whole test suite with coverage (the floor in `.coveragerc` only rises); weekly and on demand the slow end-to-end checks on the mock world against the stored golden outputs, through the stage command lines and through Snakemake.

## Starter Data Files You Need Before Running

The pipeline relies on input data artifacts. Without these files or their configured equivalents, the workflow cannot complete.

| Data file or directory | Consumed by | Status in this workspace |
| --- | --- | --- |
| `data/Enhanced_HW_WWTP__jun20_2025.geojson` | geometry correction and merge bootstrap | present |
| `data/wastewater_plant.geojson` | OSM-based location correction | present |
| `data/DL_results/csvv2-2.zip` | legacy segmentation merge (`--variant old`) | present |
| `data/DL_results/aftersort.geojson` | legacy segmentation index mapping | present |
| `data/extra_points/Canada_14_03_2025.csv` | final merge enrichment | present |
| `data/extra_points/Germany_Hydra_waste_geospatial_corrected.geojson` | final merge enrichment | present |
| `data/final_data_source/final_W_europe_WWT_dec3.geojson` | final merge enrichment | present |
| `data/final_data_source/Thailand_500m_merged.geojson` | final merge enrichment | present |
| `data/final_data_source/final_USA_WWT_dec3.geojson` (or another file set in `final_data_merge.paths.us_new_filepath`) | final merge enrichment | missing at default path |
| `data/hydroshed_river_levels/lvl{level}/` | watershed combination (`combine_watersheds`) | needed for the default `level=6` and for every level the sweep runs (5, 6, 7) |
| `data/cleaned_hydrowaste.csv` | weighted Voronoi input features | present |
| `data/bboxes.csv` | weighted Voronoi support table | present |
| `data/cities.csv` | optional city-based Voronoi approach | present |
| `data/hydrorivers.gpkg` | river assignment and downstream risk chain | missing |
| `data/extra_points/UWWTD_TreatmentPlants.gpkg` | EU validation comparison | present |
| `data/boundaries/ne_110m_admin_0_countries.shp` (+ sidecars) | figures and country overlays | present |

In addition, two canonical defaults point to external cluster paths and must usually be overridden locally:
- `merge_seg_results.paths.seg_results_filepath`
- `download_bing_annotate.paths.annotations_images_dir` (and related annotation-output paths)

If you only need to read or modify execution settings, start with `wwtp/config.yaml` and `wwtp/README.md`. The shell launchers orchestrate execution, but the pipeline dependencies are the data files above.

## Configuration Parameters You Usually Need To Change

The defaults are a mix of portable relative paths and environment-specific cluster paths. Before a local run, check at least the following keys.

| Config key | Why you may need to change it |
| --- | --- |
| `merge_seg_results.paths.seg_results_filepath` | Defaults to a cluster-only segmentation CSV path |
| `download_bing_annotate.paths.annotations_images_dir` | Defaults to a cluster-only imagery directory |
| `download_bing_annotate.paths.annotated_images_output_dir` | Defaults to a cluster-only annotated-image directory |
| `merge_annotations.paths.annotations_results_filepath` | Defaults to a cluster-only annotation results CSV |
| `annotations_inspection.paths.annotations_verf_image_outpath_dir` | Defaults to a cluster-only QA output directory |
| `correct_locations_w_OSM.paths.paul_corrected_filepath` | Must point to the corrected HydroWASTE source you actually have |
| `combine_watersheds.paths.watersheds_zip_dir` | Must match where the HydroBASINS zip files were placed |
| `create_voronoi.level` | Controls HydroBASINS granularity |
| `create_voronoi.buffer` | Controls static clip radius when dynamic buffering is off |
| `create_voronoi.weight_method` | Controls weight transformation |
| `create_voronoi.weight_func` | Controls multiplicative vs additive weighted distance |
| `create_voronoi.dynamic_buffering` | Switches between static and dynamic clipping |
| `create_voronoi.size_measure` | Plant size in people (`reported` or `footprint`) for the service radius and `size` weights |
| `create_rasters.zoom_level` | Controls raster/tile resolution used in the risk workflow |
| `find_unserved_pop.figures.pop_threshold` | Controls thresholding of non-served population polygons |
| `find_intersection_river.x_distance` | Controls maximum river-match distance for non-served polygons |
| `impact_polygons_pop.impact_polygons_pop_params.*` | Controls downstream organic-load propagation |
| `find_unconnected_industrial_areas.industrial_category_numbers` | Controls which annotation categories are treated as industrial WWTPs |

`wwtp/starter.py` resolves values in canonical YAML order. Earlier sections define shared values and later sections inherit them with `null`. Named CLI overrides (`--level`, `--buffer`, ...) take precedence over the resolved config.

### Config layers

`wwtp/config.yaml` is the portable base. On top of it, in this order:

1. overlay files listed in `WWTP_CONFIG_OVERLAYS` (separated by `:`), e.g. a `config.cluster.yaml` with cluster paths and worker counts (see `wwtp/config.cluster.example.yaml`);
2. `wwtp/config.local.yaml` if it exists (git-ignored, per machine);
3. `--set SECTION.KEY=VALUE` on any stage's command line (repeatable).

An overlay can only change keys the base already has, so a typo fails at load. `python -m wwtp.starter show <section> [--set ...]` prints what a stage will use, including `config_layers`; every run manifest records the same list.

### Typed settings

Each section has a pydantic model in `wwtp/settings.py`. After the layers and CLI overrides are applied, `load_config` checks the section against it: types, allowed values (`weight_method`, `weight_func`, `size_measure`, ...), positive counts and sizes, and cross-field rules (`weight_method: size` needs `weight_func: mult`; `dynamic_buffering` needs a `size_measure` with a `radius_calibration` entry; `start_year` ≤ `end_year`). A failure names every bad key, e.g. `create_voronoi.n_points: Input should be greater than 0`. A new config key goes into config.yaml and into its model; `wwtp/tests/unit/test_settings.py` fails while the two disagree. Stages still receive a dict; `settings.parse(section, cfg)` gives the model for attribute access.

Every stage takes the same named overrides of the run parameters: `--level`, `--version`, `--buffer`, `--weight-method`, `--weight-func`, `--dynamic-buffering`, `--size-measure` (the workflow passes all seven).

## What Is Missing Or External To The Repository

The repository is not fully self-contained. Several required inputs are expected to exist outside the code.

- Satellite imagery for annotation is not downloaded by the pipeline; the code expects an existing image directory.
- Annotation inference CSV outputs are treated as external inputs to the merge-back stage.
- Some default paths in `wwtp/config.yaml` point to cluster-local storage and must be replaced for local execution.
- HydroBASINS zip archives must be placed manually under `data/hydroshed_river_levels/lvl{level}` before watershed combination.
- The corrected HydroWASTE source referenced by `correct_locations_w_OSM.paths.paul_corrected_filepath` is used by the pipeline, but the publication metadata for that correction source is not embedded in the repository.
- River matching distance is configurable via `find_intersection_river.x_distance` (default 5000 m).

## Section Documentation

The section READMEs under `wwtp/` are the detailed reference documents for each stage.

| Section | Detailed documentation |
| --- | --- |
| `wwtp/` technical index | `wwtp/README.md` |
| Data merge | `wwtp/data_merge/README.md` |
| Annotation | `wwtp/annotation_scripts/README.md` |
| Population and river-risk calculations | `wwtp/pop_at_risk_river_calculations/README.md` |
| Validation | `wwtp/pop_validation_scripts/README.md` |
| Figures and exports | `wwtp/figures_scripts/README.md` |
| Industrial analysis | `wwtp/industrial_analysis/README.md` |
| Sensitivity analysis | `wwtp/sensitivity_analysis_scripts/README.md` |
