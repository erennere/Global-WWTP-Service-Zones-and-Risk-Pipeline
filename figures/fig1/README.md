# Figure 1

HeiGIT plants against reference plants (HydroWASTE, UWWTD, national registers) per country and
HydroBASINS level 3/4 unit (map 1), and the share of people not served as a cartogram plus the tank
map (map 2), composed into one figure per level. Moved here from `wwtp/maps/fig1_run_2026-11-04/`
(`match_10km/scripts` → `match/`, `fig1b_cartogram/scripts` → `cartogram/`, `fig1_combined/scripts` →
`compose/`). Method notes: the project docs `fig1_global_maps_method.md`, `fig1b_cartogram_method.md`,
`fig1_combined_method.md`.

## Settings (`wwtp/config.yaml`, section `figure1`)

| key | default | used by |
|---|---|---|
| `min_plants` | 5 | units with fewer matched plants are hatched, not coloured (`03_plot.py --min-corr/--min-ref`, `tanks_panel.py --min-fill/--min-glyph`) |
| `match_radius_m` | 10000 | `01_reconcile.py` pairs plants within this distance; it also names the outputs (`r10000`) the later steps read |
| `paths.match_dir` | `{data_dir}/figures/fig1/match` | work folder of map 1 (`data/`, `agg/`, `figs/`) |
| `paths.cartogram_dir` | `{data_dir}/figures/fig1/cartogram` | work folder of map 2 (`data/`, `svc/`, `figs2/`) |
| `paths.combined_dir` | `{data_dir}/figures/fig1/combined` | composed figures (`rows/`) |
| `paths.heigit_attrs` | `{match_dir}/data/HeiGIT_WWTP_LVL9_BFK0_8_LOG_MULT.gpkg` | HeiGIT plants with their attributes, read by fid (tanks panel; `ATTRS` of the render scripts) |
| `paths.hydrobasins_dir` | `{data_dir}/figures/fig1/hydrobasins` | `hybas_lev03/04_v1c.gpkg`, `hybas_lev08/09_v1c_attributes.csv` (`HB` of the render scripts) |

Command-line arguments still override every default. `python figures/fig1/fig1_config.py <key>` prints
a setting (the `render_*.sh` wrappers use it).

## Data

Inputs and outputs stay out of git, under `data_dir`. To continue from the 4 Nov run, move
`wwtp/maps/fig1_run_2026-11-04/match_10km/{data,agg,figs}` to `data/figures/fig1/match/`,
`fig1b_cartogram/{data,svc,figs2}` to `data/figures/fig1/cartogram/`, and `fig1_combined/rows` to
`data/figures/fig1/combined/`.

## Run order

`snakemake fig1` (from the repository root; rules in `workflow/rules/fig1.smk`) runs everything after the
prepared inputs: `01_reconcile` → `02_aggregate` → `02b_aggregate_basins` → map 1 (`render_all.sh`) and the tanks
panel (`render_tanks.sh`); `05_service` + `05b_service_country` → `07_cartogram` (levels 3, 4, country) → map 2
(`render_cartograms.sh`, `render_pairs.sh`, `render_above50.sh`) → `render_combined.sh`. Only what is missing or
older than its inputs reruns.

Prepared inputs (by hand, once): `match/data/` and `cartogram/data/` hold `heigit_points.csv`, `uwwtd_points.csv`,
`HydroWASTE_v10.csv`, `heigit_pop.csv`, `non_served_points.csv` and the Natural Earth layers;
`match/00_fix_utm.py <data_dir>` (repairs UTM back-projections in `heigit_points.csv`, keeping the original as
`heigit_points_original.csv`) and `match/00b_national_reference.py` (Australian and Japanese reference plants) edit
or replace those files and are run before.

Each render script changes into its work folder from the config before running; `ATTRS`/`HB` default to the two
paths above.
