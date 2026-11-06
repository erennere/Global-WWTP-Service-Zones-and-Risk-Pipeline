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

Command-line arguments still override every default. `python figures/fig1/fig1_config.py <key>` prints
a setting (the `render_*.sh` wrappers use it).

## Data

Inputs and outputs stay out of git, under `data_dir`. To continue from the 4 Nov run, move
`wwtp/maps/fig1_run_2026-11-04/match_10km/{data,agg,figs}` to `data/figures/fig1/match/`,
`fig1b_cartogram/{data,svc,figs2}` to `data/figures/fig1/cartogram/`, and `fig1_combined/rows` to
`data/figures/fig1/combined/`.

## Run order

1. `match/00_fix_utm.py`, `match/00b_national_reference.py` (reference plants), `match/01_reconcile.py <data_dir> <out_dir>`
2. `match/02_aggregate.py`, `match/02b_aggregate_basins.py` (units), then `match/render_all.sh`, `match/render_tanks.sh`
3. `cartogram/05_service.py`, `cartogram/05b_service_country.py`, `cartogram/render_cartograms.sh`, `cartogram/render_pairs.sh`
4. `ATTRS=… HB=… compose/render_combined.sh`

Each wrapper changes into its work folder from the config before running.
