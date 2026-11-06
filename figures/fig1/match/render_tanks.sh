#!/bin/bash
# Panel 1d (round-tank share + plant surroundings) for every level and both glyphs (PDF + 600 dpi PNG).
# Usage: [ATTRS=<HeiGIT gpkg>] [HB=<hydrobasins dir>] [HEIGIT_PLANTS=<plants_heigit csv>] [FIGS=dir] render_tanks.sh [TAG]
#   ATTRS, HB default to figure1.paths.heigit_attrs / hydrobasins_dir in wwtp/config.yaml
#   ATTRS          HeiGIT file with round_area, rect_area, category_number, category_name (read by fid)
#   HB             folder 02b_aggregate_basins.py reads; without it only the country level is drawn
#   HEIGIT_PLANTS  01_reconcile's plants_heigit_<TAG>.csv, if it is not in agg/
S="$(cd "$(dirname "$0")" && pwd)"                    # this folder: the scripts
C="python3 $S/../fig1_config.py"                      # config.yaml, section figure1
W="$($C match_dir)"; mkdir -p "$W"; cd "$W" || exit 1   # work folder under data_dir
TAG=${1:-$($C tag)}
ATTRS=${ATTRS:-$($C heigit_attrs)}           # figure1.paths.heigit_attrs
HB=${HB-$($C hydrobasins_dir)}              # figure1.paths.hydrobasins_dir; HB= draws the country level only
export FIGS=${FIGS:-figs}
mkdir -p $FIGS/tanks
P="--plants agg/plants_with_units_${TAG}.csv --attrs $ATTRS"
[ -n "$HEIGIT_PLANTS" ] && P="$P --heigit-plants $HEIGIT_PLANTS"
run(){ python3 $S/11_plot_tanks.py "$@" $P 2>&1 | grep -v -i warn | tail -1; }
for g in ringpie core; do
  run agg/units_country_${TAG}.gpkg Countries $FIGS/tanks/fig1d_tanks_country_$g --glyph $g &
  if [ -n "$HB" ]; then
    run agg/units_hybas3_${TAG}.gpkg "HydroBASINS level 3" $FIGS/tanks/fig1d_tanks_hybas3_$g --glyph $g --tank-rivers --hybas-dir "$HB" &
    run agg/units_hybas4_${TAG}.gpkg "HydroBASINS level 4" $FIGS/tanks/fig1d_tanks_hybas4_$g --glyph $g --tank-rivers --hybas-dir "$HB" &
  fi
  wait
done
echo DONE
