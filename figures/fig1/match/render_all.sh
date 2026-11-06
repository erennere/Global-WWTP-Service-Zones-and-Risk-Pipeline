#!/bin/bash
# Render every aggregation level x design variant (PDF + 600 dpi PNG).
# Usage: [FIGS=dir] render_all.sh [TAG] [VARIANT ...]   (default TAG from figure1.match_radius_m, FIGS figs, default: all variants)
S="$(cd "$(dirname "$0")" && pwd)"                    # this folder: the scripts
C="python3 $S/../fig1_config.py"                      # config.yaml, section figure1
W="$($C match_dir)"; mkdir -p "$W"; cd "$W" || exit 1   # work folder under data_dir
TAG=${1:-$($C tag)}; shift
export FIGS=${FIGS:-figs}
mkdir -p $FIGS/country $FIGS/hybas3 $FIGS/hybas4
P="--plants agg/plants_with_units_${TAG}.csv"
# variant spec -> file suffix
declare -A NAME=( [base]=base [europe]=europe [europe+latprofile]=europe_latprofile [spikes+europe]=spikes_europe
                  [withspikes]=main_with_spikes [europe+withspikes]=europe_with_spikes
                  [withcircles]=main_with_circles [europe+withcircles]=europe_with_circles )
VARIANTS=${*:-base europe europe+latprofile spikes+europe withspikes europe+withspikes withcircles europe+withcircles}
run(){ python3 $S/03_plot.py "$1" "$2" "$FIGS/$3/fig1_$3_${NAME[$4]}" --variant $4 $P $5 2>&1 | grep -v -i warn | tail -1; }
for v in $VARIANTS; do
  run agg/units_country_${TAG}.gpkg Countries country $v &
  run agg/units_hybas3_${TAG}.gpkg "HydroBASINS level 3" hybas3 $v --rivers &
  run agg/units_hybas4_${TAG}.gpkg "HydroBASINS level 4" hybas4 $v --rivers &
  wait
done
for l in country hybas3 hybas4; do python3 $S/04_contact_sheet.py $FIGS/contact_sheet_$l.png $l; done
echo DONE
