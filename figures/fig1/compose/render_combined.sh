#!/bin/bash
# Figure 1 combined, one diagram per level (country, hybas3, hybas4): map 1 row (a shift, b added/missing,
# europe_with_spikes design) over map 2 row (c cartogram, d tanks map), PDF + 600 dpi PNG.
# Versions: full (map 2 colours 0-100 %) and above50_quantile (only units above 50 % not served).
# Usage: ATTRS=<HeiGIT gpkg> HB=<hydrobasins dir> [TAG=r<figure1.match_radius_m>] [LEVELS="country hybas3 hybas4"]
#        [D=truearea [NL=<night-lights GeoTIFF>]] render_combined.sh
#   ATTRS  HeiGIT file with round_area, rect_area, category_number, category_name (panel d)
#   HB     folder with hybas_lev03_v1c.gpkg, hybas_lev04_v1c.gpkg, hybas_lev08/09_v1c_attributes.csv
#   D=truearea  panel d as the true-area map (as before the tanks map), with night-light versions if NL is set
S="$(cd "$(dirname "$0")" && pwd)"                    # this folder: the scripts
C="python3 $S/../fig1_config.py"                      # config.yaml, section figure1
W="$($C combined_dir)"; mkdir -p "$W"; cd "$W"   # work folder under data_dir
: "${ATTRS:?set ATTRS to the HeiGIT GeoPackage}"
: "${HB:?set HB to the HydroBASINS folder}"
TAG=${TAG:-$($C tag)}; D=${D:-tanks}; LEVELS=${LEVELS:-country hybas3 hybas4}
ROOT=$(pwd); M1=$($C match_dir); M2=$($C cartogram_dir)
R=$ROOT/rows; mkdir -p $R
q(){ "$@" 2>&1 | grep -v -i warn | tail -1; }

# map 2 at country level: aggregation and cartogram, once
cd $M2
[ -f svc/service_country.gpkg ] || q python3 $S/../cartogram/05b_service_country.py data $M1/agg svc $TAG
[ -f svc/cartogram_country.gpkg ] || q python3 $S/../cartogram/07_cartogram.py country --rounds 8 --grid 1536
cd $ROOT

declare -A LAB=( [country]=Countries [hybas3]="HydroBASINS level 3" [hybas4]="HydroBASINS level 4" )
declare -A LEV=( [country]=country [hybas3]=3 [hybas4]=4 )
for l in $LEVELS; do
  RIV=""; [ $l != country ] && RIV="--rivers"
  # map 1 row
  (cd $M1 && q python3 $S/../match/03_plot.py agg/units_${l}_${TAG}.gpkg "${LAB[$l]}" $R/map1_$l --variant europe+withspikes \
              --plants agg/plants_with_units_${TAG}.csv $RIV) &
  # map 2 rows
  if [ $D = tanks ]; then
    DD="--d tanks --map1 $M1 --tag $TAG --attrs $ATTRS --hybas-dir $HB"; [ $l != country ] && DD="$DD --tank-rivers"
    VERS="full above50_quantile"
  else
    DD=""; VERS="full above50_quantile"; [ -n "$NL" ] && VERS="$VERS full_night above50_quantile_night"
  fi
  for v in $VERS; do
    X=""; [[ $v == above50* ]] && X="--above50 --scale quantile"; [[ $v == *_night ]] && X="$X --nightlights $NL"
    (cd $M2 && q python3 $S/../cartogram/09_plot_pair.py ${LEV[$l]} $R/map2_${l}_$v --letters cd $X $DD) &
  done
  wait
  for v in $VERS; do
    q python3 $S/10_compose.py $ROOT/fig1_combined_${l}_$v $R/map1_$l $R/map2_${l}_$v
  done
done
echo DONE
