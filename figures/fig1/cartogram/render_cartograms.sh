#!/bin/bash
# Map 2: flow-based cartograms (level 3: 8 rounds, level 4: 10 rounds, 1536-cell grid) and all plot variants
S="$(cd "$(dirname "$0")" && pwd)"                    # this folder: the scripts
C="python3 $S/../fig1_config.py"                      # config.yaml, section figure1
W="$($C cartogram_dir)"; mkdir -p "$W"; cd "$W" || exit 1   # work folder under data_dir
python3 $S/07_cartogram.py 3 --rounds 8 --grid 1536 > svc/carto3_log.txt 2>&1
python3 $S/07_cartogram.py 4 --rounds 10 --grid 1536 > svc/carto4_log.txt 2>&1
mkdir -p figs2
for lev in 3 4; do
  for v in base labels locator labels+locator truearea; do
    python3 $S/08_plot_cartogram.py $lev "figs2/fig1b_cartogram_hybas${lev}_${v//+/_}" --variant $v 2>&1 | grep -v -i warn | tail -1
  done
  python3 $S/08_plot_cartogram.py $lev "figs2/fig1b_cartogram_hybas${lev}_labels_180mm" --variant labels --width 180 2>&1 | grep -v -i warn | tail -1
  python3 $S/08_plot_cartogram.py $lev "figs2/fig1b_cartogram_hybas${lev}_truearea_labels_180mm" --variant truearea+labels --width 180 2>&1 | grep -v -i warn | tail -1
done
echo DONE
