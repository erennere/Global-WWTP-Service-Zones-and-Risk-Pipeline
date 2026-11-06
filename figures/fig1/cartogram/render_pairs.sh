#!/bin/bash
# Map 2 as two-panel figures (a cartogram, b true area), HydroBASINS level 3 and 4:
#   whole range 0-100 %: default classes + linear / exp / quantile / continuous
#   basins above 50 % not served: linear / exp / quantile / continuous
# plus a contact sheet.
S="$(cd "$(dirname "$0")" && pwd)"                    # this folder: the scripts
C="python3 $S/../fig1_config.py"                      # config.yaml, section figure1
W="$($C cartogram_dir)"; mkdir -p "$W"; cd "$W" || exit 1   # work folder under data_dir
O=${OUT:-figs2/pairs}; mkdir -p $O
r(){ python3 $S/09_plot_pair.py "$@" 2>&1 | grep -v -i warn | tail -1; }
for lev in 3 4; do
  r $lev $O/fig1b_pair_hybas${lev} &
  for sc in linear exp quantile continuous; do
    r $lev $O/fig1b_pair_hybas${lev}_$sc --scale $sc &
    r $lev $O/fig1b_pair_above50_hybas${lev}_$sc --above50 --scale $sc
    wait
  done
done
OUT=$O python3 - <<'PY'
import os
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
O = os.environ["OUT"]; font = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")), 34)
S = ("linear", "exp", "quantile", "continuous")
names = ([f"fig1b_pair_hybas{l}" for l in (3, 4)] + [f"fig1b_pair_hybas{l}_{s}" for l in (3, 4) for s in S] +
         [f"fig1b_pair_above50_hybas{l}_{s}" for l in (3, 4) for s in S])
tiles = []
for n in names:
    im = Image.open(f"{O}/{n}.png").convert("RGB"); im = im.resize((2400, int(im.height * 2400 / im.width)), Image.LANCZOS)
    t = Image.new("RGB", (2400, im.height + 60), "white"); ImageDraw.Draw(t).text((10, 10), n, fill="#333333", font=font)
    t.paste(im, (0, 60)); tiles.append(t)
sheet = Image.new("RGB", (2400, sum(t.height + 20 for t in tiles)), "#e9e9e9"); y = 0
for t in tiles:
    sheet.paste(t, (0, y)); y += t.height + 20
sheet.save(f"{O}/contact_sheet_pairs.png", optimize=True); print("contact sheet", sheet.size)
PY
echo DONE
