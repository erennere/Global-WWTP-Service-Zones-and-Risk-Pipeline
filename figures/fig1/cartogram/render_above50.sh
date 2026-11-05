#!/bin/bash
# Map 2, basins with more than 50 % of the urban population not served: 4 colour scalings x
# (cartogram with labels, true area) x HydroBASINS level 3 / 4, plus a contact sheet.
S="$(cd "$(dirname "$0")" && pwd)"                    # this folder: the scripts
C="python3 $S/../fig1_config.py"                      # config.yaml, section figure1
W="$($C cartogram_dir)"; mkdir -p "$W"; cd "$W"   # work folder under data_dir
O=${OUT:-figs2/above50}; mkdir -p $O
for lev in 3 4; do
  for sc in linear exp quantile continuous; do
    python3 $S/08_plot_cartogram.py $lev $O/fig1b_above50_hybas${lev}_cartogram_$sc --variant above50+labels --scale $sc 2>&1 | grep -v -i warn | tail -1 &
    python3 $S/08_plot_cartogram.py $lev $O/fig1b_above50_hybas${lev}_truearea_$sc --variant above50+truearea --scale $sc 2>&1 | grep -v -i warn | tail -1 &
    wait
  done
done
python3 - <<'PY'
from PIL import Image, ImageDraw, ImageFont
import os
import os; O = os.environ.get("OUT", "figs2/above50"); font = ImageFont.truetype("C:/Windows/Fonts/LiberationSans-Bold.ttf", 30)
rows = []
for lev in (3, 4):
    for sc in ("linear", "exp", "quantile", "continuous"):
        ims = [Image.open(f"{O}/fig1b_above50_hybas{lev}_{k}_{sc}.png").convert("RGB") for k in ("cartogram", "truearea")]
        Wt = 1400; ims = [im.resize((Wt, int(im.height * Wt / im.width)), Image.LANCZOS) for im in ims]
        h = max(im.height for im in ims) + 50
        t = Image.new("RGB", (2 * Wt + 20, h), "white"); ImageDraw.Draw(t).text((10, 8), f"level {lev}  ·  {sc}", fill="#333333", font=font)
        t.paste(ims[0], (0, 50)); t.paste(ims[1], (Wt + 20, 50)); rows.append(t)
sheet = Image.new("RGB", (rows[0].width, sum(r.height + 20 for r in rows)), "#e9e9e9")
y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.height + 20
sheet.save(f"{O}/contact_sheet_above50.png", optimize=True); print("contact sheet", sheet.size)
PY
echo DONE
