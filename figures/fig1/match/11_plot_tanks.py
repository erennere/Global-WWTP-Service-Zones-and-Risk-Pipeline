"""Figure 1d as a standalone panel: round-tank share (fill) and plant surroundings (glyph per country).
The method is in tanks_panel.py; 09_plot_pair.py --d tanks draws the same panel into the combined figure.

Usage:
  python 11_plot_tanks.py <units.gpkg> <unit label> <out stem> --attrs <HeiGIT.gpkg|csv> [options]
  e.g. python figures/fig1/match/11_plot_tanks.py agg/units_country_r10000.gpkg Countries figs/tanks/fig1d_tanks_country \\
         --attrs data/HeiGIT_WWTP_LVL9_BFK0_8_LOG_MULT.gpkg --glyph ringpie
Writes <out>.pdf, <out>.png (600 dpi), <out>_units.csv and <out>_countries.csv.
"""

import argparse, os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tanks_panel as tp

ap = argparse.ArgumentParser()
ap.add_argument("units")
ap.add_argument("label")
ap.add_argument("out")
ap.add_argument("--width", type=float, default=87.5, help="figure width in mm (87.5 = one panel of the 180 mm pair)")
ap.add_argument("--letter", default="d")
tp.add_args(ap)
a = ap.parse_args()
if a.ne_data is None:
    a.ne_data = "data"

MM = 1 / 25.4
mpl.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.size": 6,
        "axes.linewidth": 0.4,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "hatch.linewidth": 0.35,
    }
)
T = tp.Tanks(a, a.units, a.label)

W, M = a.width, 1.0
MAPW = W - 2 * M
MAPH = MAPW * T.aspect()
TITLE, LEG = 4.0, tp.LEGEND_H + 0.5
H = TITLE + MAPH + LEG + 0.5
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([M / W, LEG / H, MAPW / W, MAPH / H])
T.draw(ax)
fig.text(M / W, (H - 0.6) / H, a.letter, fontsize=8, fontweight="bold", va="top", ha="left", color=tp.INK)
fig.text(
    (M + 3.2) / W,
    (H - 0.95) / H,
    "Tank geometry and plant surroundings",
    fontsize=6.5,
    va="top",
    ha="left",
    color=tp.INK,
)
fig.text(
    (W - M) / W,
    (H - 1.2) / H,
    "by countries" if a.label == "Countries" else f"by {a.label}",
    fontsize=tp.FS,
    va="top",
    ha="right",
    color=tp.INK2,
)
L = fig.add_axes([0, 0, 1, LEG / H])
L.axis("off")
L.set_xlim(0, W)
L.set_ylim(0, LEG)
T.legend(L, M, LEG - 0.6)
T.write_tables(a.out)
fig.savefig(a.out + ".pdf")
fig.savefig(a.out + ".png", dpi=600)
print("wrote", a.out, f"({W:.1f} x {H:.1f} mm)")
