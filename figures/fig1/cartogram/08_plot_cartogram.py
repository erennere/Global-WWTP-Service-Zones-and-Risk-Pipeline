"""Second global map: contiguous cartogram of HydroBASINS basins, area ~ urban population not served,
colour = non-served ratio R = 1 - P_served / (P_served + P_non).

Usage: python 08_plot_cartogram.py <level 3|4> <out_stem> [--width 88] [--variant base|labels|locator|labels+locator|truearea|truearea+labels]
                                    [--scale linear|exp|quantile|continuous]
  truearea: the same colours on the undistorted basins (Equal Earth), as a standalone map
  above50:  only basins where more than half of the urban population is not served are coloured (50-100 %);
            the rest are grey. --scale sets the colour classes:
              linear      50 / 60 / 70 / 80 / 90 / 100 %
              exp         each class halves the remaining gap to 100 %: 50 / 75 / 87.5 / 93.75 / 96.9 / 100 %
              quantile    the same number of basins in each class
              continuous  one continuous ramp from 50 to 100 %
"""
import argparse
import numpy as np, geopandas as gpd, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import box

ap = argparse.ArgumentParser()
ap.add_argument("level", type=int); ap.add_argument("out")
ap.add_argument("--width", type=float, default=88.0, help="figure width in mm (88 = one Nature column)")
ap.add_argument("--variant", default="base")
ap.add_argument("--scale", default="linear", choices=["linear", "exp", "quantile", "continuous"])
a = ap.parse_args()
V = set(a.variant.split("+"))

MM = 1 / 25.4
mpl.rcParams.update({"font.family": "Liberation Sans", "font.size": 6, "axes.linewidth": 0.4,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": 0.35})
INK, INK2 = "#1a1a1a", "#5a5a5a"
NODATA, HATCH_EDGE = "#dedede", "#9a9a9a"
FS, FS2 = 5.4, 5.0
R_BRK = [0.10, 0.25, 0.50, 0.75, 0.90]
R_COL = ["#01665e", "#4fa69d", "#93cdc3", "#dcbc78", "#c98d3b", "#8c510a"]   # low -> high non-served; adj dE >= 9.7
R_LAB = ["0", "10", "25", "50", "75", "90", "100"]
A50 = "above50" in V
A50_COL = ["#f6e8c3", "#dfc27d", "#bf812d", "#8c510a", "#543005"]   # BrBG brown side; adjacent dE >= 11.5 (normal/protan/deutan)
LOW = "#cfcfcf"                                                     # basins with 50 % or less not served
if A50:
    NODATA = "#ececec"
# well-known basins (level-3 Pfafstetter code -> label); at level 4 the largest child basin carries the label
LABELS = {452: "Ganges–\nBrahmaputra", 453: "Peninsular\nIndia", 456: "Indus", 434: "Yangtze", 515: "Java",
          172: "Nile", 431: "North\nChina"}

TRUE = "truearea" in V
if TRUE:
    from shapely.geometry import Polygon
    c = gpd.read_file(f"svc/service_hybas{a.level}.gpkg")
    c["geometry"] = c.geometry.make_valid()
    c = c.clip(box(-180, -58, 180, 90)).to_crs(8857)
    c["R"] = np.where((c.P_served + c.P_non) > 0, c.P_non / (c.P_served + c.P_non), np.nan)
    ok = (c.status == "ok") & c.R.notna()          # fully served basins (R = 0) are coloured too
    lon = np.r_[np.linspace(-180, 180, 361), np.full(181, 180), np.linspace(180, -180, 361), np.full(181, -180)]
    lat = np.r_[np.full(361, 90), np.linspace(90, -58, 181), np.full(361, -58), np.linspace(-58, 90, 181)]
    sphere = gpd.GeoSeries([Polygon(zip(lon, lat))], crs=4326).to_crs(8857)
    land = gpd.read_file("data/ne_50m_land.geojson").clip(box(-180, -58, 180, 90)).to_crs(8857)
else:
    c = gpd.read_file(f"svc/cartogram_hybas{a.level}.gpkg")
    c["geometry"] = c.geometry.make_valid()
    ok = (c.status == "ok") & (c.P_non > 0)
c["col"] = None
if A50:
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    CMAP = LinearSegmentedColormap.from_list("a50", A50_COL)
    low = (c.status == "ok") & c.R.notna() & (c.R <= 0.5)
    ok = (c.status == "ok") & (c.R > 0.5) & (c.P_non > 0)
    if a.scale == "linear":
        A_BRK = [0.6, 0.7, 0.8, 0.9]
    elif a.scale == "exp":
        A_BRK = [1 - 0.5 ** k for k in (2, 3, 4, 5)]
    elif a.scale == "quantile":
        A_BRK = list(np.unique(np.round(np.quantile(c.loc[ok, "R"], [.2, .4, .6, .8]), 4)))
    if a.scale == "continuous":
        c.loc[ok, "col"] = [to_hex(CMAP((r - 0.5) / 0.5)) for r in c.loc[ok, "R"]]
    else:
        cols = A50_COL if len(A_BRK) == 4 else [to_hex(CMAP(t)) for t in np.linspace(0, 1, len(A_BRK) + 1)]
        c.loc[ok, "col"] = [cols[i] for i in np.digitize(c.loc[ok, "R"], A_BRK, right=True)]
    c.loc[low, "col"] = LOW
    ok = ok | low
else:
    c.loc[ok, "col"] = [R_COL[i] for i in np.digitize(c.loc[ok, "R"], R_BRK)]
unc = c.status == "uncovered"
grey = ~ok & ~unc

x0, y0, x1, y1 = (sphere.total_bounds if TRUE else c[ok & (c.P_non > 0)].total_bounds)
pad = 0.02 * (x1 - x0)
x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad

# ------------------------------------------------------------------ layout (mm)
W = a.width
MAPW = W - 2.0
MAPH = MAPW * (y1 - y0) / (x1 - x0)
TITLE, LEG = 4.5, 13.5
H = TITLE + MAPH + LEG + 0.5
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([1.0 / W, LEG / H, MAPW / W, MAPH / H])

outline = land.boundary if TRUE else c.dissolve().boundary
if TRUE:
    sphere.plot(ax=ax, facecolor="white", edgecolor="none", zorder=0)
    land.plot(ax=ax, facecolor=NODATA, edgecolor="none", zorder=0.5)
    sphere.boundary.plot(ax=ax, color="#9a9a9a", linewidth=0.35, zorder=5)
c[grey].plot(ax=ax, facecolor=NODATA, edgecolor="none", zorder=1)
if unc.any():
    c[unc].plot(ax=ax, facecolor=NODATA, edgecolor=HATCH_EDGE, hatch="//////", linewidth=0, zorder=1)
c[ok].plot(ax=ax, color=c.loc[ok, "col"], edgecolor="none", zorder=2)
c.boundary.plot(ax=ax, color="white", linewidth=0.12 if a.level == 3 else 0.07, zorder=3)
outline.plot(ax=ax, color="#8a8a8a", linewidth=0.25, zorder=4)

if "labels" in V:
    c["p3"] = c.PFAF_ID.astype(str).str[:3].astype(int)
    for p3, lab in LABELS.items():
        if TRUE and p3 == 515:          # Java is too small to carry a label at true size
            continue
        sub = c[(c.p3 == p3) & ok]
        if sub.empty:
            continue
        pt = sub.geometry.union_all().representative_point()          # whole level-3 basin
        hit = sub[sub.geometry.contains(pt)]
        col = hit.col.iloc[0] if len(hit) else sub.loc[sub.geometry.area.idxmax(), "col"]
        rgb = mpl.colors.to_rgb(col)
        dark = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) < 0.45
        ax.text(pt.x, pt.y, lab, fontsize=FS2, color="white" if dark else INK, ha="center", va="center",
                linespacing=0.95, zorder=6)

if "locator" in V:
    # true-geography thumbnail with the same colours, lower left
    u = gpd.read_file(f"svc/service_hybas{a.level}.gpkg")
    u["geometry"] = u.geometry.make_valid()
    u = u.clip(box(-180, -58, 180, 90)).to_crs(8857)
    oku = (u.status == "ok") & (u.P_non > 0)
    u["col"] = None
    u.loc[oku, "col"] = [R_COL[i] for i in np.digitize(u.loc[oku, "R"] if "R" in u else 1 - u.loc[oku, "S"], R_BRK)]
    ins = ax.inset_axes([0.0, 0.0, 0.27, 0.27 * (MAPW / MAPH) * 0.47])
    u[~oku].plot(ax=ins, facecolor=NODATA, edgecolor="none")
    u[oku].plot(ax=ins, color=u.loc[oku, "col"], edgecolor="none")
    ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([]); ins.set_xlabel(""); ins.set_ylabel("")
    for s in ins.spines.values():
        s.set_linewidth(0.3); s.set_color("#9a9a9a")
    ins.text(0.03, 0.04, "true area", transform=ins.transAxes, fontsize=4.6, color=INK2)

ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")

fig.text(0.6 / W, (H - 0.5) / H, "b", fontsize=8, fontweight="bold", va="top", ha="left", color=INK)
fig.text(3.8 / W, (H - 0.85) / H, "Urban population not served by wastewater treatment" +
         (" (basins above 50 %)" if A50 and W > 120 else ""), fontsize=6.5, va="top", ha="left", color=INK)
fig.text((W - 0.5) / W, (H - 0.6) / H, f"HydroBASINS level {a.level}", fontsize=FS2, va="top", ha="right", color=INK2)

# ------------------------------------------------------------------ legend (mm)
L = fig.add_axes([0, 0, 1, LEG / H]); L.axis("off"); L.set_xlim(0, W); L.set_ylim(0, LEG)
lx, ly, sw = 2.0, LEG - 6.0, 5.4
fmt = lambda b: f"{100 * b:.1f}".rstrip("0").rstrip(".")
if A50:
    sw = 5.0
    L.add_patch(Rectangle((lx, ly), sw - 0.25, 2.4, facecolor=LOW, lw=0))
    L.text(lx + 0.5 * sw, ly - 0.5, "≤ 50", ha="center", va="top", fontsize=FS2, color=INK2)
    bx = lx + sw + 0.6
    if a.scale == "continuous":
        grad = np.linspace(0, 1, 256)[None, :]
        L.imshow(grad, cmap=CMAP, extent=(bx, bx + 5 * sw - 0.25, ly, ly + 2.4), aspect="auto", zorder=1)
        for k, t in enumerate(("", "60", "70", "80", "90", "100")):   # the grey swatch already reads "<= 50"
            L.text(bx + k * (5 * sw - 0.25) / 5, ly - 0.5, t, ha="center", va="top", fontsize=FS2, color=INK2)
    else:
        nb = len(A_BRK) + 1; wcl = 5 * sw / nb
        for k, col in enumerate(cols):
            L.add_patch(Rectangle((bx + k * wcl, ly), wcl - 0.25, 2.4, facecolor=col, lw=0))
        labs = [""] + [fmt(b) for b in A_BRK] + ["100"]                 # the grey swatch already reads "<= 50"
        for k, t in enumerate(labs):
            L.text(bx + k * wcl - 0.12, ly - 0.5, t, ha="center", va="top", fontsize=FS2 - (0.5 if a.scale != "linear" else 0),
                   color=INK2)
    L.set_xlim(0, W); L.set_ylim(0, LEG)
    L.text(lx, ly + 3.0, "Urban population not served (%)", fontsize=FS, color=INK, va="bottom")
else:
    for k, col in enumerate(R_COL):
        L.add_patch(Rectangle((lx + k * sw, ly), sw - 0.25, 2.4, facecolor=col, lw=0))
    for k, t in enumerate(R_LAB):
        L.text(lx + k * sw - 0.12, ly - 0.5, t, ha="center", va="top", fontsize=FS2, color=INK2)
    L.text(lx, ly + 3.0, "Urban population not served (%)", fontsize=FS, color=INK, va="bottom")
# area key: square holding 100 M non-served people at the cartogram's scale
mm_per_m = MAPW / (x1 - x0)
if TRUE:
    L.text(lx + 38.0, LEG - 2.0, "True basin area\n(Equal Earth projection)", fontsize=FS2, color=INK2, va="top",
           linespacing=1.0)
    m2_per_person = None
else:
    okp = ok & (c.P_non > 0)
    m2_per_person = c.geometry.area[okp].sum() / c.P_non[okp].sum()
    xs = lx + 38.0
    cand = [n for n in (10e6, 25e6, 50e6, 100e6) if np.sqrt(n * m2_per_person) * mm_per_m <= 5.6]
    for n in (cand[0], cand[-1]) if len(cand) > 1 else cand:
        side = np.sqrt(n * m2_per_person) * mm_per_m
        L.add_patch(Rectangle((xs, 0.9), side, side, facecolor="#9a9a9a", lw=0))
        L.text(xs + side + 0.6, 0.9 + side / 2, f"{n / 1e6:,.0f} M", fontsize=FS2, color=INK2, ha="left", va="center")
        xs += side + 6.2
    L.text(lx + 38.0, LEG - 2.0, "Basin area proportional to\nurban population not served", fontsize=FS2, color=INK2,
           va="top", linespacing=1.0)
L.add_patch(Rectangle((lx + 66.0, LEG - 5.2), 2.6, 2.6, facecolor=NODATA, lw=0))
L.text(lx + 69.4, LEG - 3.9, "no urban population" if TRUE else "none / no data", fontsize=FS2, color=INK, va="center")
L.add_patch(Rectangle((lx + 66.0, LEG - 9.0), 2.6, 2.6, facecolor=NODATA, edgecolor=HATCH_EDGE, hatch="//////", lw=0))
L.text(lx + 69.4, LEG - 7.7, "not covered", fontsize=FS2, color=INK, va="center")

fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=600)
print("wrote", a.out, f"({W:.0f} x {H:.1f} mm)")
