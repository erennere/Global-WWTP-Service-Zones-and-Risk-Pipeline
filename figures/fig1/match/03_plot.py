"""Figure 1 drafts: (a) median location shift of corrected plants, (b) plants added x missing (bivariate).
Internally the new/missing panel is keyed "a" and the shift panel "b"; POS/LETTER set where they are drawn.

Usage:
  python 03_plot.py <units.gpkg> <unit label> <out stem> [--variant V] [--rivers] [--plants CSV]
Variants (combine with '+', e.g. 'europe+latprofile'):
  base        bivariate choropleth (a) + classed median shift (b)
  latprofile  latitude profile strip beside each map (counts of new/missing; median shift)
  europe      Europe inset (2.5x) in the empty South Pacific of each map
  spikes      new/missing panel as a spike map (up = new, down = missing) instead of the bivariate fill
  withspikes  new/missing panel keeps the bivariate fill and overlays spikes (up = new, down = missing);
              shift panel overlays spikes for the number of corrected plants
  withcircles as withspikes, but the shift panel overlays circles: area ~ number of corrected plants
All spikes and circles are aggregated per country, whatever the fill unit, and drawn for every covered
country with at least one plant (no minimum count).
"""

import argparse
import numpy as np, pandas as pd, geopandas as gpd, pyproj
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import box, Polygon
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig1_config import MIN_PLANTS, TAG, enough_plants  # noqa: E402  (config.yaml, section figure1)

ap = argparse.ArgumentParser()
ap.add_argument("units")
ap.add_argument("label")
ap.add_argument("out")
ap.add_argument("--data", default="data")
ap.add_argument("--plants", default=f"agg/plants_with_units_{TAG}.csv")
ap.add_argument("--countries", default=None, help="country units gpkg (default: derived from --plants)")
ap.add_argument("--variant", default="base")
ap.add_argument("--rivers", action="store_true")
ap.add_argument("--min-corr", type=int, default=MIN_PLANTS)
ap.add_argument("--min-ref", type=int, default=MIN_PLANTS)
a = ap.parse_args()
V = set(a.variant.split("+"))
OVER = bool(V & {"withspikes", "withcircles"})  # spikes over the bivariate fill
SH_GLYPH = "spikes" if "withspikes" in V else ("circles" if "withcircles" in V else None)
POS = {"b": 0, "a": 1}  # shift panel left, new/missing panel right
LETTER = {"b": "a", "a": "b"}

# ------------------------------------------------------------------ style (Nature: sans 5-7 pt, 180 mm double column)
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
INK, INK2 = "#1a1a1a", "#5a5a5a"
NODATA, HATCH_EDGE = "#dedede", "#9a9a9a"
CRS = "EPSG:8857"  # Equal Earth (equal-area)
LAT_MIN = -58  # Antarctica cropped
FS, FS2 = 5.4, 5.0
FEW_HATCH = "......" if a.label == "Countries" else "...."

# bivariate 3x3, rows = missing (low->high), cols = new (low->high).
# Bilinear in OKLab between 4 corners; min all-pairs dE 12.7 normal / 12.9 protan / 12.7 deutan.
BIV = ["#f2f2f2", "#97b8e0", "#3a7dc9", "#f5c299", "#988b88", "#415471", "#f28e2b", "#915b2b", "#3b2b1f"]
BRK = [0.05, 0.25]
SHIFT_BRK = [25, 50, 100, 250, 1000]  # m
SHIFT_COL = ["#eee6ff", "#d1c1f2", "#b199de", "#9274c3", "#6e519d", "#48316c"]  # OKLCH hue 300, L .94->.37
NEW_C, MISS_C = BIV[2], BIV[6]

# ------------------------------------------------------------------ data
clipbox = box(-180, LAT_MIN, 180, 90)
u = gpd.read_file(a.units)
u["geometry"] = u.geometry.make_valid()
u = u.clip(clipbox).to_crs(CRS)
land = gpd.read_file(f"{a.data}/ne_50m_land.geojson").clip(clipbox).to_crs(CRS)
cty = gpd.read_file(f"{a.data}/ne_50m_admin_0_countries.geojson")[["geometry"]].clip(clipbox).to_crs(CRS)
rivers = None
if a.rivers:
    rivers = gpd.read_file(f"{a.data}/ne_50m_rivers_lake_centerlines.geojson")
    rivers = rivers[rivers.featurecla.str.contains("River")].clip(clipbox).to_crs(CRS)

lon = np.r_[np.linspace(-180, 180, 361), np.full(181, 180), np.linspace(180, -180, 361), np.full(181, -180)]
lat = np.r_[np.full(361, 90), np.linspace(90, LAT_MIN, 181), np.full(361, LAT_MIN), np.linspace(LAT_MIN, 90, 181)]
sphere = gpd.GeoSeries([Polygon(zip(lon, lat))], crs=4326).to_crs(CRS)
SX0, SY0, SX1, SY1 = sphere.total_bounds
PAD = 0.6e5
TR = pyproj.Transformer.from_crs(4326, CRS, always_xy=True)

ALLSHIFT = "idonly" not in V
shift_col = "med_shift_all" if ALLSHIFT else "med_shift"
n_shift = (u.n_corr + u.n_match) if ALLSHIFT else u.n_corr
ok_a = (u.status == "ok") & enough_plants(u.n_ref, a.min_ref)
ok_b = (u.status == "ok") & enough_plants(n_shift, a.min_corr)
cls3 = lambda v: np.digitize(v, BRK)
u["biv"] = None
u.loc[ok_a, "biv"] = [
    BIV[3 * cls3(m) + cls3(n)] for n, m in zip(u.loc[ok_a, "p_new"].fillna(0), u.loc[ok_a, "p_missing"].fillna(0))
]
u["shc"] = None
u.loc[ok_b, "shc"] = [SHIFT_COL[i] for i in np.digitize(u.loc[ok_b, shift_col], SHIFT_BRK)]
unc = u.status == "uncovered"
few_a, few_b = (u.status == "ok") & ~ok_a, (u.status == "ok") & ~ok_b


# ------------------------------------------------------------------ country glyph data (spikes / circles)
from shapely.geometry import Point

cu = gpd.read_file(a.countries or a.plants.replace("plants_with_units_", "units_country_").replace(".csv", ".gpkg"))
cu = cu.to_crs(CRS).set_index("iso2")
covered = set(cu.index[cu.status == "ok"])
pl = pd.read_csv(a.plants, keep_default_na=False, na_values=[""])
pl = pl[pl.unit.isin(covered)].copy()
pl["x"], pl["y"] = TR.transform(pl.lon.values, pl.lat.values)
grp = pl.groupby("unit")
C = pd.DataFrame(
    {
        k: (pl.cls == c).groupby(pl.unit).sum()
        for k, c in (("n_new", "new"), ("n_missing", "missing"), ("n_corr", "corrected"), ("n_match", "matched"))
    }
)
C["x"], C["y"] = grp.x.mean(), grp.y.mean()
for iso, r in C.iterrows():  # plant-weighted centre; if it falls outside the country, the plant nearest to it
    if not cu.geometry[iso].contains(Point(r.x, r.y)):
        q = pl[pl.unit == iso]
        j = np.argmin((q.x - r.x) ** 2 + (q.y - r.y) ** 2)
        C.loc[iso, ["x", "y"]] = q.x.iat[j], q.y.iat[j]
C["n_shift"] = C.n_corr + C.n_match


# ------------------------------------------------------------------ map drawing
def draw_panel(ax, fill_mask, fill_col, few_mask, spikes=False, extent=None, overlay=False, glyph=None):
    sphere.plot(ax=ax, facecolor="white", edgecolor="none", zorder=0)
    land.plot(ax=ax, facecolor=NODATA, edgecolor="none", zorder=1)
    if unc.any():
        u[unc].plot(ax=ax, facecolor=NODATA, edgecolor=HATCH_EDGE, hatch="//////", linewidth=0, zorder=2)
    if few_mask.any():
        u[few_mask].plot(ax=ax, facecolor="white", edgecolor=HATCH_EDGE, hatch=FEW_HATCH, linewidth=0, zorder=2)
    if spikes:
        u[fill_mask].plot(ax=ax, color="#f4f4f4", edgecolor="none", zorder=3)
    else:
        u[fill_mask].plot(ax=ax, color=u.loc[fill_mask, fill_col], edgecolor="none", zorder=3)
    lw = 0.12 if a.label == "Countries" else 0.07
    u.boundary.plot(ax=ax, color="white", linewidth=lw, zorder=4)
    if rivers is not None:
        rivers.plot(ax=ax, color="#7d9fb8", linewidth=0.15, alpha=0.8, zorder=5)
    land.boundary.plot(ax=ax, color="#b0b0b0", linewidth=0.15, zorder=5)
    if extent is None:
        sphere.boundary.plot(ax=ax, color="#9a9a9a", linewidth=0.35, zorder=6)
    if spikes or overlay:
        draw_spikes(ax, extent, dark=overlay)
    if glyph == "spikes":
        draw_corr_spikes(ax, extent)
    elif glyph == "circles":
        draw_circles(ax, extent)
    if extent is None:
        ax.set_xlim(SX0 - PAD, SX1 + PAD)
        ax.set_ylim(SY0 - PAD, SY1 + PAD)
    else:
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(extent is not None)
        s.set_linewidth(0.35)
        s.set_color("#9a9a9a")


SPIKE_MAX = 3.6e6  # map units (m) for the largest count
SPIKE_OVER = (
    "#fde725",
    "#1a1a1a",
)  # new, missing over the bivariate fill: min dE 11.8 to every fill (normal/protan/deutan)
SPIKE_OVER_EDGE = ("#3a3a3a", "white")
CORR_C, CORR_EDGE = "#2ca25f", "white"  # corrected plants: min dE 15.8 to every purple class (normal/protan/deutan)
CIRC_RMAX = 7.5e5  # map units (m): radius of the largest circle
CIRC_RMIN = 1.0e5  # smallest drawn radius, so countries with few plants stay visible


def spike_k():
    vmax = max(C.n_new.max(), C.n_missing.max())
    return SPIKE_MAX / np.sqrt(vmax)


def circ_r(n, extent=None):
    r = np.maximum(CIRC_RMAX * np.sqrt(np.asarray(n, float) / C.n_corr.max()), CIRC_RMIN)
    return r * (0.45 if extent is not None else 1.0)


def draw_spikes(ax, extent, dark=False):
    """New (up) and missing (down) plants per country, at the country's plant-weighted centre."""
    k = spike_k()
    half = 1.1e5 if extent is None else 0.45e5
    if extent is not None:
        k *= 0.45
    for _, r in C.sort_values("y", ascending=False).iterrows():
        x, y = r.x, r.y
        cols = SPIKE_OVER if dark else (NEW_C, MISS_C)
        edges = SPIKE_OVER_EDGE if dark else ("white", "white")
        for n, col, ec, sg in ((r.n_new, cols[0], edges[0], 1), (r.n_missing, cols[1], edges[1], -1)):
            if n > 0:
                ax.fill(
                    [x - half, x, x + half],
                    [y, y + sg * k * np.sqrt(n), y],
                    color=col,
                    alpha=1.0 if dark else 0.92,
                    lw=0.25 if dark else 0.2,
                    ec=ec,
                    zorder=7,
                )


def draw_corr_spikes(ax, extent):
    """Corrected plants per country (up), same height scale as the new/missing spikes."""
    k = spike_k() * (0.45 if extent is not None else 1.0)
    half = 1.1e5 if extent is None else 0.45e5
    for _, r in C[C.n_corr > 0].sort_values("y", ascending=False).iterrows():
        ax.fill(
            [r.x - half, r.x, r.x + half],
            [r.y, r.y + k * np.sqrt(r.n_corr), r.y],
            color=CORR_C,
            ec=CORR_EDGE,
            lw=0.25,
            zorder=7,
        )


def draw_circles(ax, extent):
    """Circle area ~ corrected plants per country, at the country's plant-weighted centre."""
    from matplotlib.patches import Circle

    p = C[C.n_corr > 0].sort_values("n_corr", ascending=False)  # big ones first, small ones on top
    for (_, r), rad in zip(p.iterrows(), circ_r(p.n_corr, extent)):
        ax.add_patch(
            Circle((r.x, r.y), rad, fc=CORR_C, ec="#1f4d33", lw=0.3 if extent is None else 0.25, alpha=0.85, zorder=7)
        )


def europe_inset(ax_map, panel):
    x0, y0 = TR.transform(-11, 34.5)
    x1, _ = TR.transform(33, 34.5)
    _, y1 = TR.transform(10, 71.5)
    ext = (x0, y0, x1, y1)
    fx, fy = (ext[2] - ext[0]), (ext[3] - ext[1])
    w = 0.205
    h = w * fy / fx * (ax_map.get_position().width * W) / (ax_map.get_position().height * H)
    ins = ax_map.inset_axes([0.004, 0.012, w, h], zorder=20)  # above the sphere outline and the spikes
    if panel == "a":
        draw_panel(ins, ok_a, "biv", few_a, spikes="spikes" in V, extent=ext, overlay=OVER)
    else:
        draw_panel(ins, ok_b, "shc", few_b, extent=ext, glyph=SH_GLYPH)
    ax_map.add_patch(Rectangle((x0, y0), fx, fy, fill=False, lw=0.35, ec="#555555", zorder=8))
    ins.text(
        0.03,
        0.97,
        "Europe",
        transform=ins.transAxes,
        fontsize=FS2,
        color=INK2,
        va="top",
        zorder=9,
        bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8),
    )


def lat_profile(ax_map, ax_p, panel, plants):
    edges = np.arange(LAT_MIN, 86, 5.0)
    yy = np.array([TR.transform(0, e)[1] for e in edges])
    pl = plants[plants.lat >= LAT_MIN]
    if panel == "a":
        for cls, col, sg in (("new", NEW_C, 1), ("missing", MISS_C, -1)):
            h, _ = np.histogram(pl.lat[pl.cls == cls], bins=edges)
            ax_p.barh(yy[:-1], sg * h, height=np.diff(yy) * 0.8, align="edge", color=col, lw=0)
        ax_p.axvline(0, color=INK2, lw=0.3)
        lim = np.abs(ax_p.get_xlim()).max()
        ax_p.set_xlim(-lim, lim)
        t = [v for v in (500, 1000, 2000, 5000, 10000) if v <= lim * 0.9][-1]
        ax_p.set_xticks([-t, 0, t])
        ax_p.set_xticklabels([f"{t:,}", "0", f"{t:,}"])
        ax_p.text(0.98, 1.0, "new", transform=ax_p.transAxes, color=NEW_C, fontsize=FS2, ha="right", va="bottom")
        ax_p.text(0.02, 1.0, "missing", transform=ax_p.transAxes, color=MISS_C, fontsize=FS2, ha="left", va="bottom")
        ax_p.set_xlabel("Plants / 5° lat.", fontsize=FS2, labelpad=1)
    else:
        sel = pl.cls.isin(["corrected", "matched"]) if ALLSHIFT else (pl.cls == "corrected")
        c = pl[sel]
        g = c.groupby(pd.cut(c.lat, edges), observed=False).shift_m
        med, q1, q3, n = g.median().values, g.quantile(0.25).values, g.quantile(0.75).values, g.size().values
        mid = (yy[:-1] + yy[1:]) / 2
        k = enough_plants(n, a.min_corr)
        ax_p.fill_betweenx(mid[k], q1[k], q3[k], color=SHIFT_COL[2], alpha=0.5, lw=0)
        ax_p.plot(med[k], mid[k], color=SHIFT_COL[5], lw=0.7, marker="o", ms=1.4, mew=0)
        ax_p.set_xscale("log")
        ax_p.set_xlim(8, 3000)
        ax_p.set_xticks([10, 100, 1000])
        ax_p.set_xticklabels(["10", "100", "1k"])
        ax_p.grid(axis="x", lw=0.2, color="#e3e3e3")
        ax_p.set_xlabel("Shift (m)", fontsize=FS2, labelpad=1)
    ax_p.set_ylim(ax_map.get_ylim())
    ax_p.set_yticks([])
    for s in ("top", "right", "left"):
        ax_p.spines[s].set_visible(False)
    ax_p.spines["bottom"].set_linewidth(0.3)
    ax_p.tick_params(axis="x", length=1.5, width=0.3, labelsize=FS2, pad=1)


# ------------------------------------------------------------------ layout in mm
W, M, GAP = 180.0, 1.0, 3.0
PROF = 12.0 if "latprofile" in V else 0.0
PW = (W - 2 * M - GAP) / 2
MAPW = PW - PROF - (1.0 if PROF else 0)
MAPH = MAPW * (SY1 - SY0 + 2 * PAD) / (SX1 - SX0 + 2 * PAD)
TITLE, LEG = 4.0, 16.5
PROF_B = 4.0 if PROF else 0.0  # room for profile x-axis labels
H = TITLE + MAPH + PROF_B + LEG + 1.0
fig = plt.figure(figsize=(W * MM, H * MM))
rect = lambda x, y, w, h: [x / W, y / H, w / W, h / H]

axes = {}
for panel in "ab":
    x = M + POS[panel] * (PW + GAP)
    axes[panel] = fig.add_axes(rect(x, LEG + PROF_B, MAPW, MAPH))
    axes[panel + "L"] = fig.add_axes(rect(x, 0.5, PW, LEG))
    if PROF:
        axes[panel + "P"] = fig.add_axes(rect(x + MAPW + 1.0, LEG + PROF_B, PROF, MAPH))

draw_panel(axes["a"], ok_a, "biv", few_a, spikes="spikes" in V, overlay=OVER)
draw_panel(axes["b"], ok_b, "shc", few_b, glyph=SH_GLYPH)
if "europe" in V:
    europe_inset(axes["a"], "a")
    europe_inset(axes["b"], "b")
if PROF:
    plants = pd.read_csv(a.plants)
    lat_profile(axes["a"], axes["aP"], "a", plants)
    lat_profile(axes["b"], axes["bP"], "b", plants)

titles = {
    "a": "Plants added and missing relative to HydroWASTE/UWWTD",
    "b": ("Location shift relative to HydroWASTE/UWWTD" if ALLSHIFT else "Location shift, ID-linked plants only"),
}
for panel in "ab":
    x = M + POS[panel] * (PW + GAP)
    fig.text(x / W, (H - 0.6) / H, LETTER[panel], fontsize=8, fontweight="bold", va="top", ha="left", color=INK)
    fig.text((x + 3.2) / W, (H - 0.95) / H, titles[panel], fontsize=6.5, va="top", ha="left", color=INK)
    fig.text(
        (x + PW) / W,
        (H - 1.2) / H,
        f"by {a.label[0].lower() + a.label[1:] if a.label == 'Countries' else a.label}",
        fontsize=FS,
        va="top",
        ha="right",
        color=INK2,
    )


# ------------------------------------------------------------------ legends (mm coordinates)
def mm_axes(ax):
    bb = ax.get_position()
    w, h = bb.width * W, bb.height * H
    ax.axis("off")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    return w, h


def swatch(ax, x, y, w, h, fc, hatch=None, ec=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or "none", hatch=hatch, lw=0))


def nodata_key(L, x, ytop, few_label):
    items = [
        (NODATA, "//////", HATCH_EDGE, "Not covered by HeiGIT"),
        ("white", FEW_HATCH, HATCH_EDGE, few_label),
        (NODATA, None, None, "No reference plants"),
    ]
    for k, (fc, hat, ec, t) in enumerate(items):
        yy = ytop - k * 3.5
        swatch(L, x, yy, 2.6, 2.6, fc, hat, ec)
        L.text(x + 3.6, yy + 1.3, t, va="center", fontsize=FS, color=INK)


L = axes["aL"]
lw_, lh_ = mm_axes(L)
if "spikes" in V:
    for k, (col, t) in enumerate(
        ((NEW_C, "New plants (not in HydroWASTE/UWWTD)"), (MISS_C, "Missing plants (not found)"))
    ):
        y = lh_ - 4 - k * 4.2
        L.fill([6, 7, 8], [y - 1.2, y + 1.6, y - 1.2] if k == 0 else [y + 1.6, y - 1.2, y + 1.6], color=col, lw=0)
        L.text(9.5, y + 0.2, t, va="center", fontsize=FS, color=INK)
    vmax = max(C.n_new.max(), C.n_missing.max())
    mm_per_m = MAPW / (SX1 - SX0 + 2 * PAD)
    base_y = 1.2
    for k, n in enumerate((100, 1000)):
        hgt = SPIKE_MAX / np.sqrt(vmax) * np.sqrt(n) * mm_per_m
        xx = 38 + k * 6
        L.fill(
            [xx - 1.1e5 * mm_per_m, xx, xx + 1.1e5 * mm_per_m], [base_y, base_y + hgt, base_y], color="#9a9a9a", lw=0
        )
        L.text(xx + 0.9, base_y + hgt, f"{n:,}", fontsize=FS2, color=INK2, va="center", ha="left")
    L.text(6, base_y + 1.5, "Per country, spike height ~ √plants", fontsize=FS2, color=INK2, va="center")
else:
    c = 3.0
    x0, y0 = 14.0, 4.4
    for j in range(3):
        for i in range(3):
            swatch(L, x0 + i * c, y0 + j * c, c - 0.25, c - 0.25, BIV[3 * j + i])
    arrow = dict(arrowstyle="-|>", lw=0.4, color=INK, mutation_scale=3.5, shrinkA=0, shrinkB=0)
    L.annotate("", xy=(x0 + 3 * c + 1.4, y0 - 0.8), xytext=(x0 - 0.2, y0 - 0.8), arrowprops=arrow)
    L.annotate("", xy=(x0 - 0.8, y0 + 3 * c + 1.4), xytext=(x0 - 0.8, y0 - 0.2), arrowprops=arrow)
    for k, t in enumerate(["5", "25"]):
        L.text(x0 + (k + 1) * c - 0.12, y0 - 1.3, t, ha="center", va="top", fontsize=FS2, color=INK2)
        L.text(x0 - 1.3, y0 + (k + 1) * c - 0.12, t, ha="right", va="center", fontsize=FS2, color=INK2)
    L.text(x0 + 1.5 * c, y0 - 3.4, "New (% of HeiGIT plants)", ha="center", va="top", fontsize=FS, color=INK)
    L.text(
        x0 - 5.6,
        y0 + 1.5 * c,
        "Missing (% of\nreference plants)",
        ha="center",
        va="center",
        fontsize=FS,
        color=INK,
        rotation=90,
        linespacing=1.0,
    )
    if OVER:
        # spike key next to the bivariate key: the diverging glyph as on the map, plus two reference sizes
        mmpm = MAPW / (SX1 - SX0 + 2 * PAD)
        hw = 1.1e5 * mmpm
        kmm = spike_k() * mmpm
        xs, yb = 31.0, 8.6
        h1k = kmm * np.sqrt(300)
        L.fill([xs - hw, xs, xs + hw], [yb, yb + h1k, yb], color=SPIKE_OVER[0], ec=SPIKE_OVER_EDGE[0], lw=0.25)
        L.fill([xs - hw, xs, xs + hw], [yb, yb - h1k, yb], color=SPIKE_OVER[1], ec=SPIKE_OVER_EDGE[1], lw=0.25)
        L.text(xs + 1.2, yb + h1k * 0.55, "new", fontsize=FS, color=INK, va="center")
        L.text(xs + 1.2, yb - h1k * 0.55, "missing", fontsize=FS, color=INK, va="center")
        for kk, n in enumerate((100, 1000)):
            xr = 42.0 + kk * 4.6
            hgt = kmm * np.sqrt(n)
            L.fill([xr - hw, xr, xr + hw], [yb - 2.4, yb - 2.4 + hgt, yb - 2.4], color="#8a8a8a", lw=0)
            L.text(xr, yb - 2.9, f"{n:,}", fontsize=FS2, color=INK2, ha="center", va="top")
        L.text(
            29.6,
            lh_ - 0.4,
            "Spikes per country,\nheight ~ √plants",
            fontsize=FS2,
            color=INK2,
            va="top",
            linespacing=1.0,
        )
    else:
        L.text(x0 + 3 * c + 1.0, y0 + 0.5 * c, "mostly additions", fontsize=FS2, color=INK2, va="center")
        L.text(x0 + 3 * c + 1.0, y0 + 2.5 * c, "additions and omissions", fontsize=FS2, color=INK2, va="center")
        L.text(x0 - 0.1, y0 + 3 * c + 0.5, "mostly omissions", fontsize=FS2, color=INK2, ha="left", va="bottom")
nodata_key(L, 54, lh_ - 4.4, f"< {a.min_ref} reference plants")

L = axes["bL"]
lw_, lh_ = mm_axes(L)
x0, y0, w = (3.0, 8.2, 5.1) if SH_GLYPH else (4.0, 8.2, 6.6)
for k, col in enumerate(SHIFT_COL):
    swatch(L, x0 + k * w, y0, w - 0.25, 2.6, col)
for k, b in enumerate(SHIFT_BRK):
    L.text(x0 + (k + 1) * w - 0.12, y0 - 0.6, f"{b:,}", ha="center", va="top", fontsize=FS2, color=INK2)
lab = "Median shift of corrected plants (m)"
L.text(x0 + 3 * w, y0 - 3.4, lab, ha="center", va="top", fontsize=FS, color=INK)
mmpm = MAPW / (SX1 - SX0 + 2 * PAD)
if SH_GLYPH == "spikes":
    kmm = spike_k() * mmpm
    hw = 1.1e5 * mmpm
    L.text(35.0, lh_ - 0.4, "Corrected plants\nper country", fontsize=FS2, color=INK2, va="top", linespacing=1.0)
    for kk, n in enumerate((100, 1000)):
        xr = 37.0 + kk * 5.0
        yb = 3.2
        hgt = kmm * np.sqrt(n)
        L.fill([xr - hw, xr, xr + hw], [yb, yb + hgt, yb], color=CORR_C, ec=CORR_EDGE, lw=0.25)
        L.text(xr, yb - 0.5, f"{n:,}", fontsize=FS2, color=INK2, ha="center", va="top")
elif SH_GLYPH == "circles":
    L.text(35.0, lh_ - 0.4, "Corrected plants\nper country", fontsize=FS2, color=INK2, va="top", linespacing=1.0)
    n_floor = (CIRC_RMIN / CIRC_RMAX) ** 2 * C.n_corr.max()  # below this all circles have the minimum size
    yc = 6.4
    for xc, n, t in ((37.2, 1, f"< {int(np.ceil(n_floor / 10) * 10)}"), (41.6, 1000, "1,000"), (47.0, 10000, "10,000")):
        r = circ_r([n])[0] * mmpm
        L.add_patch(mpl.patches.Circle((xc, yc), r, fc=CORR_C, ec="#1f4d33", lw=0.3, alpha=0.85))
        L.text(xc, yc - 2.6, t, fontsize=FS2 - 0.6, color=INK2, ha="center", va="top")
nodata_key(L, 54, lh_ - 4.4, f"< {a.min_corr} corrected plants")

fig.savefig(a.out + ".pdf")
fig.savefig(a.out + ".png", dpi=600)
print("wrote", a.out, f"({W:.0f} x {H:.1f} mm)")
