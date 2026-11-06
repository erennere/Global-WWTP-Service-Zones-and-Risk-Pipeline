"""Second global map as a two-panel figure, 180 mm wide (Nature double column):
  first panel   cartogram, unit area ~ urban population not served
  second panel  --d truearea (default): the same units at true area, with an Asia inset (66-126 E, 9 S-41 N,
                1.5x) in the empty South Pacific and, with --nightlights, night-time lights on top
                --d tanks: the tanks map from map 1's data instead (round-tank share + plant surroundings,
                see figures/fig1/match/tanks_panel.py); then each panel has its own legend
Colour = non-served ratio R = 1 - P_served / (P_served + P_non), as in 08_plot_cartogram.py.

Usage: python 09_plot_pair.py <level 3|4|country> <out_stem> [--above50] [--scale ...] [--letters cd]
                              [--d truearea|tanks] [--nightlights <GeoTIFF>] [tanks options]
  Whole range (0-100 %), diverging teal (served) / brown (not served) around 50 %:
    classes     0 / 10 / 25 / 50 / 75 / 90 / 100 (the default; as in 08_plot_cartogram.py)
    linear      ten 10-point classes
    exp         each class halves the remaining gap to 0 % or to 100 %: 0 / 6.25 / 12.5 / 25 / 50 / 75 / 87.5 / 93.75 / 100
    quantile    three classes below and three above 50 %, each holding the same number of units
    continuous  one continuous ramp from 0 to 100 %
  --above50  colour only units where more than half of the urban population is not served (50-100 %);
             the rest are grey. Scales: linear (default), exp, quantile, continuous (see 08_plot_cartogram.py).
  country    reads svc/service_country.gpkg (05b_service_country.py) and svc/cartogram_country.gpkg
             (07_cartogram.py country); the cartogram names the countries with the most non-served people.
  --nightlights  NASA Black Marble colour GeoTIFF (e.g. BlackMarble_2016_01deg_geo.tif), true-area panel only.
             Light = warm excess (red - blue) plus brightness above 90, scaled to its 99th percentile,
             square-root stretched and spread by one pixel; drawn where above a quarter of that scale, as
             magenta specks, more opaque where brighter (rasterised at 600 dpi inside the vector PDF).
  --d tanks  needs --map1 (the map 1 work folder, figure1.paths.match_dir), --attrs (HeiGIT GeoPackage) and, for basin levels,
             --hybas-dir; the other tanks options are those of 11_plot_tanks.py.
"""
import argparse, os, sys, textwrap
import numpy as np, geopandas as gpd, pyproj
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex
from shapely.geometry import box, Polygon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig1_config import PATHS, TAG  # noqa: E402  (config.yaml, section figure1)

ap = argparse.ArgumentParser()
ap.add_argument("level", choices=["3", "4", "country"]); ap.add_argument("out")
ap.add_argument("--above50", action="store_true")
ap.add_argument("--scale", default=None, choices=["classes", "linear", "exp", "quantile", "continuous"])
ap.add_argument("--letters", default="ab", help="the two panel letters, e.g. cd in the combined figure")
ap.add_argument("--d", default="truearea", choices=["truearea", "tanks"], help="what the second panel shows")
ap.add_argument("--nightlights", default=None, help="night-lights GeoTIFF over the true-area panel")
ap.add_argument("--map1", default=PATHS["match_dir"], help="map 1 work folder (tanks panel); config figure1.paths.match_dir")
ap.add_argument("--tag", default=TAG, help="radius tag of the map 1 files (tanks panel); from figure1.match_radius_m")
TANKS = "--d" in sys.argv and sys.argv[sys.argv.index("--d") + 1] == "tanks"
if TANKS:                                    # the tanks module lives with map 1's scripts
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "match"))  # tanks_panel.py
    import tanks_panel as tp
    tp.add_args(ap)
a = ap.parse_args()
SCALE = a.scale or ("linear" if a.above50 else "classes")
assert not (a.above50 and SCALE == "classes"), "--above50 takes linear, exp, quantile or continuous"
assert len(a.letters) == 2, "--letters takes two letters"
COUNTRY = a.level == "country"
NAME = "country" if COUNTRY else f"hybas{a.level}"
UNIT_LABEL = "Countries" if COUNTRY else f"HydroBASINS level {a.level}"
UNIT, UNITS = ("country", "countries") if COUNTRY else ("basin", "basins")

MM = 1 / 25.4
mpl.rcParams.update({"font.family": "Liberation Sans", "font.size": 6, "axes.linewidth": 0.4,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": 0.35})
INK, INK2 = "#1a1a1a", "#5a5a5a"
HATCH_EDGE = "#9a9a9a"
NODATA = "#ececec" if a.above50 else "#dedede"
FS, FS2 = 5.4, 5.0
R_BRK = [0.10, 0.25, 0.50, 0.75, 0.90]
R_COL = ["#01665e", "#4fa69d", "#93cdc3", "#dcbc78", "#c98d3b", "#8c510a"]   # low -> high non-served; adj dE >= 9.7
R_LAB = ["0", "10", "25", "50", "75", "90", "100"]
A50_COL = ["#f6e8c3", "#dfc27d", "#bf812d", "#8c510a", "#543005"]   # BrBG brown side; adjacent dE >= 11.5
LOW = "#cfcfcf"                                                     # 50 % or less not served
LIGHT_C = "#e0168f"                                                 # night lights: outside the teal/brown/grey classes
CMAP = LinearSegmentedColormap.from_list("a50", A50_COL)
# whole-range ramps (ColorBrewer BrBG without its white centre): dark teal at 0 % -> light at 50 %; light brown -> dark at 100 %
TEAL = LinearSegmentedColormap.from_list("teal", ["#003c30", "#01665e", "#35978f", "#80cdc1", "#c7eae5"])
BROWN = LinearSegmentedColormap.from_list("brown", ["#f6e8c3", "#dfc27d", "#bf812d", "#8c510a", "#543005"])


def div(v):
    return to_hex(TEAL(v / 0.5)) if v <= 0.5 else to_hex(BROWN((v - 0.5) / 0.5))


LABELS = {452: "Ganges–\nBrahmaputra", 453: "Peninsular\nIndia", 456: "Indus", 434: "Yangtze", 515: "Java",
          172: "Nile", 431: "North\nChina"}
N_COUNTRY_LABELS = 7
CLIP = box(-180, -58, 180, 90)
ASIA = (66, -9, 126, 41)            # lon0, lat0, lon1, lat1 of the true-area inset
ASIA_ZOOM = 1.5
TR = pyproj.Transformer.from_crs(4326, 8857, always_xy=True)


def ratio(c):
    return np.where((c.P_served + c.P_non) > 0, c.P_non / (c.P_served + c.P_non), np.nan)


# ------------------------------------------------------------------ data
cart = gpd.read_file(f"svc/cartogram_{NAME}.gpkg")
true = gpd.read_file(f"svc/service_{NAME}.gpkg").clip(CLIP).to_crs(8857)
for c in (cart, true):
    c["geometry"] = c.geometry.make_valid()
    c["R"] = ratio(c)
lon = np.r_[np.linspace(-180, 180, 361), np.full(181, 180), np.linspace(180, -180, 361), np.full(181, -180)]
lat = np.r_[np.full(361, 90), np.linspace(90, -58, 181), np.full(361, -58), np.linspace(-58, 90, 181)]
sphere = gpd.GeoSeries([Polygon(zip(lon, lat))], crs=4326).to_crs(8857)
land = gpd.read_file("data/ne_50m_land.geojson").clip(CLIP).to_crs(8857)

# class breaks, computed once from the true-area units so both panels use the same classes
if a.above50:
    hi = (true.status == "ok") & (true.R > 0.5) & (true.P_non > 0)
    A_BRK = {"linear": [0.6, 0.7, 0.8, 0.9], "exp": [1 - 0.5 ** k for k in (2, 3, 4, 5)],
             "quantile": list(np.unique(np.round(np.quantile(true.loc[hi, "R"], [.2, .4, .6, .8]), 4))),
             "continuous": None}[SCALE]
    COLS = None if A_BRK is None else (A50_COL if len(A_BRK) == 4 else
                                       [to_hex(CMAP(t)) for t in np.linspace(0, 1, len(A_BRK) + 1)])
elif SCALE not in ("classes", "continuous"):
    shown = (true.status == "ok") & (true.P_non > 0)          # units that have area in the cartogram
    if SCALE == "linear":
        lo, hi = [0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 0.9]
    elif SCALE == "exp":
        lo, hi = [1 / 16, 1 / 8, 1 / 4], [3 / 4, 7 / 8, 15 / 16]
    else:                                                    # quantile, separately below and above 50 %
        lo = list(np.round(np.quantile(true.loc[shown & (true.R <= 0.5), "R"], [1 / 3, 2 / 3]), 3))
        hi = list(np.round(np.quantile(true.loc[shown & (true.R > 0.5), "R"], [1 / 3, 2 / 3]), 3))
    F_BRK = lo + [0.5] + hi
    F_COL = ([to_hex(TEAL((j + 0.5) / (len(lo) + 1))) for j in range(len(lo) + 1)] +
             [to_hex(BROWN((j + 0.5) / (len(hi) + 1))) for j in range(len(hi) + 1)])


def colour(c, is_true):
    """Fill colour per unit; returns masks (coloured, uncovered, grey)."""
    c["col"] = None
    if a.above50:
        low = (c.status == "ok") & c.R.notna() & (c.R <= 0.5)
        ok = (c.status == "ok") & (c.R > 0.5) & (c.P_non > 0)
        if A_BRK is None:
            c.loc[ok, "col"] = [to_hex(CMAP((r - 0.5) / 0.5)) for r in c.loc[ok, "R"]]
        else:
            c.loc[ok, "col"] = [COLS[i] for i in np.digitize(c.loc[ok, "R"], A_BRK, right=True)]
        c.loc[low, "col"] = LOW
        ok = ok | low
    else:
        ok = (c.status == "ok") & (c.R.notna() if is_true else (c.P_non > 0))
        if SCALE == "classes":
            c.loc[ok, "col"] = [R_COL[i] for i in np.digitize(c.loc[ok, "R"], R_BRK)]
        elif SCALE == "continuous":
            c.loc[ok, "col"] = [div(r) for r in c.loc[ok, "R"]]
        else:
            c.loc[ok, "col"] = [F_COL[i] for i in np.digitize(c.loc[ok, "R"], F_BRK, right=True)]
    unc = c.status == "uncovered"
    return ok, unc, ~ok & ~unc


def night_lights(path):
    """Night-light specks in Equal Earth metres: x, y, strength (0.25-1)."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            rgb = src.read([1, 2, 3]).astype(float)
            tf = src.transform
        ny, nx = rgb.shape[1:]
        cols, rows = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
        lo_, la_ = tf * (cols, rows)
    except ImportError:                                       # no rasterio: global plate-carree grid assumed
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        rgb = np.asarray(Image.open(path).convert("RGB")).astype(float).transpose(2, 0, 1)
        ny, nx = rgb.shape[1:]
        lo_, la_ = np.meshgrid(-180 + (np.arange(nx) + 0.5) * 360 / nx, 90 - (np.arange(ny) + 0.5) * 180 / ny)
    r, g, b = rgb
    light = np.clip(r - b, 0, None) + np.clip(rgb.mean(0) - 90, 0, None)
    s = light / max(np.quantile(light[light > 0], 0.99), 1e-9)
    s = np.sqrt(np.clip(s, 0, 1))
    from scipy import ndimage
    s = np.maximum(s, 0.85 * ndimage.maximum_filter(s, size=2))   # spread a little: towns stay visible in print
    k = (s > 0.25) & (la_ >= -58)
    x, y = TR.transform(lo_[k], la_[k])
    return x, y, s[k]


okA, uncA, greyA = colour(cart, False)
xa0, ya0, xa1, ya1 = cart[okA & (cart.P_non > 0)].total_bounds
pad = 0.02 * (xa1 - xa0); xa0, xa1, ya0, ya1 = xa0 - pad, xa1 + pad, ya0 - pad, ya1 + pad
LIGHTS = None
if TANKS:
    if a.plants == f"agg/plants_with_units_{TAG}.csv":       # the tanks default is relative to the map 1 folder
        a.plants = os.path.join(a.map1, "agg", f"plants_with_units_{a.tag}.csv")
    a.ne_data = a.ne_data or os.path.join(a.map1, "data")
    T = tp.Tanks(a, os.path.join(a.map1, "agg", f"units_{NAME}_{a.tag}.gpkg"), UNIT_LABEL)
    xb0, yb0, xb1, yb1 = T.extent
else:
    okB, uncB, greyB = colour(true, True)
    xb0, yb0, xb1, yb1 = sphere.total_bounds
    pad = 0.02 * (xb1 - xb0); xb0, xb1, yb0, yb1 = xb0 - pad, xb1 + pad, yb0 - pad, yb1 + pad
    if a.nightlights:
        LIGHTS = night_lights(a.nightlights)
        print(f"night lights: {len(LIGHTS[0]):,} specks")

# ------------------------------------------------------------------ layout (mm)
W, M, GAP = 180.0, 1.0, 3.0
PW = (W - 2 * M - GAP) / 2
hA = PW * (ya1 - ya0) / (xa1 - xa0); hB = PW * (yb1 - yb0) / (xb1 - xb0)
ROW = max(hA, hB)
TITLE = 4.5
LEG = tp.LEGEND_H + 0.5 if TANKS else 13.5
H = TITLE + ROW + LEG + 0.5
fig = plt.figure(figsize=(W * MM, H * MM))
axA = fig.add_axes([M / W, (LEG + (ROW - hA) / 2) / H, PW / W, hA / H])
axB = fig.add_axes([(M + PW + GAP) / W, (LEG + (ROW - hB) / 2) / H, PW / W, hB / H])


def draw(ax, c, ok, unc, grey, is_true, extent=None):
    lw = 0.12 if (COUNTRY or a.level == "3") else 0.07
    if is_true:
        sphere.plot(ax=ax, facecolor="white", edgecolor="none", zorder=0)
        land.plot(ax=ax, facecolor=NODATA, edgecolor="none", zorder=0.5)
        if extent is None:
            sphere.boundary.plot(ax=ax, color="#9a9a9a", linewidth=0.35, zorder=5)
    c[grey].plot(ax=ax, facecolor=NODATA, edgecolor="none", zorder=1)
    if unc.any():
        c[unc].plot(ax=ax, facecolor=NODATA, edgecolor=HATCH_EDGE, hatch="//////", linewidth=0, zorder=1)
    c[ok].plot(ax=ax, color=c.loc[ok, "col"], edgecolor="none", zorder=2)
    c.boundary.plot(ax=ax, color="white", linewidth=lw, zorder=3)
    (land.boundary if is_true else c.dissolve().boundary).plot(ax=ax, color="#8a8a8a", linewidth=0.25, zorder=4)
    if is_true and LIGHTS is not None:
        x, y, s = LIGHTS
        if extent is not None:
            k = (x >= extent[0]) & (x <= extent[2]) & (y >= extent[1]) & (y <= extent[3]); x, y, s = x[k], y[k], s[k]
        rgba = np.tile(mpl.colors.to_rgba(LIGHT_C), (len(s), 1)); rgba[:, 3] = 0.25 + 0.75 * (s - 0.25) / 0.75
        ax.scatter(x, y, s=(0.5 if extent is None else 0.9) * (0.4 + s), c=rgba, lw=0, zorder=4.5, rasterized=True)
    if not is_true:                     # names on the cartogram, where the large units are readable
        if COUNTRY:
            top = c[ok & (c.P_non > 0)].sort_values("P_non", ascending=False).head(N_COUNTRY_LABELS)
            items = [(top.loc[[i]], textwrap.fill(str(n), 13)) for i, n in zip(top.index, top["name"])]
        else:
            c["p3"] = c.PFAF_ID.astype(str).str[:3].astype(int)
            items = [(c[(c.p3 == p3) & ok], lab) for p3, lab in LABELS.items()]
        for sub, lab in items:
            if sub.empty:
                continue
            pt = sub.geometry.union_all().representative_point()
            hit = sub[sub.geometry.contains(pt)]
            col = hit.col.iloc[0] if len(hit) else sub.loc[sub.geometry.area.idxmax(), "col"]
            r, g, b = mpl.colors.to_rgb(col)
            dark = (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.45
            ax.text(pt.x, pt.y, lab, fontsize=FS2, color="white" if dark else INK, ha="center", va="center",
                    linespacing=0.95, zorder=6)
    if extent is not None:
        ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s_ in ax.spines.values():
        s_.set_visible(extent is not None); s_.set_linewidth(0.35); s_.set_color("#9a9a9a")


def asia_inset(ax):
    """Asia enlarged 1.5x in the empty South Pacific, lower left, as the Europe inset of map 1."""
    bx = gpd.GeoSeries([box(*ASIA).segmentize(0.5)], crs=4326).to_crs(8857).total_bounds
    fx, fy = bx[2] - bx[0], bx[3] - bx[1]
    w = ASIA_ZOOM * fx / (xb1 - xb0); h = ASIA_ZOOM * fy / (yb1 - yb0)
    ins = ax.inset_axes([0.004, 0.012, w, h], zorder=20)
    draw(ins, true, okB, uncB, greyB, True, extent=bx)
    ax.add_patch(Rectangle((bx[0], bx[1]), fx, fy, fill=False, lw=0.35, ec="#555555", zorder=10))
    ins.text(0.03, 0.97, "Asia", transform=ins.transAxes, fontsize=FS2, color=INK2, va="top", zorder=11,
             bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8))


draw(axA, cart, okA, uncA, greyA, False); axA.set_xlim(xa0, xa1); axA.set_ylim(ya0, ya1); axA.axis("off")
if TANKS:
    T.draw(axB)
else:
    draw(axB, true, okB, uncB, greyB, True); axB.set_xlim(xb0, xb1); axB.set_ylim(yb0, yb1); axB.axis("off")
    asia_inset(axB)

tA = f"{UNIT.capitalize()} area scaled to the urban population not served"
tB = "Tank geometry and plant surroundings" if TANKS else "True area (Equal Earth projection)"
for k, (lab, t) in enumerate(((a.letters[0], tA), (a.letters[1], tB))):
    x = M + k * (PW + GAP)
    fig.text(x / W, (H - 0.5) / H, lab, fontsize=8, fontweight="bold", va="top", ha="left", color=INK)
    fig.text((x + 3.2) / W, (H - 0.85) / H, t, fontsize=6.5, va="top", ha="left", color=INK)
sub = f" · {UNITS} above 50 % not served" if a.above50 else ""
if TANKS:                               # both panels share the unit; it goes top right, the tanks legend is below
    fig.text((W - M) / W, (H - 1.1) / H, "by countries" if COUNTRY else f"by {UNIT_LABEL}", fontsize=FS, va="top",
             ha="right", color=INK2)
else:
    fig.text((W - 0.5) / W, 0.8 / H, f"{UNIT_LABEL}{sub}", fontsize=FS2, va="bottom", ha="right", color=INK2)

# ------------------------------------------------------------------ legend (mm)
L = fig.add_axes([0, 0, 1, LEG / H]); L.axis("off"); L.set_xlim(0, W); L.set_ylim(0, LEG)
TOP = LEG - 13.5                     # the map 2 legend fills the top 13.5 mm
lx, ly = 2.0, TOP + 7.5
fmt = lambda b: f"{100 * b:.1f}".rstrip("0").rstrip(".")
L.text(lx, ly + 3.0, "Urban population not served (%)", fontsize=FS, color=INK, va="bottom")
if a.above50:
    sw = 5.0
    L.add_patch(Rectangle((lx, ly), sw - 0.25, 2.4, facecolor=LOW, lw=0))
    L.text(lx + 0.5 * sw, ly - 0.5, "≤ 50", ha="center", va="top", fontsize=FS2, color=INK2)
    bx = lx + sw + 0.6
    if A_BRK is None:
        L.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP, extent=(bx, bx + 5 * sw - 0.25, ly, ly + 2.4),
                 aspect="auto", zorder=1)
        for k, t in enumerate(("", "60", "70", "80", "90", "100")):
            L.text(bx + k * (5 * sw - 0.25) / 5, ly - 0.5, t, ha="center", va="top", fontsize=FS2, color=INK2)
        L.set_xlim(0, W); L.set_ylim(0, LEG)
    else:
        wcl = 5 * sw / (len(A_BRK) + 1)
        for k, col in enumerate(COLS):
            L.add_patch(Rectangle((bx + k * wcl, ly), wcl - 0.25, 2.4, facecolor=col, lw=0))
        for k, t in enumerate([""] + [fmt(b) for b in A_BRK] + ["100"]):
            L.text(bx + k * wcl - 0.12, ly - 0.5, t, ha="center", va="top",
                   fontsize=FS2 - (0.5 if SCALE != "linear" else 0), color=INK2)
elif SCALE == "classes":
    sw = 5.4
    for k, col in enumerate(R_COL):
        L.add_patch(Rectangle((lx + k * sw, ly), sw - 0.25, 2.4, facecolor=col, lw=0))
    for k, t in enumerate(R_LAB):
        L.text(lx + k * sw - 0.12, ly - 0.5, t, ha="center", va="top", fontsize=FS2, color=INK2)
elif SCALE == "continuous":
    BW = 36.0
    v = np.linspace(0, 1, 512)
    L.imshow(np.linspace(0, 1, 512)[None, :], cmap=ListedColormap([div(x) for x in v]),
             extent=(lx, lx + BW, ly, ly + 2.4), aspect="auto", zorder=1)
    for t in (0, 25, 50, 75, 100):
        L.text(lx + BW * t / 100, ly - 0.5, str(t), ha="center", va="top", fontsize=FS2, color=INK2)
    L.set_xlim(0, W); L.set_ylim(0, LEG)
else:
    BW = 36.0; wcl = BW / len(F_COL)
    for k, col in enumerate(F_COL):
        L.add_patch(Rectangle((lx + k * wcl, ly), wcl - 0.2, 2.4, facecolor=col, lw=0))
    lab = lambda b: f"{100 * b:.2f}".rstrip("0").rstrip(".")
    for k, t in enumerate(["0"] + [lab(b) for b in F_BRK] + ["100"]):
        L.text(lx + k * wcl - 0.1, ly - 0.5, t, ha="center", va="top", fontsize=FS2 - (0.6 if len(F_COL) > 6 else 0.3),
               color=INK2)

# area key for the cartogram: squares holding 10 M ... 100 M non-served people at its scale
mm_per_m = PW / (xa1 - xa0)
okp = okA & (cart.P_non > 0)
m2_per_person = cart.geometry.area[okp].sum() / cart.P_non[okp].sum()
xs = 42.0
L.text(xs, TOP + 11.5, f"{a.letters[0]}: {UNIT} area proportional to\nurban population not served", fontsize=FS2,
       color=INK2, va="top", linespacing=1.0)
cand = [n for n in (10e6, 25e6, 50e6, 100e6) if np.sqrt(n * m2_per_person) * mm_per_m <= 5.6]
for n in (cand[0], cand[-1]) if len(cand) > 1 else cand:
    side = np.sqrt(n * m2_per_person) * mm_per_m
    L.add_patch(Rectangle((xs, TOP + 0.9), side, side, facecolor="#9a9a9a", lw=0))
    L.text(xs + side + 0.6, TOP + 0.9 + side / 2, f"{n / 1e6:,.0f} M", fontsize=FS2, color=INK2, ha="left", va="center")
    xs += side + 6.2
# no-data keys: under panel b, or under the cartogram's ramp when panel b is the tanks map
x2, y2 = (lx, TOP - 1.6) if TANKS else (M + PW + GAP + 2.0, TOP + 8.3)
L.add_patch(Rectangle((x2, y2), 2.6, 2.6, facecolor=NODATA, lw=0))
L.text(x2 + 3.4, y2 + 1.3, "no non-served urban population, or no data", fontsize=FS2, color=INK, va="center")
L.add_patch(Rectangle((x2, y2 - 3.8), 2.6, 2.6, facecolor=NODATA, edgecolor=HATCH_EDGE, hatch="//////", lw=0))
L.text(x2 + 3.4, y2 - 2.5, "not covered by HeiGIT", fontsize=FS2, color=INK, va="center")
if TANKS:
    T.legend(L, M + PW + GAP, LEG - 0.6)
    T.write_tables(a.out + "_tanks")
else:
    if not a.above50:
        L.text(x2 + 48.0, TOP + 11.5, f"In {a.letters[1]}, {UNITS} whose whole urban population\nis served (0 %) are "
               f"coloured; in {a.letters[0]} they have\nno area.", fontsize=FS2 - 0.4, color=INK2, va="top",
               linespacing=1.0)
    if LIGHTS is not None:
        L.scatter([x2 + 1.3], [y2 - 5.7], s=3, c=LIGHT_C, lw=0)
        L.text(x2 + 3.4, y2 - 5.7, "night-time lights (NASA Black Marble 2016)", fontsize=FS2, color=INK, va="center")

fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=600)
print("wrote", a.out, f"({W:.0f} x {H:.1f} mm)")
