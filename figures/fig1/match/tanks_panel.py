"""Panel 1d: tank geometry (fill) and plant surroundings (glyph per country). Imported by 11_plot_tanks.py
(standalone panel) and by figures/fig1/cartogram/09_plot_pair.py (--d tanks, panel d of the combined figure).

Fill, per unit (country or HydroBASINS L3/L4):
  round share  R = sum(round_area) / sum(round_area + rect_area)   over HeiGIT plants with detected tanks
  (weighting "plant": mean of the per-plant ratios instead, every plant counting once)
  coloured with >= min_fill plants with tanks (default 5); fewer -> dotted hatch, as in map 1.
Glyph, per country (on every level, as the spikes of map 1), from category_number of the HeiGIT plants:
  the most common class worldwide (the "ring class") is split off, because it holds most plants and would
  flatten a pie. Drawn for countries with >= min_glyph plants with a category (default 5).
  ringpie  outer ring: share of the country's plants in the ring class (dark arc, from 12 o'clock, clockwise);
           inner pie: how the other plants split over the other classes (shares of those plants only)
  core     area-true: the whole disc is the country's plants; the ring class is the grey outer part, the other
           classes a pie in the centre whose area is their share of all plants
  Glyphs sit at the centre of each country's plants (as in map 1), all at one size on paper; overlapping
  glyphs are pushed apart and joined to their country by a thin line. With the Europe inset, European
  countries are drawn in the inset only.

Inputs: plants_heigit_<tag>.csv (01_reconcile; fid per HeiGIT plant), plants_with_units_<tag>.csv
(02_aggregate; country per plant, same order), units_<level>_<tag>.gpkg, and the HeiGIT file with
round_area, rect_area, category_number, category_name, read by fid (GeoPackage) or from a CSV with a fid
column. Basin levels also need hybas_dir (as for 02b_aggregate_basins.py).
"""
import json, os, textwrap
import numpy as np, pandas as pd, geopandas as gpd, pyproj, pyogrio
from matplotlib.patches import Rectangle, Wedge, Circle
from shapely.geometry import box, Polygon, Point
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig1_config import MATCH_RADIUS_M, MIN_PLANTS, PATHS, TAG  # noqa: E402  (config.yaml, section figure1)

INK, INK2 = "#1a1a1a", "#5a5a5a"
NODATA, HATCH_EDGE = "#dedede", "#9a9a9a"
CRS = "EPSG:8857"
LAT_MIN = -58
FS, FS2 = 5.4, 5.0
# round share: sequential, light sand (few round tanks) -> dark brown (many), OKLCH hue 80, L .88 -> .38,
# C .105 -> .075; adjacent classes >= dE 9.6, every class >= dE 10.6 from NODATA (normal / protan / deutan)
RAMP = ["#fcd086", "#ddaf60", "#bc903e", "#9b7223", "#795710", "#573d06"]
# other classes: Okabe-Ito (minus black); min dE 15.6 normal, 9.6 protan, 7.6 deutan (green vs reddish purple,
# kept apart in the pie order). Ring class: dark grey arc on a white track.
CAT_COL = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
RING_C, TRACK_C, CORE_DOM_C, GLYPH_EDGE = "#5f5f5f", "#ffffff", "#b5b5b5", "#2f2f2f"
R_MM = 1.0             # glyph radius on paper (main map and inset alike)
RING_W = 0.30          # ring width, share of R (ringpie)
INSET_W = 0.25         # Europe inset width, share of the map width


def add_args(ap):
    """The tanks options, shared by 11_plot_tanks.py and 09_plot_pair.py."""
    g = ap.add_argument_group("panel d (tanks)")
    g.add_argument("--attrs", help="HeiGIT GeoPackage (read by fid) or CSV with a fid column")
    g.add_argument("--layer", default=None, help="layer of the HeiGIT GeoPackage (default: the first)")
    g.add_argument("--plants", default=f"agg/plants_with_units_{TAG}.csv")
    g.add_argument("--heigit-plants", default=None, help="plants_heigit_<tag>.csv (default: next to --plants)")
    g.add_argument("--countries", default=None, help="country units gpkg (default: derived from --plants)")
    g.add_argument("--ne-data", default=None, help="folder with ne_50m_land.geojson (default: data/)")
    g.add_argument("--hybas-dir", default=None, help="basin levels: hybas_lev09_v1c_attributes.csv, hybas_lev0X_v1c.gpkg")
    g.add_argument("--glyph", default="ringpie", choices=["ringpie", "core"])
    g.add_argument("--inset", default="europe", choices=["europe", "none"])
    g.add_argument("--weighting", default="area", choices=["area", "plant"])
    g.add_argument("--breaks", default="quantile",
                   help="class breaks of the round share: 'quantile' (equal numbers of coloured units per class, "
                        "from this level's data, rounded to whole %%) or comma-separated values in %%, e.g. 20,35,50,65,80")
    g.add_argument("--min-fill", type=int, default=MIN_PLANTS)
    g.add_argument("--min-glyph", type=int, default=MIN_PLANTS)
    g.add_argument("--ring-class", type=float, default=None, help="category_number split off as the ring (default: most common)")
    g.add_argument("--labels", default=None, help="JSON {category_number: label} to override category_name")
    g.add_argument("--tank-rivers", action="store_true", help="rivers on the tanks map")


def _sphere():
    lon = np.r_[np.linspace(-180, 180, 361), np.full(181, 180), np.linspace(180, -180, 361), np.full(181, -180)]
    lat = np.r_[np.full(361, 90), np.linspace(90, LAT_MIN, 181), np.full(361, LAT_MIN), np.linspace(LAT_MIN, 90, 181)]
    return gpd.GeoSeries([Polygon(zip(lon, lat))], crs=4326).to_crs(CRS)


def relax(xy, w, rad, gap, bounds, holes=(), iters=400, pull=0.03):
    """Push overlapping glyphs (radius rad, mm) apart; heavier countries (w) move less; weak pull home.
    Glyphs stay inside bounds (x0, y0, x1, y1) and out of the rectangles in holes (all in mm)."""
    home = xy.copy(); p = xy + np.random.default_rng(0).normal(0, 1e-3, xy.shape)
    need = 2 * rad + gap
    for it in range(iters):
        d = p[:, None, :] - p[None, :, :]
        dist = np.hypot(d[..., 0], d[..., 1]) + np.eye(len(p)) * 1e9
        ov = np.clip(need - dist, 0, None)
        share = w[None, :] / (w[:, None] + w[None, :])          # i takes the part of the overlap that j outweighs
        p += 0.5 * ((ov * share)[..., None] * d / dist[..., None]).sum(1)
        if it < iters * 0.7:
            p += pull * (home - p)
        for hx0, hy0, hx1, hy1 in holes:                         # out of the inset, by the shortest way
            hx0, hy0, hx1, hy1 = hx0 - rad, hy0 - rad, hx1 + rad, hy1 + rad
            inside = (p[:, 0] > hx0) & (p[:, 0] < hx1) & (p[:, 1] > hy0) & (p[:, 1] < hy1)
            for i in np.where(inside)[0]:
                dx = (p[i, 0] - hx0, hx1 - p[i, 0], p[i, 1] - hy0, hy1 - p[i, 1]); k = int(np.argmin(dx))
                p[i, k // 2] = (hx0, hx1, hy0, hy1)[k]
        p[:, 0] = np.clip(p[:, 0], bounds[0] + rad, bounds[2] - rad)
        p[:, 1] = np.clip(p[:, 1], bounds[1] + rad, bounds[3] - rad)
        if ov.max() < 1e-3 and it > 5:
            break
    return p


class Tanks:
    def __init__(self, a, units_path, label, log=print):
        self.a, self.label, self.log = a, label, log
        assert a.attrs, "the tanks panel needs --attrs (the HeiGIT GeoPackage)"
        self.BRK = None if a.breaks == "quantile" else [float(v) / 100 for v in a.breaks.split(",")]
        assert self.BRK is None or len(self.BRK) == len(RAMP) - 1, f"--breaks needs {len(RAMP) - 1} values"
        self.FEW_HATCH = "......" if label == "Countries" else "...."
        self.TR = pyproj.Transformer.from_crs(4326, CRS, always_xy=True)
        self._load_plants()
        self._fill(units_path)
        self._glyph_data()

    # -------------------------------------------------------------- data
    def _load_plants(self):
        a = self.a
        hp_path = a.heigit_plants or a.plants.replace("plants_with_units_", "plants_heigit_")
        if not os.path.exists(hp_path):
            raise SystemExit(f"{hp_path} not found: pass --heigit-plants <01_reconcile out_dir>/plants_heigit_<tag>.csv")
        hp = pd.read_csv(hp_path, keep_default_na=False, na_values=[""])
        pw = pd.read_csv(a.plants, keep_default_na=False, na_values=[""])
        hw = pw[pw.cls != "missing"].reset_index(drop=True)      # 02_aggregate writes the HeiGIT plants first, in order
        assert len(hw) == len(hp) and np.allclose(hw.lon, hp.lon) and np.allclose(hw.lat, hp.lat), \
            "plants_with_units and plants_heigit are not in the same order: rerun 02_aggregate.py"
        hp["country"] = hw.unit.values
        want = ["round_area", "rect_area", "category_number", "category_name", "ISO_2"]
        if a.attrs.lower().endswith(".csv"):
            A = pd.read_csv(a.attrs, keep_default_na=False, na_values=[""]).set_index("fid")
        else:
            layer = a.layer or pyogrio.list_layers(a.attrs)[0][0]
            have = set(pyogrio.read_info(a.attrs, layer=layer)["fields"])
            A = pyogrio.read_dataframe(a.attrs, layer=layer, columns=[c for c in want if c in have],
                                       read_geometry=False, fid_as_index=True)
        miss = ~hp.fid.isin(A.index)
        assert not miss.any(), f"{int(miss.sum())} HeiGIT plants have no row with their fid in {a.attrs}"
        A = A.reindex(hp.fid.values)
        if "ISO_2" in A:                                          # the fid must name the same plant
            agree = float(np.mean(A["ISO_2"].replace({"EL": "GR"}).values == hp.ISO_2.replace({"EL": "GR"}).values))
            self.log(f"fid join: ISO_2 agrees for {agree:.2%} of plants")
            assert agree > 0.99, "fid in plants_heigit does not point at the same plants in the HeiGIT file"
        ra = pd.to_numeric(A["round_area"], errors="coerce").clip(lower=0).fillna(0).values
        re_ = pd.to_numeric(A["rect_area"], errors="coerce").clip(lower=0).fillna(0).values
        hp["round_area"], hp["rect_area"], hp["tank_area"] = ra, re_, ra + re_
        hp["has_tank"] = hp.tank_area > 0
        hp["cat"] = pd.to_numeric(A["category_number"], errors="coerce").values
        names = {}
        if "category_name" in A:
            nm = pd.DataFrame({"c": hp.cat, "n": A["category_name"].values}).dropna()
            names = nm.groupby("c").n.agg(lambda v: v.value_counts().index[0]).to_dict()
        if a.labels:
            names.update({float(k): v for k, v in json.load(open(a.labels)).items()})
        self.hp, self.pw, self.names = hp, pw, names

    def _fill(self, units_path):
        a, hp = self.a, self.hp
        u = gpd.read_file(units_path)
        key = "iso2" if "iso2" in u.columns else "PFAF_ID"
        if key == "iso2":
            hp["unit"] = hp.country
        else:
            lev = len(str(int(u.PFAF_ID.iloc[0])))
            assert a.hybas_dir, "basin units need --hybas-dir (as 02b_aggregate_basins.py)"
            att9 = pd.read_csv(f"{a.hybas_dir}/hybas_lev09_v1c_attributes.csv")
            pfaf9 = dict(zip(att9.HYBAS_ID.astype("int64"), att9.PFAF_ID.astype("int64")))
            hb = pd.to_numeric(hp.HYBAS_ID, errors="coerce")
            route = hb.map(lambda v: pfaf9.get(int(v)) if pd.notna(v) else None)
            route = route.map(lambda v: int(str(int(v))[:lev]) if pd.notna(v) else np.nan)
            need = route.isna()
            if need.any():                                        # point-in-basin, nearest within 25 km (as 02b)
                polys = gpd.read_file(f"{a.hybas_dir}/hybas_lev{lev:02d}_v1c.gpkg")[["PFAF_ID", "geometry"]]
                polys["PFAF_ID"] = polys.PFAF_ID.astype("int64"); polys["geometry"] = polys.geometry.make_valid()
                polys = polys.dissolve(by="PFAF_ID", as_index=False)
                g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(hp.lon[need], hp.lat[need]), crs=4326,
                                     index=hp.index[need])
                j = gpd.sjoin(g, polys, how="left", predicate="within"); j = j[~j.index.duplicated()]
                m2 = j.PFAF_ID.isna()
                if m2.any():
                    nn = gpd.sjoin_nearest(g[m2].to_crs(8857), polys.to_crs(8857), how="left", max_distance=25000)
                    j.loc[m2, "PFAF_ID"] = nn[~nn.index.duplicated()].PFAF_ID.values
                route[need] = j.PFAF_ID.values
            hp["unit"] = route
            self.log(f"basin route: Pfafstetter {int((~need).sum()):,}, point-in-basin {int(need.sum()):,}, "
                     f"unassigned {int(hp.unit.isna().sum())}")
        t = hp[hp.has_tank]
        agg = pd.DataFrame({"n_plants": hp.groupby("unit").size(), "n_tank": t.groupby("unit").size(),
                            "round_area": t.groupby("unit").round_area.sum(),
                            "tank_area": t.groupby("unit").tank_area.sum(),
                            "round_share_plantmean": (t.round_area / t.tank_area).groupby(t.unit).mean()})
        agg["round_share_area"] = agg.round_area / agg.tank_area
        agg["round_share"] = agg.round_share_area if a.weighting == "area" else agg.round_share_plantmean
        u = u.drop(columns=[c for c in agg.columns if c in u.columns]).merge(agg, left_on=key, right_index=True, how="left")
        u[["n_plants", "n_tank"]] = u[["n_plants", "n_tank"]].fillna(0).astype(int)
        ok = u.n_tank >= a.min_fill
        u["_ok"], u["_few"] = ok, (u.n_plants > 0) & ~ok
        u["_unc"] = (u.status == "uncovered") & (u.n_plants == 0)
        u["fill"] = None
        if self.BRK is None:
            q = np.linspace(0, 1, len(RAMP) + 1)[1:-1]
            self.BRK = [round(float(v), 2) for v in u.loc[ok, "round_share"].quantile(q)]
            assert all(b0 < b1 for b0, b1 in zip(self.BRK, self.BRK[1:])), \
                f"quantile breaks collapse after rounding to whole %: {self.BRK}; pass --breaks explicitly"
            self.log(f"round share quantile breaks (%): {[round(b * 100) for b in self.BRK]}")
        u.loc[ok, "fill"] =[RAMP[i] for i in np.digitize(u.loc[ok, "round_share"], self.BRK)]
        self.units_table = u.drop(columns="geometry")
        self.log(f"plants with tanks {len(t):,} of {len(hp):,}; global round share "
                 f"{t.round_area.sum() / t.tank_area.sum():.1%}; units coloured {int(ok.sum())}, few "
                 f"{int(u._few.sum())}, uncovered {int(u._unc.sum())}")
        self.log("round share of coloured units, quantiles: " +
                 str(u.loc[ok, "round_share"].quantile([.1, .25, .5, .75, .9]).round(3).to_dict()))
        clip = box(-180, LAT_MIN, 180, 90)
        u["geometry"] = u.geometry.make_valid()
        u = u.clip(clip).to_crs(CRS)
        for c in ("_ok", "_few", "_unc"):
            u[c] = u[c].astype(bool)
        self.u, self.key = u, key
        ne = self.a.ne_data or "data"
        self.land = gpd.read_file(f"{ne}/ne_50m_land.geojson").clip(clip).to_crs(CRS)
        self.rivers = None
        if self.a.tank_rivers:
            r = gpd.read_file(f"{ne}/ne_50m_rivers_lake_centerlines.geojson")
            self.rivers = r[r.featurecla.str.contains("River")].clip(clip).to_crs(CRS)
        self.sphere = _sphere()
        X0, Y0, X1, Y1 = self.sphere.total_bounds
        PAD = 0.6e5
        self.extent = (X0 - PAD, Y0 - PAD, X1 + PAD, Y1 + PAD)

    def _glyph_data(self):
        a, hp, pw, TR = self.a, self.hp, self.pw, self.TR
        cu = gpd.read_file(a.countries or a.plants.replace("plants_with_units_", "units_country_").replace(".csv", ".gpkg"))
        cu = cu.to_crs(CRS).set_index("iso2")
        covered = set(cu.index[cu.status == "ok"]) | set(hp.country.dropna())
        g = hp[hp.cat.notna() & hp.country.isin(covered)]
        comp = pd.crosstab(g.country, g.cat)
        self.RING = a.ring_class if a.ring_class is not None else float(g.cat.value_counts().index[0])
        self.OTHERS = [c for c in sorted(comp.columns) if c != self.RING]
        assert len(self.OTHERS) <= len(CAT_COL), f"{len(self.OTHERS)} classes besides the ring class; only {len(CAT_COL)} colours"
        self.COLS = {c: CAT_COL[i] for i, c in enumerate(self.OTHERS)}
        G = pd.DataFrame({"n_cat": comp.sum(axis=1)})
        G["p_ring"] = comp.get(self.RING, 0) / G.n_cat
        for c in self.OTHERS:
            G[f"p_{int(c)}"] = comp[c] / G.n_cat
        G = G[G.n_cat >= a.min_glyph].copy()
        pl = pw[pw.unit.isin(G.index)].copy()                    # centre of HeiGIT + missing plants, as map 1
        pl["x"], pl["y"] = TR.transform(pl.lon.values, pl.lat.values)
        G["x"], G["y"] = pl.groupby("unit").x.mean(), pl.groupby("unit").y.mean()
        for iso, r in G.iterrows():
            if iso in cu.index and not cu.geometry[iso].contains(Point(r.x, r.y)):
                q = pl[pl.unit == iso]; j = np.argmin((q.x - r.x) ** 2 + (q.y - r.y) ** 2)
                G.loc[iso, ["x", "y"]] = q.x.iat[j], q.y.iat[j]
        glob = g.cat.value_counts(normalize=True)
        self.log(f"ring class {self.lab(self.RING)}: {glob.get(self.RING, 0):.1%} of plants worldwide; glyphs for "
                 f"{len(G)} countries (>= {a.min_glyph} plants with a category)")
        self.log("worldwide shares: " + str({self.lab(c): round(float(v), 4) for c, v in glob.sort_index().items()}))
        self.EU = None
        G["in_eu"] = False
        if a.inset == "europe":
            x0, y0 = TR.transform(-11, 34.5); x1, _ = TR.transform(33, 34.5); _, y1 = TR.transform(10, 71.5)
            self.EU = (x0, y0, x1, y1)
            G["in_eu"] = (G.x >= x0) & (G.x <= x1) & (G.y >= y0) & (G.y <= y1)
        self.G = G

    def lab(self, c):
        return self.names.get(c, f"Class {int(c)}")

    def write_tables(self, out):
        self.units_table.to_csv(out + "_units.csv", index=False)
        self.G.to_csv(out + "_countries.csv", index_label="iso2")

    # -------------------------------------------------------------- drawing
    def _wedge(self, ax, x, y, r, f0, f1, col, width=None, z=8):
        if f1 - f0 > 1e-9:
            ax.add_patch(Wedge((x, y), r, 90 - 360 * f1, 90 - 360 * f0, width=width, fc=col, ec="white", lw=0.12,
                               zorder=z))

    def glyph(self, ax, x, y, r, p_ring, p_other):
        """p_other: shares of the other classes among ALL plants (sum = 1 - p_ring)."""
        rest = 1 - p_ring
        if self.a.glyph == "ringpie":
            rw = RING_W * r
            ax.add_patch(Circle((x, y), r, fc=TRACK_C, ec="none", zorder=8))
            self._wedge(ax, x, y, r, 0, p_ring, RING_C, width=rw)
            ri = r - rw - 0.05 * r
        else:
            ax.add_patch(Circle((x, y), r, fc=CORE_DOM_C, ec="none", zorder=8))
            ri = r * np.sqrt(rest)
            if rest > 0:
                ax.add_patch(Circle((x, y), ri, fc="white", ec="none", zorder=8))
            ri *= 0.97
        if rest > 0:
            cum = 0.0
            for c, v in p_other.items():
                self._wedge(ax, x, y, ri, cum, cum + v / rest, self.COLS[c]); cum += v / rest
        ax.add_patch(Circle((x, y), r, fc="none", ec=GLYPH_EDGE, lw=0.25, zorder=9))

    def _glyphs(self, ax, sel, mm_per_m, holes=()):
        if not len(sel):
            return 0
        xy = np.c_[sel.x, sel.y] * mm_per_m
        bounds = np.r_[ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]] * mm_per_m
        p = relax(xy, np.sqrt(sel.n_cat.values.astype(float)), R_MM, 0.18, bounds,
                  [np.asarray(h) * mm_per_m for h in holes])
        r = R_MM / mm_per_m
        moved = 0
        for (iso, s), (px, py), (hx, hy) in zip(sel.iterrows(), p / mm_per_m, xy / mm_per_m):
            if np.hypot(px - hx, py - hy) > 0.45 * r:           # displaced: thin line back to the country
                moved += 1
                f = 1 - r / np.hypot(px - hx, py - hy)
                ax.plot([hx, hx + f * (px - hx)], [hy, hy + f * (py - hy)], color=GLYPH_EDGE, lw=0.25, zorder=7,
                        solid_capstyle="round")
                ax.add_patch(Circle((hx, hy), 0.22 * r, fc=GLYPH_EDGE, ec="white", lw=0.1, zorder=7))
            self.glyph(ax, px, py, r, s.p_ring, {c: s[f"p_{int(c)}"] for c in self.OTHERS})
        return moved

    def _map(self, ax, extent):
        u = self.u
        self.sphere.plot(ax=ax, facecolor="white", edgecolor="none", zorder=0)
        self.land.plot(ax=ax, facecolor=NODATA, edgecolor="none", zorder=1)
        if u._unc.any():
            u[u._unc].plot(ax=ax, facecolor=NODATA, edgecolor=HATCH_EDGE, hatch="//////", linewidth=0, zorder=2)
        if u._few.any():
            u[u._few].plot(ax=ax, facecolor="white", edgecolor=HATCH_EDGE, hatch=self.FEW_HATCH, linewidth=0, zorder=2)
        u[u._ok].plot(ax=ax, color=u.loc[u._ok, "fill"], edgecolor="none", zorder=3)
        u.boundary.plot(ax=ax, color="white", linewidth=0.12 if self.key == "iso2" else 0.07, zorder=4)
        if self.rivers is not None:
            self.rivers.plot(ax=ax, color="#7d9fb8", linewidth=0.15, alpha=0.8, zorder=5)
        self.land.boundary.plot(ax=ax, color="#b0b0b0", linewidth=0.15, zorder=5)
        if extent is self.extent:
            self.sphere.boundary.plot(ax=ax, color="#9a9a9a", linewidth=0.35, zorder=6)
        ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for s in ax.spines.values():
            s.set_visible(extent is not self.extent); s.set_linewidth(0.35); s.set_color("#9a9a9a")

    def draw(self, ax):
        """Whole panel into ax (its position must already be final): map, Europe inset, glyphs."""
        fig = ax.figure; bb = ax.get_position()
        mapw, maph = bb.width * fig.get_figwidth() * 25.4, bb.height * fig.get_figheight() * 25.4
        self._map(ax, self.extent)
        X0, Y0, X1, Y1 = self.extent
        mmpm = mapw / (X1 - X0)
        moved, holes = 0, []
        if self.EU is not None:
            E = self.EU; fx, fy = E[2] - E[0], E[3] - E[1]
            h_in = INSET_W * fy / fx * mapw / maph
            ins = ax.inset_axes([0.004, 0.012, INSET_W, h_in], zorder=20)   # above the sphere outline
            self._map(ins, E)
            moved += self._glyphs(ins, self.G[self.G.in_eu], INSET_W * mapw / fx)
            ax.add_patch(Rectangle((E[0], E[1]), fx, fy, fill=False, lw=0.35, ec="#555555", zorder=10))
            ins.text(0.03, 0.97, "Europe", transform=ins.transAxes, fontsize=FS2, color=INK2, va="top", zorder=11,
                     bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8))
            holes = [(X0, Y0, X0 + (0.004 + INSET_W) * (X1 - X0), Y0 + (0.012 + h_in) * (Y1 - Y0))]
        moved += self._glyphs(ax, self.G[~self.G.in_eu], mmpm, holes)
        self.log(f"glyphs moved off their country centre: {moved}")

    def aspect(self):
        X0, Y0, X1, Y1 = self.extent
        return (Y1 - Y0) / (X1 - X0)

    def legend(self, L, x_left, head):
        """Legend into the mm-coordinate axes L: three blocks from x_left, headings at y = head (87.5 mm wide)."""
        a = self.a

        def swatch(x, y, w, h, fc, hatch=None, ec=None):
            L.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or "none", hatch=hatch, lw=0))

        x0, sw = x_left + 1.0, 4.2
        L.text(x0, head, "Round tanks\n" + ("(% of tank area, plant mean)" if a.weighting == "plant"
                                           else "(% of detected tank area)"),
               fontsize=FS, color=INK, va="top", linespacing=1.0)
        y0 = head - 8.6
        for k, col in enumerate(RAMP):
            swatch(x0 + k * sw, y0, sw - 0.25, 2.6, col)
        for k, b in enumerate(self.BRK):
            L.text(x0 + (k + 1) * sw - 0.12, y0 - 0.5, f"{b * 100:g}", ha="center", va="top", fontsize=FS2, color=INK2)
        L.text(x0, y0 + 3.1, "fewer", fontsize=FS2, color=INK2, ha="left", va="bottom")
        L.text(x0 + 6 * sw - 0.25, y0 + 3.1, "more round", fontsize=FS2, color=INK2, ha="right", va="bottom")
        items = [(NODATA, "//////", HATCH_EDGE, "Not covered by HeiGIT"),
                 ("white", self.FEW_HATCH, HATCH_EDGE, f"< {a.min_fill} plants with tanks")]
        for k, (fc, hat, ec, txt) in enumerate(items):
            yy = y0 - 5.9 - k * 3.1
            swatch(x0, yy, 2.4, 2.4, fc, hat, ec)
            L.text(x0 + 3.2, yy + 1.2, txt, va="center", fontsize=FS2, color=INK)
        # glyph key: one worked example, enlarged
        gx0 = x_left + 32.5
        L.text(gx0, head, f"Plant surroundings per country\n(≥ {a.min_glyph} plants)", fontsize=FS, color=INK,
               va="top", linespacing=1.0)
        gx, gy, gr = gx0 + 2.8, head - 11.4, 2.6
        ex_ring = 0.72
        ex = dict(zip(self.OTHERS[:4], (0.12, 0.08, 0.05, 0.03)))
        sx = sum(ex.values()); ex = {c: v / sx * (1 - ex_ring) for c, v in ex.items()}
        self.glyph(L, gx, gy, gr, ex_ring, {c: ex.get(c, 0.0) for c in self.OTHERS})
        line = dict(color=INK2, lw=0.3)
        ring_name = textwrap.shorten(self.lab(self.RING), 22, placeholder="…")
        c45 = np.cos(np.radians(45))
        if a.glyph == "ringpie":
            rr, ri = gr * (1 - RING_W / 2), gr * (1 - RING_W) * 0.5
            top = (f"{ring_name}:", "share of all plants"); bot = ("other classes:", "share of the rest")
        else:
            rr, ri = gr * 0.93, gr * np.sqrt(1 - ex_ring) * 0.5
            top = (f"{ring_name}:", "area ~ plants"); bot = ("other classes:", "area ~ plants")
        tx = gx + gr + 2.0
        L.plot([gx + rr * c45, gx + rr * c45 + 0.8, tx - 0.4], [gy + rr * c45, gy + 2.6, gy + 2.6], **line)
        L.plot([gx + ri * c45, gx + ri * c45 + 1.2, tx - 0.4], [gy - ri * c45, gy - 2.6, gy - 2.6], **line)
        for yy, (l1, l2) in ((gy + 2.6, top), (gy - 2.6, bot)):
            L.text(tx, yy + 0.05, l1, fontsize=FS2, color=INK, va="bottom")
            L.text(tx, yy - 0.05, l2, fontsize=FS2, color=INK2, va="top")
        # category colours
        cx = x_left + 65.0
        L.text(cx, head, "Other classes", fontsize=FS, color=INK, va="top")
        for k, c in enumerate(self.OTHERS):
            yy = head - 3.9 - k * 2.45
            swatch(cx, yy - 0.9, 1.8, 1.8, self.COLS[c])
            L.text(cx + 2.5, yy, textwrap.shorten(self.lab(c), 22, placeholder="…"), fontsize=FS2, color=INK,
                   va="center")


LEGEND_H = 21.0        # mm the tanks legend needs (headings to the last class)
