"""Calibrate the service radius against real catchments and census population.

Two steps, both from published data matched to our own plants:

1. **Shape** (``radius_exponent`` and the ``radius_calibration`` rows).
   EPA National Sewershed Dataset, sourced polygons only, matched to our plants
   by CWNS ID. For each plant, r95 is the distance from the plant within which
   95 % of its catchment area lies. ``log10 r95 = a + b log10 F`` is fitted on
   plants with a usable footprint F (detected tanks + rectangles, no lagoon).
   Row q holds ``coefficient_km = 10 ** (a + quantile_q(residuals))`` and, for
   plants without a usable footprint, ``unsized_km = quantile_q(r95)``.
   Optionally the GB catchments (Hoffmann et al.) are used as an independent
   check: same fit, coverage of the US rows.

2. **Level** (``radius_quantile``). Census 2020 block-group population centroids
   (USpopcenters, ``block_group2020.rda``) inside the sourced sewersheds of
   states where almost every plant has a sourced boundary, so that people
   outside them are unsewered. Each plant's service area is approximated by
   its Thiessen cell among all our plants, cut by the circle; the chosen q is
   the one whose total population matches the sewershed total. A
   leave-one-state-out run reports the out-of-sample error.

Neighbour distance is reported in step 1 as well; it explained 0.1 % of r95
in the 2026-11 run and is therefore not part of the rule.

Usage::

    python -m src.calibration.service_radius \\
        --plants data/all_merged_v2.gpkg \\
        --epa-gdb data/validation_data/usa_epa_sewersheds/Version_1_3.gdb \\
        --gb-dir data/validation_data/gbr_catchments \\
        --blockgroups block_group2020.rda --out notes/service_radius_calibration.json
"""
from __future__ import annotations

import argparse
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_008.8
US_ALBERS = 5070
BNG = 27700
DEFAULT_STATES = {"MA": "25", "CT": "09", "RI": "44", "VT": "50", "NH": "33", "UT": "49", "KY": "21"}
QUANTILES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def load_plants(path):
    """Our plants with lon, lat, footprint_m2, lagoon, CWNS ID and spacing.

    Accepts the merged GeoPackage (footprint computed exactly as the pipeline
    does) or a CSV with ``lon``, ``lat`` and ``footprint_m2`` columns.
    """
    if path.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(path, low_memory=False, dtype={"CWNS_2022_ID": str})
    else:
        try:
            from ..create_voronoi import calculate_area
        except ImportError:
            from src.create_voronoi import calculate_area
        gdf = gpd.read_file(path).to_crs(4326)
        gdf = calculate_area(gdf)
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        pts = gdf.geometry.representative_point()
        df["lon"], df["lat"] = pts.x.values, pts.y.values
    df = df[df["lon"].notna() & df["lat"].notna()].reset_index(drop=True)
    df["footprint_m2"] = pd.to_numeric(df.get("footprint_m2", 0), errors="coerce").fillna(0.0)
    lagoon = df["lagoon_detected"] if "lagoon_detected" in df else pd.Series(False, index=df.index)
    df["lagoon"] = lagoon.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})
    df["sized"] = (df["footprint_m2"] > 0) & ~df["lagoon"]
    cwns = df["CWNS_2022_ID"] if "CWNS_2022_ID" in df else pd.Series(np.nan, index=df.index)
    df["cwns"] = cwns.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11).where(cwns.notna())
    df["nnd2_km"] = two_neighbour_distance_km(df["lon"].to_numpy(), df["lat"].to_numpy())
    return df


def two_neighbour_distance_km(lon, lat):
    """Mean great-circle distance to the two nearest other plants (km)."""
    lon, lat = np.radians(lon), np.radians(lat)
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    d, _ = cKDTree(xyz).query(xyz, k=3)
    return (2 * EARTH_RADIUS_M * np.arcsin(np.clip(d / 2, 0, 1)))[:, 1:3].mean(1) / 1000.0


def reach_km(geoms, xs, ys, n=4000, seed=0):
    """(r_far, r95, r50) in km of each polygon around its plant (metric CRS)."""
    rng = np.random.default_rng(seed)
    out = []
    for geom, x, y in zip(geoms, xs, ys):
        coords = shapely.get_coordinates(geom)
        far = np.hypot(coords[:, 0] - x, coords[:, 1] - y).max()
        minx, miny, maxx, maxy = geom.bounds
        pts = shapely.points(rng.uniform(minx, maxx, n), rng.uniform(miny, maxy, n))
        inside = shapely.contains(geom, pts)
        if inside.sum() < 20:
            out.append((far, np.nan, np.nan))
            continue
        q = shapely.get_coordinates(pts[inside])
        d = np.hypot(q[:, 0] - x, q[:, 1] - y)
        out.append((far, np.percentile(d, 95), np.percentile(d, 50)))
    return np.asarray(out, dtype=float) / 1000.0


def us_matches(plants, gdb):
    """Our plants matched one-to-one to EPA sourced sewersheds (CONUS)."""
    s = gpd.read_file(gdb, layer="Sewersheds")
    s = s[(s["Method"] == "Sourced") & ~s["CWNS_ID"].str.contains(";") & s["Flag"].isna()
          & (s["RESIDENTIAL_POP_2022"] > 0) & ~s["STATE_CODE"].isin(["AK", "HI", "PR", "GU", "VI", "AS", "MP"])]
    j = s.merge(plants, left_on="CWNS_ID", right_on="cwns")
    j = j[j.groupby("CWNS_ID")["CWNS_ID"].transform("size") == 1].reset_index(drop=True)
    poly = gpd.GeoSeries(j.geometry, crs=4326).to_crs(US_ALBERS).make_valid()
    pt = gpd.GeoSeries(gpd.points_from_xy(j["lon"], j["lat"]), crs=4326).to_crs(US_ALBERS)
    near = poly.buffer(500).contains(pt).to_numpy()  # plant at its catchment
    r = reach_km(poly.values, pt.x.values, pt.y.values)
    j = j.assign(r_far_km=r[:, 0], r95_km=r[:, 1], r50_km=r[:, 2], poly=poly.values, pt=pt.values)
    return j[near & np.isfinite(j["r95_km"])].reset_index(drop=True)


def gb_matches(plants, gb_dir):
    """Our plants inside exactly one GB catchment polygon, one plant per catchment."""
    import tempfile
    import zipfile
    zpath = os.path.join(gb_dir, "catchments_consolidated.zip")
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zpath).extractall(tmp)
        shp = next(f for f in os.listdir(tmp) if f.endswith(".shp"))
        c = gpd.read_file(os.path.join(tmp, shp))
    c = c[c.geometry.notna()].copy()
    c["geometry"] = c.geometry.make_valid()
    c = c.dissolve(by="identifier", as_index=False)
    pts = gpd.GeoDataFrame(plants, geometry=gpd.points_from_xy(plants["lon"], plants["lat"]), crs=4326).to_crs(BNG)
    j = gpd.sjoin(pts, c[["identifier", "geometry"]], predicate="within")
    j = j[j["identifier"].map(j.groupby("identifier").size()) == 1]
    j = j[j.index.map(j.groupby(level=0).size()) == 1]
    poly = c.set_index("identifier").loc[j["identifier"]].geometry
    r = reach_km(poly.values, j.geometry.x.values, j.geometry.y.values)
    j = j.assign(r_far_km=r[:, 0], r95_km=r[:, 1], r50_km=r[:, 2])
    return j[np.isfinite(j["r95_km"])].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Step 1: shape                                                               #
# --------------------------------------------------------------------------- #
def ols(y, cols):
    X = np.column_stack([np.ones(len(y))] + list(cols))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, float(1 - r.var() / y.var()), r


def fit_shape(matches):
    """Fit log10 r95 on log10 footprint; report what neighbour distance adds."""
    sized = matches[matches["sized"]]
    x, y = np.log10(sized["footprint_m2"].to_numpy()), np.log10(sized["r95_km"].to_numpy())
    (a, b), r2, resid = ols(y, [x])
    _, r2_nnd, _ = ols(y, [np.log10(sized["nnd2_km"].to_numpy())])
    _, r2_both, _ = ols(y, [x, np.log10(sized["nnd2_km"].to_numpy())])
    unsized = matches.loc[~matches["sized"], "r95_km"].to_numpy()
    rows = {q: {"coefficient_km": round(float(10 ** (a + np.quantile(resid, q))), 3),
                "unsized_km": round(float(np.quantile(unsized, q)), 2)} for q in QUANTILES}
    return {"intercept": float(a), "exponent": float(b), "r2_footprint": r2, "r2_neighbour_distance": r2_nnd,
            "r2_footprint_and_neighbours": r2_both, "residual_sd": float(resid.std()), "n_sized": len(sized),
            "n_unsized": len(unsized), "rows": rows}


def coverage(matches, shape):
    """Share of catchments whose r95 lies inside the circle of each row."""
    F = np.where(matches["sized"], matches["footprint_m2"], 1.0)
    out = {}
    for q, row in shape["rows"].items():
        radius = np.where(matches["sized"], row["coefficient_km"] * F ** shape["exponent"], row["unsized_km"])
        out[q] = float(np.mean(matches["r95_km"].to_numpy() <= radius))
    return out


# --------------------------------------------------------------------------- #
# Step 2: level                                                               #
# --------------------------------------------------------------------------- #
def choose_quantile(plants, us, shape, blockgroups, states=DEFAULT_STATES):
    """Pick q from census population; leave-one-state-out check."""
    import pyreadr

    bg = pyreadr.read_r(blockgroups)["block_group2020"]
    bg = bg[bg["STATEFP"].isin(states.values()) & (bg["POPULATION"] > 0)]
    pop = gpd.GeoSeries(gpd.points_from_xy(bg["LONGITUDE"], bg["LATITUDE"]), crs=4326).to_crs(US_ALBERS)
    popv = bg["POPULATION"].to_numpy(dtype=float)
    tree = shapely.STRtree(pop.values)

    conus = plants[plants["lon"].between(-125, -66) & plants["lat"].between(24, 50)]
    P = gpd.GeoSeries(gpd.points_from_xy(conus["lon"], conus["lat"]), crs=4326).to_crs(US_ALBERS)
    P = P[~P.duplicated()]
    region = shapely.box(*P.total_bounds).buffer(200_000)
    cells = list(shapely.get_parts(shapely.voronoi_polygons(shapely.multipoints(P.values), extend_to=region)))
    cell_tree = shapely.STRtree(cells)

    ev = us[us["STATE_CODE"].isin(states)].reset_index(drop=True)
    truth = [set(tree.query(g, predicate="contains").tolist()) for g in ev["poly"]]
    P_true = np.array([popv[list(t)].sum() for t in truth])
    near = []  # population points in each plant's Thiessen cell, with their distance to the plant
    for pt in ev["pt"]:
        cell = cells[int(cell_tree.query(pt, predicate="within")[0])]
        idx = tree.query(cell, predicate="contains")
        xy = shapely.get_coordinates(pop.values[idx])
        near.append((idx, np.hypot(xy[:, 0] - pt.x, xy[:, 1] - pt.y)))

    F = np.where(ev["sized"], ev["footprint_m2"], 1.0)
    def predicted(q):
        row = shape["rows"][q]
        r = 1000 * np.where(ev["sized"], row["coefficient_km"] * F ** shape["exponent"], row["unsized_km"])
        return np.array([popv[idx[d <= rr]].sum() for (idx, d), rr in zip(near, np.minimum(r, 50_000))])

    preds = {q: predicted(q) for q in shape["rows"]}
    def score(pred, mask):
        ok = mask & (P_true > 0) & (pred > 0)
        lr = np.log10(pred[ok] / P_true[ok])
        return {"total_ratio": float(pred[mask].sum() / P_true[mask].sum()), "median_abs_log10": float(np.median(np.abs(lr))),
                "within_factor_2": float(np.mean(np.abs(lr) <= np.log10(2))), "n": int(mask.sum())}
    every = np.ones(len(ev), bool)
    table = {q: score(p, every) for q, p in preds.items()}
    chosen = min(preds, key=lambda q: abs(table[q]["total_ratio"] - 1))
    folds = []
    for st in sorted(ev["STATE_CODE"].unique()):
        train = ev["STATE_CODE"].to_numpy() != st
        qs = min(preds, key=lambda q: abs(preds[q][train].sum() / P_true[train].sum() - 1))
        folds.append({"held_out": st, "q": qs, **score(preds[qs], ~train)})
    return {"states": sorted(states), "n_plants": len(ev), "people": float(popv.sum()),
            "by_quantile": table, "chosen": chosen, "leave_one_state_out": folds}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plants", required=True, help="Merged plant GeoPackage or CSV with lon/lat/footprint_m2")
    parser.add_argument("--epa-gdb", required=True, help="EPA National Sewershed Dataset Version_1_3.gdb")
    parser.add_argument("--gb-dir", default=None, help="Folder with catchments_consolidated.zip (independent check)")
    parser.add_argument("--blockgroups", default=None, help="USpopcenters block_group2020.rda (step 2)")
    parser.add_argument("--out", default=None, help="Write the full result as JSON")
    args = parser.parse_args()

    plants = load_plants(args.plants)
    us = us_matches(plants, args.epa_gdb)
    shape = fit_shape(us)
    result = {"shape_us": shape, "coverage_us": coverage(us, shape)}
    print(f"US: n={shape['n_sized']} sized + {shape['n_unsized']} unsized; log10 r95 = {shape['intercept']:.3f} + "
          f"{shape['exponent']:.3f} log10 F, R2 {shape['r2_footprint']:.3f}; neighbour distance alone R2 "
          f"{shape['r2_neighbour_distance']:.3f}, with footprint {shape['r2_footprint_and_neighbours']:.3f}")
    if args.gb_dir:
        gb = gb_matches(plants, args.gb_dir)
        result["shape_gb"] = {k: v for k, v in fit_shape(gb).items() if k != "rows"}
        result["coverage_gb_with_us_rows"] = coverage(gb, shape)
        print(f"GB: exponent {result['shape_gb']['exponent']:.3f}, R2 {result['shape_gb']['r2_footprint']:.3f}; "
              f"coverage of US rows {result['coverage_gb_with_us_rows']}")
    if args.blockgroups:
        level = choose_quantile(plants, us, shape, args.blockgroups)
        result["level"] = level
        print(pd.DataFrame(level["by_quantile"]).T.round(3).to_string())
        print(pd.DataFrame(level["leave_one_state_out"]).round(3).to_string(index=False))
        print("chosen radius_quantile:", level["chosen"])
    print("\nconfig.yaml rows:")
    print(f"  radius_exponent: {shape['exponent']:.3f}")
    for q, row in shape["rows"].items():
        print(f"    {q}: {{coefficient_km: {row['coefficient_km']:.3f}, unsized_km: {row['unsized_km']:.2f}}}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1, default=str)


if __name__ == "__main__":
    main()
