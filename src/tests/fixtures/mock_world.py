"""A small, fully synthetic region with every input layer the pipeline reads.

Two countries side by side (FR west of 8.0 E, DE east of it, 49-50 N), ten
level-8 basins on a grid that does not follow the border (one basin straddles
it), towns with WWTPs, WorldPop-style population rasters, urban areas, a
dendritic river network with confluences, a UWWTD-style reference layer and
industrial land.

The layers deliberately contain the situations the audit fixes are about:

* duplicate plants a few hundred metres apart (clustering, B6),
* industrial and mixed-use plants (remove_industrial, R10, B8),
* new plants without WASTE_ID / POP_SERVED / QUAL_POP (R8, R12),
* a basin that crosses the border (R4) and a basin with one plant (B9),
* nodata cells (a lake) in the population rasters (R9),
* tributaries joining main stems (R6),
* industrial land straddling a plant buffer edge (B8).

Everything is generated from one seed, so a run is reproducible.

Usage::

    from src.tests.fixtures.mock_world import build_mock_world
    paths = build_mock_world("/tmp/mock/data", seed=0)
"""
from __future__ import annotations

import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

CRS = "EPSG:4326"

COUNTRIES = {
    # iso2: (iso3, lon_min, lon_max)
    "FR": ("fra", 6.5, 8.0),
    "DE": ("deu", 8.0, 9.5),
}
LAT_MIN, LAT_MAX = 49.0, 50.0
BASIN_LON_EDGES = [6.5, 7.0, 7.6, 8.3, 8.9, 9.5]  # 7.6-8.3 crosses the border
BASIN_LAT_EDGES = [49.0, 49.5, 50.0]
RASTER_RES = 0.005  # degrees (~360 m x 550 m here)
YEARS = (2020, 2024)
NODATA = -99999.0

# Default file names, relative to data_dir, matching config.yaml (version 2; {level} is the basin level)
DEFAULT_PATHS = {
    "annotated_all_filepath": "all_annotated_v2.gpkg",
    "bboxes": "bboxes.csv",
    "hydrowaste": "cleaned_hydrowaste.csv",
    "watershed": "hydroshed_river_levels/hydrobase_lvl{level}_combined.gpkg",
    "overture": "overture_maps_countries.parquet",
    "pop_tif_dir": "population/unzipped",
    "urban_areas_filepath": "urban_areas/GHS_UCDB_GLOBE_R2024A.gpkg",
    "rivershed": "hydrorivers.gpkg",
    "eu_ref_filepath": "extra_points/UWWTD_TreatmentPlants.gpkg",
    "industrial_merged_filepath": "industrial_analysis/industrial_land_merged_v2_lvl{level}_mp100.gpkg",
}

KM_PER_DEG_LAT = 111.195


def _km_per_deg_lon(lat):
    return KM_PER_DEG_LAT * np.cos(np.radians(lat))


def _offset(lon, lat, dx_km, dy_km):
    return lon + dx_km / _km_per_deg_lon(lat), lat + dy_km / KM_PER_DEG_LAT


def _country_of(lon):
    return "FR" if lon < 8.0 else "DE"


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def make_countries():
    rows = [{"country": iso2, "ISO_2": iso2, "subtype": "country",
             "geometry": box(lon0, LAT_MIN, lon1, LAT_MAX)} for iso2, (_, lon0, lon1) in COUNTRIES.items()]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def make_basins():
    rows = []
    hid = 2080000001
    for j in range(len(BASIN_LAT_EDGES) - 1):
        for i in range(len(BASIN_LON_EDGES) - 1):
            rows.append({
                "HYBAS_ID": hid,
                "geometry": box(BASIN_LON_EDGES[i], BASIN_LAT_EDGES[j], BASIN_LON_EDGES[i + 1], BASIN_LAT_EDGES[j + 1]),
            })
            hid += 1
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def make_towns(rng):
    """Town centres with populations; one basin (6.5-7.0, 49.5-50.0) gets a single town."""
    towns = []
    for iso2, (_, lon0, lon1) in COUNTRIES.items():
        n = 20
        lons = rng.uniform(lon0 + 0.05, lon1 - 0.05, n)
        lats = rng.uniform(LAT_MIN + 0.05, LAT_MAX - 0.05, n)
        for lon, lat in zip(lons, lats):
            if 6.5 <= lon < 7.0 and lat >= 49.5:
                continue  # reserved for the single-plant basin
            towns.append((lon, lat, float(np.exp(rng.normal(9.3, 1.0)))))
    towns.append((6.75, 49.75, 25_000.0))  # the only town in its basin
    return towns


def make_sites(rng, towns):
    """WWTP points: one or two per town, duplicates, industrial and new plants."""
    rows = []
    waste_id = 100001
    for t, (lon, lat, pop) in enumerate(towns):
        n_plants = 2 if pop > 30_000 else 1
        for k in range(n_plants):
            plon, plat = _offset(lon, lat, rng.normal(0, 1.5), rng.normal(0, 1.5))
            category = rng.choice([1, 1, 1, 1, 1, 1, 2, 4, 6], p=None)
            is_new = rng.random() < 0.15
            served = float(pop / n_plants * rng.uniform(0.6, 1.0))
            d1, d2 = rng.integers(8, 45, 2)
            rows.append({
                "WASTE_ID": np.nan if is_new else float(waste_id),
                "POP_SERVED": np.nan if is_new else round(served),
                "QUAL_POP": np.nan if is_new else float(rng.choice([1, 1, 1, 2])),
                "category_number": int(category),
                "diameters": f"[{d1}, {d2}]",
                "wwtp_area_rect": f"[{rng.integers(20, 200)} {rng.integers(20, 150)}]",
                "num_detection_rect": int(rng.integers(0, 4)),
                "num_detection_circle": int(rng.integers(0, 3)),
                "ISO_2": _country_of(plon),
                "geometry": Point(plon, plat),
            })
            waste_id += 1
    # Duplicates: the same plant from a second source, 150-350 m away, no ID
    for idx in rng.choice(len(rows), 4, replace=False):
        src = rows[idx]
        dlon, dlat = _offset(src["geometry"].x, src["geometry"].y, rng.uniform(0.1, 0.25), rng.uniform(0.1, 0.25))
        dup = dict(src)
        dup.update({"WASTE_ID": np.nan, "QUAL_POP": np.nan,
                    "POP_SERVED": src["POP_SERVED"] if pd.notna(src["POP_SERVED"]) else 5000.0,
                    "ISO_2": _country_of(dlon), "geometry": Point(dlon, dlat)})
        rows.append(dup)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def make_population(rng, towns, iso2):
    """One raster per year: rural background, Gaussian towns, a nodata lake."""
    _, lon0, lon1 = COUNTRIES[iso2]
    width = int(round((lon1 - lon0) / RASTER_RES))
    height = int(round((LAT_MAX - LAT_MIN) / RASTER_RES))
    lon = lon0 + (np.arange(width) + 0.5) * RASTER_RES
    lat = LAT_MAX - (np.arange(height) + 0.5) * RASTER_RES
    glon, glat = np.meshgrid(lon, lat)
    pop = rng.gamma(2.0, 4.0, size=(height, width))  # ~8 people per cell
    cell_km2 = (RASTER_RES * _km_per_deg_lon(glat)) * (RASTER_RES * KM_PER_DEG_LAT)
    for tlon, tlat, tpop in towns:
        sigma_km = 1.0 + 1.5 * np.log10(tpop / 1000.0)
        dx = (glon - tlon) * _km_per_deg_lon(tlat)
        dy = (glat - tlat) * KM_PER_DEG_LAT
        density = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_km ** 2)) / (2 * np.pi * sigma_km ** 2)
        pop += tpop * density * cell_km2
    lake = ((glon - (lon0 + 0.4)) ** 2 / 0.03 ** 2 + (glat - 49.3) ** 2 / 0.02 ** 2) < 1
    layers = {}
    for year in YEARS:
        grown = pop * (1.0 + 0.008 * (year - YEARS[0]))
        layers[year] = np.where(lake, NODATA, grown).astype("float32")
    return layers, from_origin(lon0, LAT_MAX, RASTER_RES, RASTER_RES)


def make_urban_areas(towns):
    rows = []
    for k, (lon, lat, pop) in enumerate(towns):
        if pop < 15_000:
            continue
        radius_km = 1.0 + np.log10(pop / 1000.0)
        circle = Point(0, 0).buffer(radius_km, quad_segs=16)
        coords = [_offset(lon, lat, x, y) for x, y in circle.exterior.coords]
        rows.append({"ID_UC_G0": k + 1, "GC_UCN_MAI_2025": f"Town {k + 1}", "GC_POP_TOT_2025": pop,
                     "geometry": gpd.GeoSeries.from_xy([c[0] for c in coords], [c[1] for c in coords]).union_all().convex_hull})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def make_rivers(rng):
    """Two east-to-west main stems with tributaries, cut into ~5 km segments.

    HydroRIVERS conventions: HYRIV_ID, NEXT_DOWN (0 at the outlet), MAIN_RIV
    (the outlet segment's ID), DIS_AV_CMS accumulating downstream.
    """
    rows = []
    next_id = 20_000_001
    seg_deg = 0.07

    def add_line(points, q_start, q_end, down_id):
        nonlocal next_id
        ids = []
        n = len(points) - 1
        for k in range(n):
            ids.append(next_id)
            next_id += 1
        for k in range(n):
            q = q_start + (q_end - q_start) * (k + 1) / n
            rows.append({"HYRIV_ID": ids[k], "NEXT_DOWN": ids[k + 1] if k + 1 < n else down_id,
                         "DIS_AV_CMS": round(float(q), 3), "ORD_STRA": 3 if q > 2 else 2,
                         "LENGTH_KM": None, "geometry": LineString([points[k], points[k + 1]])})
        return ids

    for lat in (49.25, 49.75):
        lons = np.arange(9.5, 6.5 - 1e-9, -seg_deg)
        main_pts = [(float(x), lat + 0.01 * np.sin(x * 7)) for x in lons]
        main_ids = add_line(main_pts, 0.4, 6.0, 0)  # small rivers, so plumes reach the confluences
        outlet = main_ids[-1]
        # Tributaries from north and south joining every ~6 segments
        for k in range(3, len(main_ids) - 2, 6):
            jx, jy = main_pts[k + 1]
            side = 1 if (k // 6) % 2 == 0 else -1
            trib_pts = [(jx + 0.02 * (4 - s), jy + side * 0.035 * (4 - s)) for s in range(5)]  # ends at the junction
            add_line(trib_pts, 0.05, float(rng.uniform(0.2, 1.0)), main_ids[k + 1])
        for row in rows:
            row.setdefault("MAIN_RIV", None)
        for row in rows:
            if row["MAIN_RIV"] is None:
                row["MAIN_RIV"] = outlet
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    gdf["LENGTH_KM"] = gdf.to_crs(32632).length / 1000.0
    return gdf


def make_eu_reference(rng, sites):
    """UWWTD-style points for ~70 % of the plants that have a WASTE_ID."""
    known = sites[sites["WASTE_ID"].notna()]
    chosen = known.sample(frac=0.7, random_state=int(rng.integers(1_000_000)))
    rows = []
    for _, row in chosen.iterrows():
        lon, lat = _offset(row.geometry.x, row.geometry.y, rng.normal(0, 0.15), rng.normal(0, 0.15))
        rows.append({"uwwCode": f"EU{int(row['WASTE_ID'])}", "uwwCapacity": float(row["POP_SERVED"] * rng.uniform(0.9, 1.6)),
                     "geometry": Point(lon, lat)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def make_industrial_land(rng, sites):
    """Industrial polygons: near plants, far from plants, and straddling a buffer edge."""
    rows = []
    for k, row in enumerate(sites.itertuples()):
        if k % 3:
            continue
        dist = rng.choice([0.5, 9.0, 30.0])  # inside, near the edge, far away (km)
        ang = rng.uniform(0, 2 * np.pi)
        lon, lat = _offset(row.geometry.x, row.geometry.y, dist * np.cos(ang), dist * np.sin(ang))
        if not (6.55 < lon < 9.45 and 49.05 < lat < 49.95):
            continue
        w, h = rng.uniform(0.3, 1.5, 2)
        x0, y0 = _offset(lon, lat, -w / 2, -h / 2)
        x1, y1 = _offset(lon, lat, w / 2, h / 2)
        rows.append({"industrial_id": len(rows) + 1, "num_cells": int(rng.integers(100, 400)), "geometry": box(x0, y0, x1, y1)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def config_level():
    """The basin level the repository's config.yaml runs at."""
    from src.starter import load_config

    return str(load_config(script_name="combine_watersheds")["level"])


def build_mock_world(data_dir, seed=0, paths=None, level=None):
    """Write every input layer under ``data_dir``; return the absolute paths written.

    ``level`` names the basin files (default: the level in config.yaml); the
    basins themselves are the same mock grid at every level.
    """
    rng = np.random.default_rng(seed)
    level = config_level() if level is None else str(level)
    rel = {key: value.format(level=level) for key, value in dict(DEFAULT_PATHS, **(paths or {})).items()}
    out = {key: os.path.join(os.path.abspath(data_dir), value) for key, value in rel.items()}

    countries = make_countries()
    countries_wkb = pd.DataFrame({"country": countries["country"], "subtype": countries["subtype"],
                                  "geometry": countries.geometry.to_wkb()})
    countries_wkb.to_parquet(_ensure_dir(out["overture"]), index=False)

    make_basins().to_file(_ensure_dir(out["watershed"]), driver="GPKG")

    towns = make_towns(rng)
    sites = make_sites(rng, towns)
    sites.to_file(_ensure_dir(out["annotated_all_filepath"]), driver="GPKG")
    pd.DataFrame(columns=["WASTE_ID", "geometry"]).to_csv(_ensure_dir(out["bboxes"]), index=False)
    hw = sites[sites["WASTE_ID"].notna()].copy()
    hw["LON_WWTP"], hw["LAT_WWTP"] = hw.geometry.x, hw.geometry.y
    pd.DataFrame(hw.drop(columns="geometry")).assign(geometry=hw.geometry.to_wkt()).to_csv(
        _ensure_dir(out["hydrowaste"]), index=False)

    for iso2, (iso3, _, _) in COUNTRIES.items():
        layers, transform = make_population(rng, towns, iso2)
        for year, data in layers.items():
            path = _ensure_dir(os.path.join(out["pop_tif_dir"], iso3, f"{iso3}_pop_{year}_CN_100m_R2024B_v1.tif"))
            with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                               dtype="float32", crs=CRS, transform=transform, nodata=NODATA) as dst:
                dst.write(data, 1)

    make_urban_areas(towns).to_file(_ensure_dir(out["urban_areas_filepath"]), driver="GPKG")
    make_rivers(rng).to_file(_ensure_dir(out["rivershed"]), driver="GPKG")
    make_eu_reference(rng, sites).to_file(_ensure_dir(out["eu_ref_filepath"]), driver="GPKG")
    make_industrial_land(rng, sites).to_file(_ensure_dir(out["industrial_merged_filepath"]), driver="GPKG")
    return out


def main():
    parser = argparse.ArgumentParser(description="Write the synthetic two-country region used by the tests.")
    parser.add_argument("data_dir")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for key, path in build_mock_world(args.data_dir, args.seed).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
