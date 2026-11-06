#!/usr/bin/env python3
"""
00b_national_reference.py: national WWTP reference plants for Australia and Japan.

Runs before 01_reconcile. It
  1. downloads
     - Geoscience Australia, Wastewater Treatment Facilities, 2026 revision
       (CC BY 4.0, DOI 10.26186/151185), through the ArcGIS REST service,
       with the data zip as fallback;
     - MLIT National Land Numerical Information, sewerage facilities P22,
       FY2012 edition (non-commercial licence), one zip per prefecture;
       the treatment plants are layer P22b (pump stations, P22a, are ignored);
  2. harmonises both into one WGS84 point table;
  3. pairs them one-to-one with the HydroWASTE plants of the same country,
     candidates closer than the radius, shortest pairs accepted first
     (the rule of 01_reconcile);
  4. writes HydroWASTE with its Australian and Japanese plants replaced by the
     national ones, in HydroWASTE's own column layout, so 01_reconcile can read
     it in place of HydroWASTE (as the UWWTD replaces HydroWASTE in Europe).

Outputs in --outdir:
  raw/ga_wwtf_2026.geojson, raw/ksj_p22/P22-12_XX_GML.zip   downloaded files
  national_reference.gpkg
    ga_wwtf_2026          all GA records, original attributes
    ksj_p22b_2012         all KSJ treatment plants, attributes renamed to English
    national_plants       harmonised table with its HydroWASTE partner (if any)
    hydrowaste_unpaired   HydroWASTE plants in AUS/JPN without a national partner
    hydrowaste_merged     HydroWASTE, AUS/JPN rows replaced by the national plants
  national_reference_summary.csv   pairing counts per country and radius
  national_reference_report.json   sources, licences, download date, counts, checks
  <attention>_with_national.geojson  (with --attention) every feature annotated
                                     with the nearest national plant within the radius

Usage
  python 00b_national_reference.py --hydrowaste HydroWASTE_v10.csv --outdir ../national_ref
  python 00b_national_reference.py --hydrowaste HydroWASTE_v10.shp --outdir ../national_ref \
      --attention <figure1.paths.match_dir>/data/reference_plants_without_heigit_coverage.geojson
  --offline       use the files already in <outdir>/raw, download nothing
  --ksj-dir DIR   use prefecture zips downloaded by hand (P22-12_01_GML.zip ...)

Requires geopandas >= 0.14, pyogrio, pyproj, scipy, requests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import requests
from pyproj import Geod
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- sources

GA_REST = "https://services.ga.gov.au/gis/rest/services/Wastewater_Treatment_Facilities/MapServer/0/query"
GA_ZIP = "https://d28rz98at9flks.cloudfront.net/151185/151185_01_2.zip"
GA_CITATION = (
    "Talento, M. (2026): Wastewater Treatment Facilities. Commonwealth of "
    "Australia (Geoscience Australia). https://doi.org/10.26186/151185"
)
GA_LICENCE = "CC BY 4.0"

# The landing page lists the zips by name; this is the KSJ download-path pattern.
KSJ_URL = "https://nlftp.mlit.go.jp/ksj/gml/data/P22/P22-12/P22-12_{pref:02d}_GML.zip"
KSJ_CITATION = (
    "MLIT, National Land Numerical Information: Sewerage facilities (P22), "
    "FY2012 edition, from the FY2010 Sewerage Statistics (JSWA). "
    "https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-P22.html"
)
KSJ_LICENCE = "non-commercial (KSJ terms, https://nlftp.mlit.go.jp/ksj/other/agreement_02.html)"

# P22b attribute codes (KSJ product page), renamed to English.
KSJ_P22B_COLS = {
    "P22b_001": "admin_code",  # prefecture + municipality code
    "P22b_002": "name",
    "P22b_003": "address",
    "P22b_004": "operator",  # municipality or body running the plant
    "P22b_005": "business_type",  # public / regional / special-environment sewerage
    "P22b_006": "business_name",
    "P22b_007": "plan_area_total_ha",
    "P22b_008": "plan_area_sewage_sep_ha",
    "P22b_009": "plan_area_storm_sep_ha",
    "P22b_010": "plan_area_combined_ha",
    "P22b_011": "plan_pop_treated",
    "P22b_012": "plan_pop_in_area",
    "P22b_013": "rain_intensity_min_mmh",
    "P22b_014": "rain_intensity_max_mmh",
    "P22b_015": "return_period_min_yr",
    "P22b_016": "return_period_max_yr",
    "P22b_017": "sewage_avg_min_lpcd",
    "P22b_018": "sewage_avg_max_lpcd",
    "P22b_019": "sewage_peak_min_lpcd",
    "P22b_020": "sewage_peak_max_lpcd",
    "P22b_021": "has_water_treatment",
    "P22b_022": "has_sludge_treatment",
    "P22b_023": "has_reuse_facility",
    "P22b_024": "reclaimed_water_use",
    "P22b_025": "receiving_river",
    "P22b_026": "treat_area_ha_now",
    "P22b_027": "treat_area_ha_plan",
    "P22b_028": "pop_treated_now",
    "P22b_029": "pop_treated_plan",
    "P22b_030": "pop_in_plan_area",
    "P22b_031": "dry_max_m3d_now",
    "P22b_032": "dry_max_m3d_plan",
    "P22b_033": "wet_max_m3d_now",
    "P22b_034": "wet_max_m3d_plan",
}
KSJ_SENTINELS = (-999, -998, -997)  # "-", unknown, lost in the 2011 earthquake

# New integer WASTE_IDs for the national plants (HydroWASTE v1.0 ends near 58,502).
ID_OFFSET = {"AUS": 1_000_000, "JPN": 2_000_000}
COUNTRY_NAME = {"AUS": "Australia", "JPN": "Japan"}
SENSITIVITY_KM = (1, 2, 5, 10)

GEOD = Geod(ellps="WGS84")
R_EARTH = 6_371_008.8


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- download


def download(url: str, dest: Path, session: requests.Session, tries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                tmp.replace(dest)
            return dest
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            log(f"  retry {k + 1} for {url}: {e}")
            time.sleep(2 * (k + 1))
    return dest


def fetch_ga_rest(url: str, dest: Path, session: requests.Session, page: int = 1000) -> int:
    """Page through the ArcGIS layer and save one GeoJSON FeatureCollection (WGS84)."""
    r = session.get(url, params={"where": "1=1", "returnCountOnly": "true", "f": "json"}, timeout=60)
    r.raise_for_status()
    total = int(r.json()["count"])
    feats, offset = [], 0
    while True:
        r = session.get(
            url,
            timeout=120,
            params={
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "orderByFields": "objectid",
                "resultOffset": offset,
                "resultRecordCount": page,
                "f": "geojson",
            },
        )
        r.raise_for_status()
        js = r.json()
        if "error" in js:
            raise RuntimeError(f"GA REST error: {js['error']}")
        fs = js.get("features", [])
        feats.extend(fs)
        exceeded = js.get("exceededTransferLimit") or (js.get("properties") or {}).get("exceededTransferLimit")
        offset += len(fs)
        if not fs or (len(fs) < page and not exceeded):
            break
    ids = [f["properties"].get("objectid") for f in feats]
    if len(set(ids)) != len(ids):
        raise RuntimeError("GA REST paging returned duplicate objectids")
    if len(feats) != total:
        raise RuntimeError(f"GA REST returned {len(feats)} features, the layer reports {total}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    return total


# --------------------------------------------------------------------------- readers


def read_ga(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".zip":
        return read_ga_zip(path)
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(4326)
    for c in [c for c in g.columns if c.endswith("_date")]:
        if pd.api.types.is_numeric_dtype(g[c]):  # ArcGIS dates are epoch ms
            g[c] = pd.to_datetime(g[c], unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")
    return g.to_crs(4326)


def read_ga_zip(path: Path) -> gpd.GeoDataFrame:
    """Fallback: read the largest point layer from the GA data zip, whatever its format."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    cands = sorted({n.split(".gdb/")[0] + ".gdb" for n in names if ".gdb/" in n})
    cands += [n for n in names if n.lower().endswith((".shp", ".gpkg", ".geojson", ".json"))]
    best = None
    for c in cands:
        src = f"/vsizip/{path.resolve()}/{c}"
        try:
            layers = pyogrio.list_layers(src)
        except Exception:
            continue
        for lyr, geom in layers:
            if geom and "Point" in str(geom):
                n = pyogrio.read_info(src, layer=lyr)["features"]
                if best is None or n > best[2]:
                    best = (src, lyr, n)
    if best is None:
        raise RuntimeError(f"no point layer found in {path}")
    log(f"  GA zip: reading {best[0]} layer {best[1]} ({best[2]} features)")
    g = gpd.read_file(best[0], layer=best[1])
    g.columns = [c.lower() if c != "geometry" else c for c in g.columns]
    return g.to_crs(4326)


def read_ksj_zip(path: Path) -> gpd.GeoDataFrame:
    m = re.search(r"P22-12_(\d{2})", path.name)
    pref = int(m.group(1)) if m else -1
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    shp = [n for n in names if re.search(r"P22b[^/]*\.shp$", n, re.I)]
    if not shp:
        raise RuntimeError(f"{path.name}: no P22b shapefile inside ({len(names)} members)")
    member = shp[0]
    has_cpg = any(n.lower() == member[:-4].lower() + ".cpg" for n in names)
    src = f"/vsizip/{path.resolve()}/{member}"
    # Old KSJ shapefiles are Shift-JIS without a .cpg and often without a .prj.
    g = pyogrio.read_dataframe(src, encoding=None if has_cpg else "cp932")
    if g.crs is None:
        g = g.set_crs(4612)  # JGD2000
    g = g.to_crs(4326)
    g.insert(0, "pref_code", pref)
    g.insert(1, "row_in_file", np.arange(len(g)))
    g.insert(2, "source_file", path.name)
    return g


def read_hydrowaste(path: Path, lon_col: str, lat_col: str) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False, encoding_errors="replace")
    else:
        df = gpd.read_file(path)
        if df.crs is not None:
            df = df.to_crs(4326)
        df = pd.DataFrame(df.drop(columns="geometry"))
    for c in (lon_col, lat_col):
        if c not in df.columns:
            raise SystemExit(f"HydroWASTE has no column {c!r}; set --hw-lon/--hw-lat")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=4326)


# --------------------------------------------------------------------------- harmonise


def _col(g: pd.DataFrame, name: str, default=None) -> pd.Series:
    """Column by name, also matching shapefile-truncated (10-character) names."""
    for c in (name, name[:10]):
        if c in g.columns:
            return g[c]
    return pd.Series(default, index=g.index)


def harmonise_ga(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    sid = _col(g, "globalid")
    if sid.isna().all():
        sid = _col(g, "objectid")
    sid = sid.astype(str).str.strip("{}")
    addr = _col(g, "address", "").fillna("").astype(str)
    loc = _col(g, "locality", "").fillna("").astype(str)
    op = _col(g, "operator").fillna(_col(g, "owner"))
    out = pd.DataFrame(
        {
            "nat_id": "GA2026:" + sid,
            "source": "GA Wastewater Treatment Facilities 2026",
            "source_id": sid,
            "name": _col(g, "feature_name"),
            "country_iso3": "AUS",
            "admin": _col(g, "state"),
            "address": (addr + ", " + loc).str.strip(", "),
            "operator": op,
            "status": _col(g, "status"),
            "pop_served": np.nan,
            "capacity_m3d": np.nan,
            "loc_quality": "GA spatial_confidence=" + _col(g, "spatial_confidence").astype(str),
        },
        index=g.index,
    )
    return gpd.GeoDataFrame(out, geometry=g.geometry.values, crs=4326)


def clean_ksj(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = g.rename(columns={k: v for k, v in KSJ_P22B_COLS.items() if k in g.columns})
    for c in g.columns:
        if c != "geometry" and pd.api.types.is_numeric_dtype(g[c]) and c not in ("pref_code", "row_in_file"):
            g[c] = g[c].where(~g[c].isin(KSJ_SENTINELS))
    return g


def harmonise_ksj(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    sid = g["pref_code"].map("{:02d}".format) + "-" + g["row_in_file"].map("{:04d}".format)
    pos = g["PosStat"].astype(str) if "PosStat" in g else pd.Series("", index=g.index)
    out = pd.DataFrame(
        {
            "nat_id": "KSJP22b:" + sid,
            "source": "KSJ P22 sewerage facilities FY2012",
            "source_id": sid,
            "name": g.get("name"),
            "country_iso3": "JPN",
            "admin": g.get("admin_code"),
            "address": g.get("address"),
            "operator": g.get("operator"),
            "status": None,
            "pop_served": g.get("pop_treated_now", np.nan),
            "capacity_m3d": g.get("dry_max_m3d_now", np.nan),
            "loc_quality": "address geocode" + np.where(pos != "", " PosStat=" + pos, ""),
        },
        index=g.index,
    )
    return gpd.GeoDataFrame(out, geometry=g.geometry.values, crs=4326)


# --------------------------------------------------------------------------- pairing


def _xyz(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def pair_one_to_one(lon_a, lat_a, lon_b, lat_b, radius_m):
    """Pairs closer than radius_m (WGS84 geodesic), accepted shortest first, one-to-one.
    Returns (idx_a, idx_b, dist_m) of the accepted pairs."""
    lon_a, lat_a, lon_b, lat_b = map(np.asarray, (lon_a, lat_a, lon_b, lat_b))
    if len(lon_a) == 0 or len(lon_b) == 0:
        return (np.empty(0, int),) * 2 + (np.empty(0),)
    chord = 2 * np.sin(radius_m * 1.01 / (2 * R_EARTH))  # sphere vs ellipsoid margin
    cand = cKDTree(_xyz(lon_b, lat_b)).query_ball_point(_xyz(lon_a, lat_a), r=chord)
    ia = np.repeat(np.arange(len(cand)), [len(c) for c in cand])
    ib = np.fromiter((j for c in cand for j in c), int, count=len(ia))
    if len(ia) == 0:
        return (np.empty(0, int),) * 2 + (np.empty(0),)
    _, _, d = GEOD.inv(lon_a[ia], lat_a[ia], lon_b[ib], lat_b[ib])
    keep = d <= radius_m
    ia, ib, d = ia[keep], ib[keep], d[keep]
    order = np.lexsort((ib, ia, d))  # shortest first, stable ties
    used_a = np.zeros(len(lon_a), bool)
    used_b = np.zeros(len(lon_b), bool)
    acc = []
    for k in order:
        if not used_a[ia[k]] and not used_b[ib[k]]:
            used_a[ia[k]] = used_b[ib[k]] = True
            acc.append(k)
    acc = np.array(acc, int)
    return ia[acc], ib[acc], d[acc]


def nearest_within(lon_a, lat_a, lon_b, lat_b, radius_m):
    """Nearest b for every a (not one-to-one); -1 / nan beyond radius_m."""
    idx = np.full(len(lon_a), -1)
    dist = np.full(len(lon_a), np.nan)
    if len(lon_a) == 0 or len(lon_b) == 0:
        return idx, dist
    _, j = cKDTree(_xyz(lon_b, lat_b)).query(_xyz(lon_a, lat_a), k=1)
    _, _, d = GEOD.inv(np.asarray(lon_a), np.asarray(lat_a), np.asarray(lon_b)[j], np.asarray(lat_b)[j])
    ok = d <= radius_m
    idx[ok], dist[ok] = j[ok], d[ok]
    return idx, dist


# --------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--hydrowaste", required=True, type=Path, help="HydroWASTE v1.0 (.csv, .shp, .gpkg or zipped shapefile)"
    )
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--radius-km", type=float, default=10.0)
    ap.add_argument("--attention", type=Path, help="reference_plants_without_heigit_coverage.geojson to annotate")
    ap.add_argument("--offline", action="store_true", help="download nothing")
    ap.add_argument("--ksj-dir", type=Path, help="folder with P22-12_XX_GML.zip files")
    ap.add_argument("--ga-file", type=Path, help="GA GeoJSON or data zip to use as is")
    ap.add_argument("--ga-rest-url", default=GA_REST)
    ap.add_argument("--ga-zip-url", default=GA_ZIP)
    ap.add_argument("--ksj-url", default=KSJ_URL, help="pattern with {pref:02d}")
    ap.add_argument(
        "--keep-hw-unpaired",
        action="store_true",
        help="also keep AUS/JPN HydroWASTE plants without a national partner "
        "in hydrowaste_merged (default: national data replace them)",
    )
    ap.add_argument("--hw-id", default="WASTE_ID")
    ap.add_argument("--hw-iso", default="CNTRY_ISO")
    ap.add_argument("--hw-lon", default="LON_WWTP")
    ap.add_argument("--hw-lat", default="LAT_WWTP")
    a = ap.parse_args(argv)

    out, raw = a.outdir, a.outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    gpkg = out / "national_reference.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    radius_m = a.radius_km * 1000
    report = {
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "radius_km": a.radius_km,
        "sources": {},
        "counts": {},
        "checks": {},
    }
    s = requests.Session()
    s.headers["User-Agent"] = "HeiGIT-WWTP-reference/1.0 (research)"

    # ---- Australia
    ga_path = a.ga_file or raw / "ga_wwtf_2026.geojson"
    ga_via = "file"
    if a.ga_file is None and not a.offline:
        try:
            log("GA: paging the REST service ...")
            n = fetch_ga_rest(a.ga_rest_url, ga_path, s)
            ga_via = "REST"
            log(f"  {n} features")
        except Exception as e:
            log(f"  REST failed ({e}); downloading the data zip")
            ga_path = download(a.ga_zip_url, raw / "ga_wwtf_2026.zip", s)
            ga_via = "zip"
    if not ga_path.exists():
        raise SystemExit(f"missing {ga_path}")
    ga = read_ga(ga_path)
    for c in ("feature_type", "feature_subtype", "status", "spatial_confidence"):
        v = _col(ga, c)
        if v.notna().any():
            log(f"  GA {c}: " + ", ".join(f"{k}={n}" for k, n in v.value_counts(dropna=False).items()))
    report["sources"]["AUS"] = {
        "citation": GA_CITATION,
        "licence": GA_LICENCE,
        "retrieved_via": ga_via,
        "file": str(ga_path),
    }

    # ---- Japan
    kdir = a.ksj_dir or raw / "ksj_p22"
    kdir.mkdir(parents=True, exist_ok=True)
    if a.ksj_dir is None and not a.offline:
        log("KSJ P22: downloading 47 prefecture zips ...")
        for pref in range(1, 48):
            dest = kdir / f"P22-12_{pref:02d}_GML.zip"
            if not dest.exists():
                download(a.ksj_url.format(pref=pref), dest, s)
                time.sleep(0.3)
    zips = sorted(kdir.glob("P22-12_*_GML.zip"))
    if len(zips) != 47:
        log(f"  warning: {len(zips)} of 47 prefecture zips present")
    ksj = clean_ksj(pd.concat([read_ksj_zip(z) for z in zips], ignore_index=True))
    ksj = gpd.GeoDataFrame(ksj, geometry="geometry", crs=4326)
    log(f"  KSJ P22b: {len(ksj)} treatment plants from {len(zips)} prefectures")
    report["sources"]["JPN"] = {"citation": KSJ_CITATION, "licence": KSJ_LICENCE, "files": len(zips), "dir": str(kdir)}

    # ---- harmonise; drop exact duplicates (same name and position)
    nat = pd.concat([harmonise_ga(ga), harmonise_ksj(ksj)], ignore_index=True)
    nat = gpd.GeoDataFrame(nat, geometry="geometry", crs=4326)
    nat = nat[~nat.geometry.is_empty & nat.geometry.notna()].copy()
    nat["lon"], nat["lat"] = nat.geometry.x, nat.geometry.y
    key = (
        nat["country_iso3"]
        + "|"
        + nat["name"].fillna("").astype(str)
        + "|"
        + nat["lon"].round(6).astype(str)
        + "|"
        + nat["lat"].round(6).astype(str)
    )
    dup = key.duplicated()
    report["counts"]["exact_duplicates_dropped"] = {
        k: int(v) for k, v in nat.loc[dup, "country_iso3"].value_counts().items()
    }
    nat = nat[~dup].reset_index(drop=True)
    assert nat["nat_id"].is_unique, "national IDs are not unique"

    # ---- HydroWASTE
    hw = read_hydrowaste(a.hydrowaste, a.hw_lon, a.hw_lat)
    if a.hw_iso not in hw.columns or a.hw_id not in hw.columns:
        raise SystemExit(f"HydroWASTE lacks {a.hw_iso!r} or {a.hw_id!r}; set --hw-iso/--hw-id")
    hw_sub = hw[hw[a.hw_iso].isin(["AUS", "JPN"])]

    # ---- pairing, per country, at the main radius and the sensitivity radii
    id_int = pd.api.types.is_integer_dtype(hw[a.hw_id])
    nat["hw_id"] = pd.Series(pd.NA, index=nat.index, dtype="Int64" if id_int else "object")
    nat["hw_dist_m"] = np.nan
    hw_paired = set()
    rows = []
    for iso in ("AUS", "JPN"):
        n_i = nat.index[nat["country_iso3"] == iso]
        h_i = hw_sub.index[hw_sub[a.hw_iso] == iso]
        for r_km in sorted(set(SENSITIVITY_KM) | {a.radius_km}):
            ia, ib, d = pair_one_to_one(
                nat.loc[n_i, "lon"],
                nat.loc[n_i, "lat"],
                hw_sub.loc[h_i].geometry.x,
                hw_sub.loc[h_i].geometry.y,
                r_km * 1000,
            )
            rows.append(
                {
                    "country": iso,
                    "radius_km": r_km,
                    "n_national": len(n_i),
                    "n_hydrowaste": len(h_i),
                    "paired": len(ia),
                    "national_only": len(n_i) - len(ia),
                    "hydrowaste_only": len(h_i) - len(ia),
                    "median_dist_m": float(np.median(d)) if len(d) else np.nan,
                    "p90_dist_m": float(np.percentile(d, 90)) if len(d) else np.nan,
                }
            )
            if r_km == a.radius_km:
                nat.loc[n_i[ia], "hw_id"] = hw_sub.loc[h_i[ib], a.hw_id].values
                nat.loc[n_i[ia], "hw_dist_m"] = d
                hw_paired.update(h_i[ib])
                assert len(set(ia)) == len(ia) and len(set(ib)) == len(ib)
                assert (d <= r_km * 1000).all()
    nat["pair_class"] = np.where(nat["hw_dist_m"].notna(), "paired", "national_only")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "national_reference_summary.csv", index=False)
    log("\n" + summary.to_string(index=False))

    hw_unpaired = hw_sub.drop(index=list(hw_paired)).copy()

    # ---- HydroWASTE with AUS/JPN replaced, in HydroWASTE's layout
    new_ids = pd.Series(0, index=nat.index, dtype="int64")
    for iso, off in ID_OFFSET.items():
        m = nat["country_iso3"] == iso
        new_ids[m] = off + np.arange(1, m.sum() + 1)
    nat["new_" + a.hw_id] = new_ids
    nat_hw = pd.DataFrame({a.hw_id: new_ids, a.hw_iso: nat["country_iso3"], a.hw_lat: nat["lat"], a.hw_lon: nat["lon"]})
    if "WWTP_NAME" in hw.columns:
        nat_hw["WWTP_NAME"] = nat["name"]
    if "COUNTRY" in hw.columns:
        nat_hw["COUNTRY"] = nat["country_iso3"].map(COUNTRY_NAME)
    if "POP_SERVED" in hw.columns:
        nat_hw["POP_SERVED"] = nat["pop_served"]
    nat_hw["REF_SOURCE"] = nat["source"]
    nat_hw["NAT_ID"] = nat["nat_id"]
    nat_hw["HW_PAIRED_ID"] = nat["hw_id"]
    nat_hw["HW_DIST_M"] = nat["hw_dist_m"]
    keep = hw[~hw[a.hw_iso].isin(["AUS", "JPN"])]
    parts = [pd.DataFrame(keep.drop(columns="geometry")).assign(REF_SOURCE="HydroWASTE v1.0"), nat_hw]
    if a.keep_hw_unpaired:
        parts.append(
            pd.DataFrame(hw_unpaired.drop(columns="geometry")).assign(
                REF_SOURCE="HydroWASTE v1.0 (no national partner)"
            )
        )
    merged = pd.concat(parts, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry=gpd.points_from_xy(merged[a.hw_lon], merged[a.hw_lat]), crs=4326)
    assert merged[a.hw_id].is_unique, "WASTE_IDs collide after the merge"

    # ---- checks
    exp = len(hw) - len(hw_sub) + len(nat) + (len(hw_unpaired) if a.keep_hw_unpaired else 0)
    report["checks"] = {
        "merged_rows_add_up": len(merged) == exp,
        "pairs_one_to_one": bool(nat["hw_id"].dropna().is_unique),
        "pairs_within_radius": bool((nat["hw_dist_m"].dropna() <= radius_m).all()),
        "hydrowaste_classes_add_up": int(nat["hw_dist_m"].notna().sum()) + len(hw_unpaired) == len(hw_sub),
        "national_coords_in_bbox": {
            "AUS": bool(
                nat.query("country_iso3=='AUS'")["lon"].between(112, 155).all()
                and nat.query("country_iso3=='AUS'")["lat"].between(-45, -9).all()
            ),
            "JPN": bool(
                nat.query("country_iso3=='JPN'")["lon"].between(122, 154).all()
                and nat.query("country_iso3=='JPN'")["lat"].between(20, 46).all()
            ),
        },
    }
    report["counts"].update(
        {
            "national": nat["country_iso3"].value_counts().to_dict(),
            "hydrowaste_in_AUS_JPN": hw_sub[a.hw_iso].value_counts().to_dict(),
            "paired_at_radius": nat.loc[nat["hw_dist_m"].notna(), "country_iso3"].value_counts().to_dict(),
            "merged_rows": len(merged),
            "hydrowaste_rows": len(hw),
        }
    )

    # ---- write
    ga.to_file(gpkg, layer="ga_wwtf_2026", driver="GPKG")
    ksj.to_file(gpkg, layer="ksj_p22b_2012", driver="GPKG")
    nat.to_file(gpkg, layer="national_plants", driver="GPKG")
    hw_unpaired.to_file(gpkg, layer="hydrowaste_unpaired", driver="GPKG")
    merged.to_file(gpkg, layer="hydrowaste_merged", driver="GPKG")

    if a.attention:
        att = gpd.read_file(a.attention).to_crs(4326)
        j, d = nearest_within(att.geometry.x, att.geometry.y, nat["lon"], nat["lat"], radius_m)
        att["nat_id"] = np.where(j >= 0, nat["nat_id"].to_numpy()[np.maximum(j, 0)], None)
        att["nat_source"] = np.where(j >= 0, nat["source"].to_numpy()[np.maximum(j, 0)], None)
        att["nat_dist_m"] = d
        dest = out / (a.attention.stem + "_with_national.geojson")
        att.to_file(dest, driver="GeoJSON")
        report["counts"]["attention_features"] = len(att)
        report["counts"]["attention_with_national_within_radius"] = int((j >= 0).sum())
        log(
            f"\nattention file: {(j >= 0).sum()} of {len(att)} features have a national "
            f"plant within {a.radius_km:g} km -> {dest.name}"
        )

    (out / "national_reference_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    bad = [k for k, v in report["checks"].items() if v is False or (isinstance(v, dict) and not all(v.values()))]
    log(f"\nchecks: {'all passed' if not bad else 'FAILED: ' + ', '.join(bad)}")
    log(f"wrote {gpkg}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
