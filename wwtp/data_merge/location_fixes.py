"""Known location errors of plant list v3, corrected from a table (``plants_v3.paths.location_fixes``).

Found on 24 Nov 2026 with Figure 1's UTM check (``figures/fig1/match/00_fix_utm.py``) and checked by hand
against HydroWASTE and the corrected EU file. Each row of the table names one plant by its HydroWASTE id
(``WASTE_ID``) and the wrong position it has, and says what to do:

- ``move``: the plant was converted back from UTM with the wrong zone (its longitude off by whole 6 degree
  zones). It goes to ``lon``/``lat`` (the corrected EU file's location where the plant has an EU code, else
  its own position re-projected with the right zone) and gets ``ISO_2``, ``epsg`` (UTM zone of the new place),
  ``location_source``, ``location_shift_m`` and ``geometry_v2`` (the old place). Its tanks, site category and
  OpenStreetMap match were seen on the image of the wrong place, so they leave the plant
  (``size_source`` no_image). With ``split`` yes, what was seen there (tanks or an OpenStreetMap plant) stays
  as a plant of its own (``origin`` location_fix_split). With ``clear`` hydrowaste, the plant's HydroWASTE
  columns are emptied: its HydroWASTE link is another plant.
- ``keep``: the plant is where it should be, but carries another plant's EU code (``clear`` eu_code): the EU
  file's columns are emptied and the plant of that code (``add_uwwCode``) is added from the corrected EU file
  (``origin`` new_plant_eu_file), unless a plant with that code, or any plant within
  ``same_site_distance_m``, is already there.

A plant is found by its ``WASTE_ID`` within ``location_fix_match_m`` of the wrong position. A row whose plant
is already at its new position (or, for ``keep``, already without the code) counts as applied, so the step can
run again. ``plants_v3`` applies the table before it merges same sites; the command below applies it to an
existing v3 file and merges a moved or added plant with a plant already at its site, as ``plants_v3`` does.

Command: ``python -m wwtp.data_merge.location_fixes [--dry-run]`` - rewrites ``plants_v3.paths.output`` and
writes ``paths.location_fixes_report`` (one row per fix: what was done). It exits with an error and writes
nothing when a fix finds no plant.
"""

from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Geod
from shapely.geometry import Point

from ..service_areas.size import NO_IMAGE, SIZE_SOURCE_COLUMN
from ..starter import load_config
from ..utils import atomic_output, configure_logging
from .work_list import file_countries

logger = logging.getLogger(__name__)
GEOD = Geod(ellps="WGS84")
ACTIONS = ("move", "keep")
# HydroWASTE's columns as the plant list holds them (emptied for a plant whose HydroWASTE link is another plant)
HYDROWASTE_COLUMNS = [
    "WASTE_ID", "SOURCE", "ORG_ID", "WWTP_NAME", "COUNTRY", "CNTRY_ISO", "QUAL_LOC", "LAT_WWTP", "LON_WWTP",
    "LAT_OUT", "LON_OUT", "STATUS", "POP_SERVED", "QUAL_POP", "WASTE_DIS", "QUAL_WASTE", "LEVEL", "QUAL_LEVEL",
    "DF", "HYRIV_ID", "RIVER_DIS", "COAST_10KM", "COAST_50KM", "DESIGN_CAP", "QUAL_CAP", "HW_unique_ID",
    "HW_2014_ORG_ID", "HW_POP_2014",
]  # fmt: skip
# What the image of a place shows, besides plants_v3.detection_columns and category_columns
SITE_COLUMNS = ["lagoon_detected", "File Name", "matched_osm_geometry", "new_image", SIZE_SOURCE_COLUMN]
FIXED = "_location_fixed"  # marks moved and added plants while the step runs


def read_fixes(path):
    fixes = pd.read_csv(path, dtype={"WASTE_ID": str, "uwwCode": str, "add_uwwCode": str}, keep_default_na=False)
    bad = ~fixes["action"].isin(ACTIONS)
    if bad.any():
        raise ValueError(f"{path}: action must be one of {ACTIONS}, got {sorted(fixes.loc[bad, 'action'].unique())}")
    return fixes


def source_id(values):
    """HydroWASTE ids as text without a trailing '.0' ('57058.0' -> '57058'); missing -> ''."""
    text = pd.Series(values, dtype=object).astype(str).str.strip()
    text = text.str.replace(r"\.0+$", "", regex=True)
    return text.where(~text.isin(["", "nan", "None", "<NA>", "NaN"]), "")


def blank(value):
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() in ("", "nan", "<NA>", "None")


def metres(geoms, lon, lat):
    x, y = geoms.x.to_numpy(), geoms.y.to_numpy()
    return GEOD.inv(x, y, np.full(len(x), lon), np.full(len(x), lat))[2]


def utm_epsg(lon, lat):
    zone = int(min(max(np.floor((lon + 180) / 6) + 1, 1), 60))
    return float((32600 if lat >= 0 else 32700) + zone)


def site_columns(plants, cfg):
    cols = [*cfg["detection_columns"], *cfg["category_columns"], *SITE_COLUMNS]
    return [c for c in dict.fromkeys(cols) if c in plants.columns]


def eu_file_plants(path, codes, columns, cfg):
    """The corrected EU file's plants with these codes, in the plant list's columns."""
    eu = gpd.read_file(path).to_crs(4326)
    eu = eu[eu["uwwCode"].isin(codes) & eu.geometry.notna() & ~eu.geometry.is_empty].copy()
    eu["ISO_2"] = file_countries(eu, cfg)
    return eu[[c for c in columns if c in eu.columns and c not in ("lat", "lon")] + ["geometry"]]


def apply_location_fixes(plants, fixes, cfg, eu_file=None, merge=False):
    """The table's fixes on a plant list (EPSG:4326); returns (plants, report).

    ``merge``: merge each moved or added plant with the plants within ``same_site_distance_m`` (for an
    existing v3 file; ``plants_v3`` merges all same sites itself afterwards).
    """
    plants = plants.copy()
    columns = list(plants.columns.drop("geometry"))
    sites = site_columns(plants, cfg)
    hw_columns = [c for c in HYDROWASTE_COLUMNS if c in plants.columns]
    eu_columns = []
    if eu_file is not None:
        eu_columns = [c for c in pyogrio.read_info(eu_file)["fields"] if c in columns and c not in ("lat", "lon")]
    match_m, site_m = float(cfg["location_fix_match_m"]), float(cfg["same_site_distance_m"])
    ids = source_id(plants["WASTE_ID"]) if "WASTE_ID" in plants else pd.Series("", index=plants.index)
    report, new_rows = [], []
    plants[FIXED] = False
    for fix in fixes.itertuples(index=False):
        row = {"WASTE_ID": fix.WASTE_ID, "name": fix.name, "action": fix.action}
        mine = plants.index[(ids == fix.WASTE_ID).to_numpy()]
        if fix.action == "move" and fix.clear == "hydrowaste" and fix.uwwCode and "uwwCode" in plants:
            mine = mine.union(plants.index[(plants["uwwCode"].astype(str) == fix.uwwCode).to_numpy()])
        near = lambda i, lon, lat: metres(plants.geometry.loc[[i]], float(lon), float(lat))[0] <= match_m  # noqa: E731
        at_wrong = [i for i in mine if near(i, fix.wrong_lon, fix.wrong_lat)]
        if fix.action == "move":
            done = [i for i in mine if near(i, fix.lon, fix.lat)]
        else:  # applied: the plant is still there, without the code
            done = [i for i in at_wrong if blank(plants.at[i, "uwwCode"])]
            at_wrong = [i for i in at_wrong if i not in done]
        if len(at_wrong) != 1:
            row["status"] = "already applied" if done and not at_wrong else f"not found ({len(at_wrong)} plants at the wrong position)"
            if not done or at_wrong:
                logger.warning("location fix %s (%s): %s", fix.WASTE_ID, fix.name, row["status"])
            report.append(row)
            continue
        i = at_wrong[0]
        old = plants.geometry.loc[i]
        if fix.action == "move":
            new = Point(float(fix.lon), float(fix.lat))
            if fix.split == "yes":
                split = pd.Series(np.nan, index=plants.columns, dtype=object)
                split[sites] = plants.loc[i, sites]
                split["geometry"] = old
                split.update({"epsg": utm_epsg(old.x, old.y), "origin": "location_fix_split", "location_source": "v2",
                              "same_site_plants": 1, "idx_v2": pd.NA, FIXED: True})  # fmt: skip
                new_rows.append(split[[c for c in plants.columns]])
                row["split"] = "detections stay at the old place as a plant of their own"
            plants.loc[i, sites] = np.nan
            if SIZE_SOURCE_COLUMN in plants.columns:
                plants.at[i, SIZE_SOURCE_COLUMN] = NO_IMAGE
            if fix.clear == "hydrowaste":
                plants.loc[i, hw_columns] = np.nan
                row["cleared"] = "HydroWASTE columns"
            if "geometry_v2" in plants.columns and pd.isna(plants.at[i, "geometry_v2"]):
                plants.at[i, "geometry_v2"] = old.wkt
            shift = float(GEOD.inv(old.x, old.y, new.x, new.y)[2])
            for col, value in (("epsg", utm_epsg(new.x, new.y)), ("ISO_2", fix.ISO_2),
                               ("location_source", fix.location_source), ("location_shift_m", shift)):  # fmt: skip
                if col in plants.columns:
                    plants.at[i, col] = value
            plants.loc[[i], "geometry"] = gpd.GeoSeries([new], index=[i], crs=plants.crs)
            plants.at[i, FIXED] = True
            row.update(status="moved", shift_km=round(shift / 1000, 1), lon=new.x, lat=new.y)
        else:
            plants.loc[i, eu_columns] = np.nan
            row.update(status="kept, EU code removed", cleared=fix.uwwCode)
            if fix.add_uwwCode and eu_file is not None:
                row["added"] = add_eu_plant(plants, new_rows, fix.add_uwwCode, eu_file, columns, cfg, site_m)
        report.append(row)
    added = pd.DataFrame(new_rows)
    if len(added):
        if "idx" in plants.columns and merge:
            start = int(pd.to_numeric(plants["idx"], errors="coerce").max()) + 1
            added["idx"] = np.arange(start, start + len(added))
        added = gpd.GeoDataFrame(added, geometry="geometry", crs=plants.crs)
        plants = pd.concat([plants, added], ignore_index=False)
        plants = gpd.GeoDataFrame(plants.reset_index(drop=True), geometry="geometry", crs=4326)
    report = pd.DataFrame(report)
    if merge:
        plants, merged = merge_with_neighbours(plants, cfg)
        if merged:
            report = pd.concat([report, pd.DataFrame(merged)], ignore_index=True)
    return plants.drop(columns=FIXED), report


def add_eu_plant(plants, new_rows, code, eu_file, columns, cfg, site_m):
    """Queue the corrected EU file's plant ``code`` for adding; returns what happened."""
    have = plants.get("uwwCode", pd.Series(index=plants.index, dtype=object)).astype(str) == code
    if any(str(r.get("uwwCode")) == code for r in new_rows) or have.any():
        return "not added: a plant with that code exists"
    eu = eu_file_plants(eu_file, [code], columns, cfg)
    if not len(eu):
        return "not added: code not in the EU file (or without location)"
    pt = eu.geometry.iloc[0]
    if (metres(plants.geometry, pt.x, pt.y) <= site_m).any():
        return f"not added: a plant within {site_m:g} m"
    row = pd.Series(np.nan, index=plants.columns, dtype=object)
    for c in eu.columns:
        if c in row.index:
            row[c] = eu[c].iloc[0]
    row.update({"geometry": pt, "epsg": utm_epsg(pt.x, pt.y), "origin": "new_plant_eu_file",
                "location_source": "corrected_eu_file", SIZE_SOURCE_COLUMN: NO_IMAGE, "same_site_plants": 1,
                "idx_v2": pd.NA, FIXED: True})  # fmt: skip
    new_rows.append(row[[c for c in plants.columns]])
    return f"added {code} from the corrected EU file"


def merge_with_neighbours(plants, cfg):
    """Merge every moved or added plant with the plants within same_site_distance_m (plants_v3's rule)."""
    from .plants_v3 import same_sites

    fixed = plants[FIXED].fillna(False).astype(bool).to_numpy()
    groups = [g for g in same_sites(plants, float(cfg["same_site_distance_m"])) if fixed[g].any()]
    if not groups:
        return plants, []
    rank = {s: k for k, s in enumerate(cfg["location_priority"])}
    drop, rows, report = [], [], []
    for g in groups:
        sub = plants.iloc[g]
        sub = sub.iloc[np.argsort([rank.get(s, len(rank)) for s in sub["location_source"]], kind="stable")]
        merged = sub.bfill().iloc[0].copy()
        merged["geometry"] = sub.geometry.iloc[0]
        if "same_site_plants" in sub:
            merged["same_site_plants"] = int(pd.to_numeric(sub["same_site_plants"], errors="coerce").fillna(1).sum())
        if "POP_SERVED" in sub:
            merged["POP_SERVED"] = pd.to_numeric(sub["POP_SERVED"], errors="coerce").max()
        if "idx_v2" in sub:
            oldest = sub.iloc[np.argmin(pd.to_numeric(sub["idx_v2"], errors="coerce").fillna(np.inf).to_numpy())]
            merged["idx_v2"] = oldest["idx_v2"]
            if "idx" in sub:
                merged["idx"] = oldest["idx"]
        rows.append(merged)
        drop.extend(sub.index)
        names = ", ".join(str(n) for n in sub.get("WWTP_NAME", sub.get("uwwName", pd.Series(dtype=object))).dropna())
        report.append({"status": f"same site: {len(sub)} plants merged", "name": names})
    out = pd.concat([plants.drop(index=drop), pd.DataFrame(rows)])
    if "idx" in out:
        out = out.sort_values("idx", kind="stable")
    return gpd.GeoDataFrame(out.reset_index(drop=True), geometry="geometry", crs=4326), report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()
    cfg = load_config(script_name="plants_v3")
    paths = cfg["paths"]
    plants = gpd.read_file(paths["output"]).to_crs(4326)
    fixes = read_fixes(paths["location_fixes"])
    fixed, report = apply_location_fixes(plants, fixes, cfg, eu_file=paths["eu_file"], merge=True)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(report.to_string(index=False))
    print(f"plants: {len(plants)} -> {len(fixed)}")
    missing = report["status"].astype(str).str.startswith("not found")
    if missing.any():  # stop before anything is written, so that a run chained after this does not start
        raise SystemExit(f"{int(missing.sum())} fixes found no plant: nothing written")
    if args.dry_run:
        return
    with atomic_output(paths["location_fixes_report"]) as tmp:
        report.to_csv(tmp, index=False)
    if len(fixed) == len(plants) and report["status"].isin(["already applied"]).all():
        print(f"nothing to change: {paths['output']} left as it is")
        return
    with atomic_output(paths["output"]) as tmp:
        fixed.to_file(tmp, driver="GPKG", engine="pyogrio")
    print(f"written: {paths['output']}, {paths['location_fixes_report']}")


if __name__ == "__main__":
    configure_logging()
    main()
