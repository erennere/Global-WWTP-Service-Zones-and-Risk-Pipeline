"""Work list: the plants that need a new image to get a size, and the new plants.

Settings: config.yaml, section ``work_list``. Command: ``python -m wwtp.data_merge.work_list``.

Three parts (column ``origin``), in the columns of the population layer:
- ``without_heigit_size``: plants of ``population_layer`` (the configured variant of version 2) without
  detected tanks (``footprint_m2`` 0); ``heigit_size_group`` says whether a lagoon was detected.
- ``new_plant_national``: the national plants (``national_plants``: AU, CA, JP, NZ).
- ``new_plant_eu_file``: plants of the corrected EU file (``eu_file``) that all_merged_v2 lacks.

The corrected EU file is the truth in its countries. A file plant's country is its uwwCode prefix
(``code_countries`` maps the prefixes that are not ISO 3166 codes, EL -> GR), or the overseas territory
with an ISO code of its own that it lies in (``territories``: Guadeloupe GP, Martinique MQ, ...).
- A plant without size whose uwwCode is in the file sits at the file's location. In a file country, a
  plant whose code is not in the file is kept at the nearest file location within
  ``same_site_distance_m``, otherwise removed (``plants_removed_not_in_eu_file.csv``).
- A located file plant whose code is not in the population layer is new, unless its code is in
  all_merged_v2 or it lies within same_site_distance_m of an all_merged_v2 plant
  (``eu_file_plants_not_added.csv``).

``idx`` names the plant's tile, OSM files and image. Plants of the population layer keep theirs (their
all_merged_v2 row). New plants are numbered after all_merged_v2's highest: national ones first, each
part in its source's order. ``plant_id``: the SITE_UID, the national WASTE_ID or the uwwCode.
``location_source`` (heigit_layer, corrected_eu_file, national_plants), ``location_shift_m`` and
``eu_file_uwwCode`` say where the location comes from.

The new images were taken from the list of 17 Nov 2026; ``paths.reference_list`` (in git) keeps its
plants and row order (idx, plant_id, origin, ISO_2, lon, lat). Its plants stay as they were, matched by
plant_id: first, in its row order, with its idx and its position (a plant the rebuild puts farther than
``reference_offset_m`` away goes back there). A plant of the reference that the rebuild lacks is kept
from the reference (those columns; ``location_source`` reference_list). Plants the reference lacks come
after it: plants without size keep their idx, new plants are numbered after the highest idx.

Writes the list to ``paths.work_list`` (GeoPackage) and the two side lists next to it, and the new plants
alone (the national and EU-file plants all_merged_v2 lacks, in the list's order and numbering, without the
columns empty for all of them) to ``paths.new_plants``. ``--new-plants-only`` writes only that file, from
the list as it is.
"""

from __future__ import annotations

import argparse
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Geod

from ..geo_utils import estimate_utm_epsg_for_geom, nearest_distances_m
from ..service_areas import _truthy
from ..starter import load_config
from ..utils import atomic_output, configure_logging

logger = logging.getLogger(__name__)
GEOD = Geod(ellps="WGS84")

# Columns of the population layer (create_voronoi / add_pop; its geometry is the service area, the plant
# point is WKT_WWTP) and of the corrected EU file
FOOTPRINT, LAGOON, CODE, COUNTRY, SITE, POINT = (
    "footprint_m2",
    "lagoon_detected",
    "uwwCode",
    "ISO_2",
    "SITE_UID",
    "WKT_WWTP",
)
# The columns that hold plant_id, per origin (national(), eu_new_plants())
ID_COLUMNS = {"new_plant_national": ["OLD_WASTE_ID"], "new_plant_eu_file": [CODE, "eu_file_uwwCode"]}
# Columns the list adds after those of the population layer
ADDED = ["heigit_size_group", "location_source", "location_shift_m", "eu_file_uwwCode", "newly_found_WWTPs",
         "origin", "plant_id"]  # fmt: skip


def located(gdf):
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]


def file_countries(eu, cfg):
    """ISO 3166 country of each file plant: its uwwCode prefix (``code_countries`` maps EL -> GR), or the
    overseas territory with an ISO code of its own that it lies in (``territories``, e.g. Guadeloupe GP)."""
    prefix = eu[CODE].str[:2]
    country = prefix.map(cfg["code_countries"]).fillna(prefix)
    x, y = eu.geometry.x, eu.geometry.y  # NaN without a location
    for code, (x0, y0, x1, y1) in cfg["territories"].items():
        country[x.between(x0, x1) & y.between(y0, y1)] = code
    return country


def as_text(value):
    """A value as the population layer's text columns hold it: numbers as '12.0', True / False as '1.0' / '0.0'."""
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return str(float(value))
    return value


def nearest(source, target):
    """Geodesic metres from each target row to the nearest source row, and that row's position."""
    if not len(source):
        return np.full(len(target), np.inf), np.full(len(target), -1)
    return nearest_distances_m(source, target)


def plant_points(pop):
    """The population layer with the plant points as geometry."""
    return pop.set_geometry(gpd.GeoSeries.from_wkt(pop[POINT], crs=4326))


def without_size(pop, cfg):
    """Plants of the population layer without detected tanks, with their size group."""
    out = pop[pd.to_numeric(pop[FOOTPRINT], errors="coerce").fillna(0) == 0].copy()
    groups = cfg["size_groups"]
    out["heigit_size_group"] = np.where(_truthy(out[LAGOON]), groups["lagoon"], groups["nothing"])
    out["location_source"], out["origin"], out["plant_id"] = "heigit_layer", "without_heigit_size", out[SITE]
    return out


def apply_eu_file(plants, eu, cfg):
    """The file's locations for the plants in it; plants of its countries that are not in it removed."""
    limit = float(cfg["same_site_distance_m"])
    eu_loc = located(eu).reset_index(drop=True)
    by_code = eu_loc.set_index(CODE).geometry
    countries = set(file_countries(eu, cfg))
    in_file = plants[CODE].isin(set(eu[CODE]))
    d_near, j_near = nearest(eu_loc, plants)
    near = (~in_file & plants[COUNTRY].isin(countries)).to_numpy()
    keep = ~near | (d_near <= limit)
    code = pd.Series(np.where(near, eu_loc[CODE].to_numpy()[np.clip(j_near, 0, None)], plants[CODE]), plants.index)
    code = code.where(in_file | near)
    moved = code.notna() & code.isin(set(by_code.index))
    new_geom = gpd.GeoSeries(by_code.reindex(code[moved]).to_numpy(), index=code[moved].index, crs=plants.crs)
    removed = pd.DataFrame({
        "idx": plants["idx"], "SITE_UID": plants[SITE], "hydrowaste_id": plants["OLD_WASTE_ID"],
        "WWTP_NAME": plants["WWTP_NAME"], "ISO_2": plants[COUNTRY], "lon": plants.geometry.x,
        "lat": plants.geometry.y, "heigit_size_group": plants["heigit_size_group"],
        "distance_to_nearest_plant_of_eu_file_m": np.round(d_near), "nearest_uwwCode_in_eu_file": code,
    })[~keep]  # fmt: skip
    out = plants[keep].copy()
    moved = moved[keep]
    to = new_geom[moved[moved].index]
    shift = GEOD.inv(
        out.geometry[moved].x.to_numpy(), out.geometry[moved].y.to_numpy(), to.x.to_numpy(), to.y.to_numpy()
    )[2]
    out.loc[moved, "location_shift_m"] = np.round(shift, 1)
    out.loc[moved, "geometry"] = to
    out.loc[moved, POINT] = to.to_wkt(rounding_precision=-1)
    out.loc[moved, "location_source"] = "corrected_eu_file"
    out["eu_file_uwwCode"] = code[keep].where(moved)
    logger.info("EU file: %d plants without size at its location (%d moved > 1 m), %d removed", int(moved.sum()),
                int((shift > 1).sum()), len(removed))  # fmt: skip
    return out, removed


def eu_new_plants(eu, pop, merged, columns, cfg):
    """Located file plants that neither the population layer nor all_merged_v2 has."""
    limit = float(cfg["same_site_distance_m"])
    cand = located(eu)
    cand = cand[~cand[CODE].isin(set(pop[CODE].dropna()))].reset_index(drop=True)
    d_merged = nearest(located(merged), cand)[0]
    d_pop = nearest(located(pop), cand)[0]
    reason = np.select(
        [cand[CODE].isin(set(merged[CODE].dropna())).to_numpy(), d_merged <= limit],
        ["code in all_merged_v2", f"within {limit:g} m of an all_merged_v2 plant"],
        "",
    )
    country = file_countries(cand, cfg)
    not_added = pd.DataFrame({
        CODE: cand[CODE], "country": country, "lon": cand.geometry.x, "lat": cand.geometry.y, "reason": reason,
        "distance_to_all_merged_v2_m": np.round(d_merged), "distance_to_population_layer_m": np.round(d_pop),
        "in_population_layer": d_pop <= limit,
    })[reason != ""]  # fmt: skip
    new = cand[reason == ""].copy()
    new[COUNTRY] = country[reason == ""]
    new = new[[c for c in columns if c in new.columns] + ["geometry"]]
    text = [c for c in new.columns.drop(["geometry", COUNTRY]) if pd.api.types.is_string_dtype(pop[c])]
    new = new.assign(**{c: new[c].map(as_text, na_action="ignore") for c in text})  # text in the population layer
    new["combined_geometry"] = new.geometry.to_wkt(rounding_precision=-1)
    new["location_source"], new["eu_file_uwwCode"] = "corrected_eu_file", new[CODE]
    new["origin"], new["plant_id"] = "new_plant_eu_file", new[CODE]
    logger.info("EU file: %d located plants not in the population layer; %d new, %d not added (%s)", len(cand),
                len(new), len(not_added), not_added["reason"].value_counts().to_dict())  # fmt: skip
    return new, not_added


def national(nat):
    """The national plants; their id moves to OLD_WASTE_ID, as in the population layer."""
    out = nat.copy()
    out["OLD_WASTE_ID"], out["WASTE_ID"] = out["WASTE_ID"], None
    out["location_source"], out["origin"], out["plant_id"] = (
        "national_plants",
        "new_plant_national",
        out["OLD_WASTE_ID"],
    )
    return out


def build(cfg):
    paths = cfg["paths"]
    pop = plant_points(gpd.read_file(paths["population_layer"]))
    eu = gpd.read_file(paths["eu_file"]).to_crs(4326)
    if eu[CODE].isna().any() or eu[CODE].duplicated().any():
        raise ValueError(f"{paths['eu_file']}: every plant needs its own uwwCode")
    merged = pyogrio.read_dataframe(paths["corrected_all_filepath"], columns=["idx", CODE])
    nat = gpd.read_file(os.path.join(paths["national_plants_dir"], "national_plants.gpkg"), layer="national_plants")

    old, removed = apply_eu_file(without_size(pop, cfg), eu, cfg)
    columns = list(pop.columns.drop("geometry"))
    eu_new, not_added = eu_new_plants(eu, pop, merged, columns, cfg)
    new = pd.concat([national(nat), eu_new], ignore_index=True)
    new["idx"] = np.arange(len(new)) + int(merged["idx"].max()) + 1
    new["heigit_size_group"] = cfg["size_groups"]["new"]
    new["WKT_WWTP"] = new.geometry.to_wkt(rounding_precision=-1)
    new["epsg"] = [estimate_utm_epsg_for_geom(g) for g in new.geometry]
    out = pd.concat([old, new], ignore_index=True)
    out = gpd.GeoDataFrame(out.reindex(columns=columns + ADDED + ["geometry"]), geometry="geometry", crs=4326)
    check_ids(out)
    return out, removed, not_added


def check_ids(out):
    for column in ("idx", "plant_id"):
        if out[column].isna().any() or out[column].duplicated().any():
            raise ValueError(f"work list: {column} must be set and unique")


def keep_reference(out, ref, max_offset_m, new_group):
    """The plants of ``ref`` (the list the new images were taken from) as they were, matched by plant_id:
    first, in its row order, with its idx and its position (when the rebuild puts them farther than
    ``max_offset_m`` away). A plant of ``ref`` that ``out`` lacks is kept from ``ref``. Then the plants
    ``ref`` lacks: plants without size keep their idx, new plants are numbered after the highest idx."""
    ref = gpd.GeoDataFrame(ref, geometry=gpd.points_from_xy(ref["lon"], ref["lat"]), crs=4326)
    pid = out["plant_id"].astype(str)
    row_of = pd.Series(np.arange(len(out)), index=pid.to_numpy())
    found = ref["plant_id"].isin(row_of.index).to_numpy()

    rows = row_of[ref.loc[found, "plant_id"]].to_numpy()
    kept = out.iloc[rows].copy()
    renumbered = int((kept["idx"].to_numpy() != ref.loc[found, "idx"].to_numpy()).sum())
    reordered = int((rows != ref.index[found]).sum())
    kept.index = ref.index[found]
    kept["idx"] = ref.loc[found, "idx"]
    there = ref.geometry[found]
    offset = GEOD.inv(kept.geometry.x.to_numpy(), kept.geometry.y.to_numpy(), there.x.to_numpy(), there.y.to_numpy())[2]
    back = offset > max_offset_m
    kept.loc[back, "geometry"] = there[back]
    kept.loc[back, POINT] = there[back].to_wkt(rounding_precision=-1)

    gone = ref.loc[~found, ["idx", "plant_id", "origin", "ISO_2", "geometry"]].copy()
    gone[POINT] = gone.geometry.to_wkt(rounding_precision=-1)
    gone["epsg"] = [estimate_utm_epsg_for_geom(g) for g in gone.geometry]
    gone["heigit_size_group"] = np.where(gone["origin"] == "without_heigit_size", None, new_group)
    gone["location_source"] = "reference_list"
    for origin, columns in ID_COLUMNS.items():
        gone.loc[gone["origin"] == origin, columns] = gone["plant_id"]

    extra = out[~pid.isin(set(ref["plant_id"])).to_numpy()].copy()
    new = (extra["origin"] != "without_heigit_size").to_numpy()
    first = max(int(ref["idx"].max()), int(extra.loc[~new, "idx"].max()) if (~new).any() else -1) + 1
    extra.loc[new, "idx"] = np.arange(int(new.sum())) + first

    kept_and_gone = pd.concat([kept, gone]).sort_index()
    result = pd.concat([kept_and_gone, extra], ignore_index=True)
    result = gpd.GeoDataFrame(result.reindex(columns=out.columns), geometry="geometry", crs=out.crs)
    result["idx"] = result["idx"].astype(int)
    logger.info("work list: the %d plants of the reference first, in its order (%d rows of the rebuild at another "
                "row, %d with another idx); %d moved back to their reference position (up to %.0f m: %s); %d "
                "kept from the reference, not in the rebuild (%s); %d plants the reference lacks after it (%s)",
                len(ref), reordered, renumbered, int(back.sum()), offset.max(initial=0),
                kept.loc[back, "plant_id"].astype(str).tolist(), int((~found).sum()), gone["plant_id"].tolist(),
                len(extra), extra["origin"].value_counts().to_dict())  # fmt: skip
    return result


def new_plants(out):
    """The plants all_merged_v2 lacks (``origin`` new_plant_national, new_plant_eu_file) in the list's order,
    with their idx, without the columns that are empty for all of them."""
    new = out[out["origin"].isin(list(ID_COLUMNS))]
    filled = [c for c in new.columns if c == "geometry" or new[c].replace("", np.nan).notna().any()]
    return new[filled]


def write_new_plants(cfg, out):
    path = cfg["paths"]["new_plants"]
    new = new_plants(out)
    with atomic_output(path) as tmp:
        new.to_file(tmp, driver="GPKG", layer="new_plants", engine="pyogrio")
    logger.info("new plants: %d %s -> %s", len(new), new.groupby("origin")["ISO_2"].nunique().to_dict(), path)


def write(cfg, out, removed, not_added):
    path = cfg["paths"]["work_list"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with atomic_output(path) as tmp:
        out.to_file(tmp, driver="GPKG", engine="pyogrio")
    for key, frame in (("removed_plants", removed), ("eu_file_not_added", not_added)):
        with atomic_output(cfg["paths"][key]) as tmp:
            frame.to_csv(tmp, index=False)
    write_new_plants(cfg, out)


def main():
    parser = argparse.ArgumentParser(description="Work list of plants that need a new image, and the new plants.")
    parser.add_argument("--new-plants-only", action="store_true",
                        help="only write paths.new_plants, from the work list as it is (nothing rebuilt)")  # fmt: skip
    args = parser.parse_args()
    cfg = load_config(script_name="work_list")
    if args.new_plants_only:
        write_new_plants(cfg, gpd.read_file(cfg["paths"]["work_list"], engine="pyogrio"))
        return
    out, removed, not_added = build(cfg)
    path = cfg["paths"]["reference_list"]
    ref = pd.read_csv(path, dtype={"plant_id": str, "origin": str})
    out = keep_reference(out, ref, float(cfg["reference_offset_m"]), cfg["size_groups"]["new"])
    check_ids(out)
    write(cfg, out, removed, not_added)
    logger.info("work list: %d plants %s", len(out), out["origin"].value_counts().to_dict())


if __name__ == "__main__":
    configure_logging()
    main()
