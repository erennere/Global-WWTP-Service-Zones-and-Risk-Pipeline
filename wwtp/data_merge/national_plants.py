"""National plant lists for the countries our plant list lacks, in the columns of the plant list.

Settings: config.yaml, section ``national_plants``. Command: ``python -m wwtp.data_merge.national_plants``.

Sources (``paths``):
- Australia: Geoscience Australia National Wastewater Treatment Facilities (``registers.AU`` in
  ``register_plants``, the national_data download); operational plants, without storm sewage and
  septage facilities (``left_out_subtypes``).
- Canada: ECCC WSER plants farther than national_data.match_distance_m from every plant of ours
  (``registers.CA``, flag ``new`` of ``register_plants``).
- Japan: MLIT KSJ P22 FY2012, sewage treatment plants (layer P22b) with a water treatment facility,
  one zip per prefecture in ``p22_dir``.
- New Zealand: no national point layer; OpenStreetMap ``man_made=wastewater_plant``
  (``osmgeo_filepath``) inside ``osm_bbox``, parts closer than ``osm_group_distance_m`` one plant,
  named pump stations, drinking-water plants and oil separators left out (``osm_skip_names``).
- HydroWASTE (``hydrowaste``): the records of ``hydrowaste_countries`` (not covered by our list),
  operational or of unknown status.

A national (or OpenStreetMap) plant and a HydroWASTE record are one plant, one to one and shortest
pairs first, when they are closer than ``match_distance_m``; or carry the same name and are closer
than ``name_match_distance_m``; or (``approximate_match_countries``) the HydroWASTE location is
approximate (QUAL_LOC not 1) and they are closer than ``approximate_match_distance_m``. The national
location is kept, the HydroWASTE one goes to ``HW_geometry``. Left out (``national_plants_left_out.csv``):
a second HydroWASTE record named like an already matched plant, and in ``complete_register_countries``
a HydroWASTE record without a register plant within match_distance_m but one within
``complete_register_distance_m`` (the register is complete there: a shifted duplicate).

Writes to ``paths.output_dir``: ``national_plants.gpkg`` (layer ``national_plants`` with exactly the columns
and types of the plant list; table ``national_attributes`` with every original attribute as JSON, the
sources, register id and year, and the matching rule), ``national_plants.csv`` (main fields) and
``national_plants_left_out.csv``.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import Point

from ..starter import load_config
from ..utils import atomic_output, configure_logging, get_iso_codes

logger = logging.getLogger(__name__)

# HydroWASTE attributes copied into the plant list (names shared by both)
HW_COLUMNS = ["WASTE_ID", "SOURCE", "ORG_ID", "WWTP_NAME", "COUNTRY", "CNTRY_ISO", "QUAL_LOC", "LAT_OUT", "LON_OUT",
              "STATUS", "POP_SERVED", "QUAL_POP", "WASTE_DIS", "QUAL_WASTE", "LEVEL", "QUAL_LEVEL", "DF", "HYRIV_ID",
              "RIVER_DIS", "COAST_10KM", "COAST_50KM", "DESIGN_CAP", "QUAL_CAP"]  # fmt: skip
HW_TEXT = {"WWTP_NAME", "COUNTRY", "CNTRY_ISO", "STATUS", "LEVEL"}  # kept as written; numbers become '12.0'
# Field names of the P22b shapefiles (KS-PS-P22 v1.1), in the order of the specification
P22B = {
    "P22b_001": "administrative_area_code", "P22b_002": "facility_name", "P22b_003": "address",
    "P22b_004": "corporate_name", "P22b_005": "implementing_type", "P22b_006": "sewerage_type",
    "P22b_007": "plan_area_total_ha", "P22b_008": "plan_area_sewage_ha", "P22b_009": "plan_area_stormwater_ha",
    "P22b_010": "plan_area_combined_ha", "P22b_011": "plan_population_to_treat", "P22b_012": "plan_population_in_area",
    "P22b_013": "rainfall_intensity_min", "P22b_014": "rainfall_intensity_max", "P22b_015": "return_period_min",
    "P22b_016": "return_period_max", "P22b_017": "sewage_daily_average_min", "P22b_018": "sewage_daily_average_max",
    "P22b_019": "sewage_daily_maximum_min", "P22b_020": "sewage_daily_maximum_max", "P22b_021": "water_treatment",
    "P22b_022": "sludge_treatment", "P22b_023": "effective_utilization", "P22b_024": "reclaimed_water",
    "P22b_025": "discharge_river", "P22b_026": "treatment_area_present_ha", "P22b_027": "treatment_area_plan_ha",
    "P22b_028": "treated_population_present", "P22b_029": "treated_population_plan",
    "P22b_030": "population_in_plan_area", "P22b_031": "dry_weather_max_capacity_present_m3d",
    "P22b_032": "dry_weather_max_capacity_plan_m3d", "P22b_033": "wet_weather_max_capacity_present_m3d",
    "P22b_034": "wet_weather_max_capacity_plan_m3d", "TreeCode": "tree_code", "PosStat": "position_status",
    "ORigna": "information_source",
}  # fmt: skip
P22_REGISTER = "jpn_p22"  # label of the P22 plants in national_attributes
P22_CRS = 4612  # JGD2000 (the .prj of the P22 shapefiles)
P22_ENCODING = "cp932"  # Shift-JIS shapefiles
WSER_ID = "Identification de l'installation/ Facility Identification"
WSER_NAME = "Nom du système / System Name"


def text(value):
    """A value as the plant list stores it: numbers as '12.0' text, missing as None."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value in ("", "nan", "<NA>", "None"):
        return None
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return str(value)


def utm_epsg(lon, lat):
    """EPSG code of the UTM zone of a point, as the plant list's ``epsg`` column."""
    return float((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)


def tokens(name, stopwords):
    if not isinstance(name, str):
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", name.lower()) if t not in stopwords and len(t) > 1}


def same_name(a, b, stopwords):
    """The same plant name: the words of one contain the other's, or share at least half of them."""
    ta, tb = tokens(a, stopwords), tokens(b, stopwords)
    return bool(ta and tb) and (ta <= tb or tb <= ta or len(ta & tb) / len(ta | tb) >= 0.5)


def km(metres):
    """A distance as the label of a match rule, e.g. 25000 -> '25 km'."""
    return f"{metres / 1000:g} km"


def match(reg, hw, crs, approximate, cfg):
    """One-to-one pairs (register row, HydroWASTE row, metres, rule), shortest first, in passes; and the
    duplicate HydroWASTE records (a second one named like an already matched register plant)."""
    if not len(reg) or not len(hw):
        return [], []
    stop = set(cfg["name_stopwords"])
    pa, pb = reg.to_crs(crs), hw.to_crs(crs)
    d = np.hypot(
        pa.geometry.x.to_numpy()[:, None] - pb.geometry.x.to_numpy()[None, :],
        pa.geometry.y.to_numpy()[:, None] - pb.geometry.y.to_numpy()[None, :],
    )
    names_b = [a["WWTP_NAME"] for a in hw["attrs"]]
    approx_b = np.array([text(a["QUAL_LOC"]) not in (None, "1.0") for a in hw["attrs"]])
    near_limit, name_limit = float(cfg["match_distance_m"]), float(cfg["name_match_distance_m"])
    passes = [(km(near_limit), d < near_limit), (f"name {km(name_limit)}", d < name_limit)]
    if approximate:
        approx_limit = float(cfg["approximate_match_distance_m"])
        passes.append((f"approximate {km(approx_limit)}", (d < approx_limit) & approx_b[None, :]))
    used_a, used_b, pairs = set(), set(), []
    for rule, ok in passes:
        ia, ib = np.nonzero(ok)
        for k in np.argsort(d[ia, ib], kind="stable"):
            i, j = ia[k], ib[k]
            if i in used_a or j in used_b:
                continue
            if rule.startswith("name") and not same_name(reg["name"].iloc[i], names_b[j], stop):
                continue
            used_a.add(i), used_b.add(j), pairs.append((reg.index[i], hw.index[j], float(d[i, j]), rule))
    duplicates = []
    for j in range(len(hw)):
        if j in used_b:
            continue
        near = np.nonzero(d[:, j] < name_limit)[0]
        for i in near[np.argsort(d[near, j], kind="stable")]:
            if i in used_a and same_name(reg["name"].iloc[i], names_b[j], stop):
                duplicates.append((reg.index[i], hw.index[j], float(d[i, j])))
                break
    return pairs, duplicates


# --------------------------------------------------------------------------- #
# Sources                                                                     #
# --------------------------------------------------------------------------- #
def hydrowaste_records(cfg, iso3_of):
    """HydroWASTE records of hydrowaste_countries, operational or of unknown status, attributes as text."""
    hw = pd.read_csv(cfg["paths"]["hydrowaste"], low_memory=False)
    keep = hw["CNTRY_ISO"].isin([iso3_of[c] for c in cfg["hydrowaste_countries"]])
    keep &= (
        pd.to_numeric(hw["LON_WWTP"], errors="coerce").notna() & pd.to_numeric(hw["LAT_WWTP"], errors="coerce").notna()
    )
    hw = hw[keep].reset_index(drop=True)
    iso2_of = {v: k for k, v in iso3_of.items()}
    attrs = hw.astype(str).to_dict("records")
    out = gpd.GeoDataFrame(
        {"iso2": hw["CNTRY_ISO"].map(iso2_of).to_numpy(), "attrs": attrs},
        geometry=[Point(float(a["LON_WWTP"]), float(a["LAT_WWTP"])) for a in attrs],
        crs=4326,
    )
    return out[~out["attrs"].map(lambda a: a["STATUS"] in set(cfg["not_operational"]))]


def register_records(cfg):
    """Australia and Canada from the national_data register file."""
    table = gpd.read_file(cfg["paths"]["register_plants"])
    rows = []
    au, ca = cfg["registers"]["AU"], cfg["registers"]["CA"]
    for r in table[table["register"] == au].itertuples():
        a = json.loads(r.attrs)
        keep = a["status"] == "Operational" and a["feature_subtype"] not in set(cfg["left_out_subtypes"])
        rows.append({"iso2": "AU", "register": au, "rid": f"AU_GA_{a['objectid']}", "name": a["feature_name"],
                     "status": a["status"], "pop": np.nan, "attrs": a, "keep": keep, "geometry": Point(r.lon, r.lat)})  # fmt: skip
    for r in table[(table["register"] == ca) & table["new"].astype(bool)].itertuples():
        a = json.loads(r.attrs)
        rows.append({"iso2": "CA", "register": ca, "rid": f"CA_WSER_{a[WSER_ID]}", "name": a[WSER_NAME],
                     "status": "Operational", "pop": np.nan, "attrs": a, "keep": True, "geometry": Point(r.lon, r.lat)})  # fmt: skip
    return rows


def p22_records(cfg):
    """Japan: P22b of every prefecture zip, in file order. The plants without a water treatment
    facility (e.g. sludge composting works) are read too, so HydroWASTE records at them go with them."""
    files = sorted(glob.glob(os.path.join(cfg["paths"]["p22_dir"], "P22-12_*_GML.zip")))
    if not files:
        raise FileNotFoundError(f"no P22-12_<prefecture>_GML.zip in {cfg['paths']['p22_dir']}")
    rows = []
    for path in files:
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.namelist() if re.search(r"P22b-12_\d+\.shp$", m)]
        if len(members) != 1:
            raise ValueError(f"{path}: expected one P22b shapefile, found {members}")
        part = gpd.read_file(f"/vsizip/{path}/{members[0]}", encoding=P22_ENCODING)
        part = part.set_crs(P22_CRS, allow_override=True).to_crs(4326)
        for r in part.itertuples():
            a = {}
            for field, name in P22B.items():
                v = getattr(r, field)
                a[name] = None if isinstance(v, (int, float)) and v < 0 else v  # -1, -998: not given
            pop = a["treated_population_present"]
            rows.append({"iso2": "JP", "register": P22_REGISTER, "rid": f"JP_P22_{r.TreeCode}",
                         "name": r.P22b_002, "status": None, "pop": float(pop) if pop else np.nan, "attrs": a,
                         "keep": r.P22b_021 == 1, "geometry": r.geometry})  # fmt: skip
    logger.info("P22: %d treatment plants in %d prefecture files", len(rows), len(files))
    return rows


def osm_records(cfg, iso2, crs):
    """OpenStreetMap wastewater plants inside the country's box, parts within osm_group_distance_m one plant."""
    osm = gpd.read_file(cfg["paths"]["osmgeo_filepath"], bbox=tuple(cfg["osm_bbox"][iso2]))
    osm = osm[~osm["name"].fillna("").str.contains(cfg["osm_skip_names"], regex=True)].to_crs(crs)
    blobs = gpd.GeoDataFrame(geometry=[osm.buffer(float(cfg["osm_group_distance_m"]) / 2).union_all()], crs=crs)
    blobs = blobs.explode(index_parts=False).reset_index(drop=True).reset_index(names="g")
    osm["group"] = gpd.sjoin(osm, blobs, predicate="intersects")["g"].groupby(level=0).first()
    rows = []
    for _, part in osm.groupby("group"):
        names = [n for n in part["name"].dropna() if n]
        props = [{k: v for k, v in rec.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
                 for rec in part.drop(columns=["group", "geometry"]).to_dict("records")]  # fmt: skip
        point = gpd.GeoSeries([part.geometry.union_all().centroid], crs=crs).to_crs(4326).iloc[0]
        rows.append({"iso2": iso2, "register": "osm", "rid": f"{iso2}_OSM_{props[0]['@osmId'].replace('/', '-')}",
                     "name": names[0] if names else None, "status": None, "pop": np.nan,
                     "attrs": {"osm_features": props, "snapshot": props[0]["@snapshotTimestamp"]}, "keep": True,
                     "geometry": point})  # fmt: skip
    return rows


# --------------------------------------------------------------------------- #
# Rows in the plant list's columns                                            #
# --------------------------------------------------------------------------- #
def plant_columns(path):
    """Columns and SQLite types of the plant list (layer 0), without fid and geometry."""
    info = pyogrio.read_info(path)
    return [(name, dtype) for name, dtype in zip(info["fields"], info["dtypes"])]


def build(cfg):
    _, iso3_of, _, names = get_iso_codes()
    columns = plant_columns(cfg["paths"]["corrected_all_filepath"])
    keys = [c for c, _ in columns]
    hw = hydrowaste_records(cfg, iso3_of)
    reg_rows = register_records(cfg) + p22_records(cfg)
    for iso2 in cfg["osm_bbox"]:
        reg_rows += osm_records(cfg, iso2, int(cfg["metric_crs"][iso2]))
    reg = gpd.GeoDataFrame(reg_rows, crs=4326)

    rows, side, left_out, counts = [], [], [], {}

    def base(iso2, pt):
        row = dict.fromkeys(keys)
        row.update(COUNTRY=names[iso2], CNTRY_ISO=iso3_of[iso2], ISO_2=iso2, lat=text(pt.y), lon=text(pt.x),
                   neigh_lat=text(pt.y), neigh_lon=text(pt.x), combined_geometry=pt.wkt, epsg=utm_epsg(pt.x, pt.y))  # fmt: skip
        return row

    def add_hw(row, a):
        for c in HW_COLUMNS:
            row[c] = a[c] if c in HW_TEXT else text(a[c])
        row["POP_SERVED"] = float(a["POP_SERVED"]) if text(a["POP_SERVED"]) else None
        row["HW_geometry"], row["hydrowaste"] = Point(float(a["LON_WWTP"]), float(a["LAT_WWTP"])).wkt, "1.0"

    for iso2 in cfg["countries"]:
        r_c, h_c = reg[reg["iso2"] == iso2], hw[hw["iso2"] == iso2]
        crs = int(cfg["metric_crs"][iso2])
        pairs, duplicates = match(r_c, h_c, crs, iso2 in cfg["approximate_match_countries"], cfg)
        hw_of = {i: (j, d, rule) for i, j, d, rule in pairs}
        matched = {j for _, j, _, _ in pairs} | {j for _, j, _ in duplicates}
        for i, j, d in duplicates:
            left_out.append({"ISO_2": iso2, "WASTE_ID": text(h_c.at[j, "attrs"]["WASTE_ID"]),
                             "reason": f"second HydroWASTE record named like {r_c.at[i, 'rid']} ({d / 1000:.1f} km "
                                       "away), which already has one: a duplicate record",
                             "lon": h_c.at[j, "geometry"].x, "lat": h_c.at[j, "geometry"].y})  # fmt: skip
        if iso2 in cfg["complete_register_countries"]:
            rest = h_c[~h_c.index.isin(matched)]
            if len(rest):
                pr, pj = rest.to_crs(crs), r_c.to_crs(crs)
                dist = np.hypot(pr.geometry.x.to_numpy()[:, None] - pj.geometry.x.to_numpy()[None, :],
                                pr.geometry.y.to_numpy()[:, None] - pj.geometry.y.to_numpy()[None, :]).min(axis=1)  # fmt: skip
                for j, d in zip(rest.index, dist):
                    if d < float(cfg["complete_register_distance_m"]):
                        matched.add(j)
                        left_out.append({"ISO_2": iso2, "WASTE_ID": text(h_c.at[j, "attrs"]["WASTE_ID"]),
                                         "reason": f"HydroWASTE record {d / 1000:.1f} km from the nearest register "
                                                   "plant, where the register is complete: a shifted duplicate",
                                         "lon": h_c.at[j, "geometry"].x, "lat": h_c.at[j, "geometry"].y})  # fmt: skip
        counts[iso2] = {"left_out_register_records": int((~r_c["keep"]).sum()),
                        "hydrowaste_with_them": sum(1 for i, _, _, _ in pairs if not r_c.at[i, "keep"])}  # fmt: skip
        for i, r in r_c[r_c["keep"]].iterrows():
            row = base(iso2, r.geometry)
            sources = r["register"]
            if i in hw_of:
                j, d, _ = hw_of[i]
                add_hw(row, h_c.at[j, "attrs"])
                row["distance"], row["newly_found_WWTPs"] = text(d), "False"
                sources += "+hydrowaste"
            else:
                row["WASTE_ID"], row["newly_found_WWTPs"] = r["rid"], "True"
            row["corrected_geometry"], row["used_corrected_geometry"] = r.geometry.wkt, "True"
            row["COUNTRY"], row["CNTRY_ISO"] = names[iso2], iso3_of[iso2]
            if isinstance(r["name"], str) and r["name"]:
                row["WWTP_NAME"] = r["name"]
            if isinstance(r["status"], str) and r["status"]:
                row["STATUS"] = r["status"]
            if pd.notna(r["pop"]) and r["pop"] > 0:
                row["POP_SERVED"], row["QUAL_POP"] = r["pop"], "1.0"  # reported by the national register
            if r["register"] == cfg["registers"]["CA"]:  # WSER fields have their own columns, as in our list
                for k, v in r["attrs"].items():
                    if k in row:
                        row[k] = None if v in ("", None) else str(v)
            if r["register"] == "osm":
                row["osm"], row["matched_osm_geometry"] = "1.0", r.geometry.wkt
            rows.append((row, r.geometry))
            side.append({"WASTE_ID": row["WASTE_ID"], "ISO_2": iso2, "sources": sources, "register": r["register"],
                         "register_year": cfg["register_years"][r["register"]], "register_id": r["rid"],
                         "hydrowaste_id": row["WASTE_ID"] if i in hw_of else None,
                         "distance_register_hydrowaste_m": hw_of[i][1] if i in hw_of else None,
                         "match_rule": hw_of[i][2] if i in hw_of else None,
                         "register_attrs": json.dumps(r["attrs"], ensure_ascii=False, default=str),
                         "hydrowaste_attrs": json.dumps(h_c.at[hw_of[i][0], "attrs"], ensure_ascii=False)
                         if i in hw_of else None})  # fmt: skip
        for j, h in h_c.iterrows():
            if j in matched:
                continue
            row = base(iso2, h.geometry)
            add_hw(row, h["attrs"])
            row["used_corrected_geometry"], row["newly_found_WWTPs"] = "False", "False"
            row["COUNTRY"], row["CNTRY_ISO"] = names[iso2], iso3_of[iso2]
            rows.append((row, h.geometry))
            side.append({"WASTE_ID": row["WASTE_ID"], "ISO_2": iso2, "sources": "hydrowaste", "register": None,
                         "register_year": None, "register_id": None, "hydrowaste_id": row["WASTE_ID"],
                         "distance_register_hydrowaste_m": None, "match_rule": None, "register_attrs": None,
                         "hydrowaste_attrs": json.dumps(h["attrs"], ensure_ascii=False)})  # fmt: skip
        logger.info("%s: %d plants (%s)", iso2, sum(1 for s in side if s["ISO_2"] == iso2),
                    pd.Series([s["sources"] for s in side if s["ISO_2"] == iso2]).value_counts().to_dict())  # fmt: skip

    out = gpd.GeoDataFrame([r for r, _ in rows], geometry=[g for _, g in rows], crs=4326)[keys + ["geometry"]]
    for c, dtype in columns:
        if dtype.startswith("float"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        elif dtype.startswith("int"):
            out[c] = pd.array(pd.to_numeric(out[c], errors="coerce"), dtype="Int64")
        else:
            out[c] = out[c].astype(object).where(out[c].notna(), None)
    if not (out["WASTE_ID"].notna().all() and out["WASTE_ID"].is_unique):
        raise ValueError("national plants: WASTE_ID must be set and unique")
    return (
        out,
        pd.DataFrame(side),
        pd.DataFrame(left_out, columns=["ISO_2", "WASTE_ID", "reason", "lon", "lat"]),
        counts,
    )


def write(cfg, out, side, left_out):
    folder = cfg["paths"]["output_dir"]
    os.makedirs(folder, exist_ok=True)
    gpkg = os.path.join(folder, "national_plants.gpkg")
    with atomic_output(gpkg) as tmp:
        out.to_file(tmp, layer="national_plants", driver="GPKG", engine="pyogrio")
        pyogrio.write_dataframe(side, tmp, layer="national_attributes", driver="GPKG")
    flat = pd.DataFrame(out.drop(columns="geometry")).merge(
        side[["WASTE_ID", "sources", "register_id", "hydrowaste_id", "match_rule"]], on="WASTE_ID"
    )
    main_fields = ["WASTE_ID", "ISO_2", "sources", "WWTP_NAME", "lon", "lat", "STATUS", "POP_SERVED", "QUAL_POP",
                   "register_id", "hydrowaste_id", "match_rule", "HW_geometry", "distance"]  # fmt: skip
    for name, frame in (("national_plants.csv", flat[main_fields]), ("national_plants_left_out.csv", left_out)):
        with atomic_output(os.path.join(folder, name)) as tmp:
            frame.to_csv(tmp, index=False)


def main(argv=None):
    cfg = load_config(script_name="national_plants")
    out, side, left_out, counts = build(cfg)
    write(cfg, out, side, left_out)
    logger.info("national plants: %d (%s); left out %d; register records not kept %s", len(out),
                out["ISO_2"].value_counts().to_dict(), len(left_out), counts)  # fmt: skip


if __name__ == "__main__":
    configure_logging()
    main()
