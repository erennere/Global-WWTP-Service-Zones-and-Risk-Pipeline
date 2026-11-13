"""Plant- and district-level check of the calibration on every dataset of real catchments, and the choice.

``calibrate`` (``wwtp.calibration.connection_radius``) fits the radius rule and the national size scale
on national shares (JMP) for every pair of size-scale limits and cap. This module tests every such pair
where the real catchments are known, and picks the pair the pipeline uses.

**Plant references** (``connection_radius.plant_reference_kinds`` and ``plant_reference_polygons``): one
unit is a real catchment with one or more of our plants; its reference people are

==================  =========================================================================  ==========
reference           unit and our plants                                                        people
==================  =========================================================================  ==========
us_sewersheds       EPA sourced sewershed (one CWNS ID, no flag); our plant with that CWNS ID,  population
                    within ``us_match_distance_m`` of the polygon                              raster
gb_catchments       Hoffmann et al. catchment(s) of one plant (``match_gb`` of the sweep        population
                    validation)                                                                raster
nl_cbs              CBS RWZI; our plant with uwwCode NL + RWZI code                             CBS count
polygons            any polygon file (NL purification units, Irish agglomerations, Spanish     population
                    agglomerations, ...): our plants of the country within                    raster
                    ``polygon_match_distance_m`` of the polygon, each to the nearest polygon
==================  =========================================================================  ==========

"population raster" is the WorldPop raster of ``year`` summed inside the polygon (exactextract), the
same raster the stand-in counts: only the shape of the service areas is tested. A reference whose file
is missing or unreadable is listed in ``plant_check_status.csv`` and left out.

**Stand-in** (as the curves of ``connection_radius``): every 1 km cell of the country goes to the plant
with the smallest distance / sqrt(people) among its ``candidate_plants`` nearest and is served when it
lies inside that plant's circle ``min(r10k * (k * people / 10,000) ** e, max_buffer)``. Per
configuration: the fixed radius (``radius_at_10k_people_km``, no scale) and every row of
``calibration_by_cap.csv`` at the size measure's exponent (``create_voronoi.radius_calibration``):
``k`` = the country's scale within the row's limits, ``r10k = min(cap, exp(a + b logit C))``. Error per
reference: class-balanced error of people per unit (``service_radius.class_balanced_error``) and total.

**District check** (``district_references``): census localities as points (Mexico's ITER), each served
when inside the circle of the plant it goes to; per municipality the served share against the share of
dwellings with drainage (an upper bound: drainage includes septic tanks). Reported, not used in the
choice.

**Choice** per size measure: among the rows whose plant error (mean over the reference countries of the
mean over their references, log10) is not above the fixed radius's, the one with the smallest national
error, the mean of ``choice_national`` (``jmp``: leaving one SDG region out; ``sdg631`` etc.: the same
served shares against that reference). Also the choice with each reference country left out.

Command: ``python -m wwtp.calibration.plant_check run`` -> plant_references.csv, plant_check.csv,
plant_choice.csv, district_check.csv, plant_check_status.csv in ``connection_radius.paths.output_dir``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..pop_validation_scripts.national_data import _spatial_layers, source_path
from ..sensitivity_analysis_scripts.sweep_validation import match_gb, match_nl, zonal_people
from ..starter import load_config
from ..utils import atomic_output, configure_logging
from .connection_radius import aggregate_cells, logit, raster_for_year, sphere_xyz, weighted_nearest
from .service_radius import class_balanced_error

logger = logging.getLogger(__name__)

DMS = re.compile(r"(\d+)\D+(\d+)\D+([\d.]+)\D*([NSEW])")


# --------------------------------------------------------------------------- #
# References                                                                  #
# --------------------------------------------------------------------------- #
def plant_points(plants):
    """plants.csv of ``curves`` as points; SITE_UID is the row number."""
    return gpd.GeoDataFrame(
        plants[["iso2", "uwwCode", "cwns"]].assign(SITE_UID=np.arange(len(plants))),
        geometry=gpd.points_from_xy(plants["lon"], plants["lat"]),
        crs=4326,
    )


def read_polygons(path):
    """Every polygon layer of a file (or of the spatial members of a zip), valid, in EPSG:4326."""
    parts = []
    for _, file, layer in _spatial_layers(path):
        part = gpd.read_file(file, layer=layer)
        if part.crs is None:
            raise ValueError(f"{file} layer {layer} has no CRS")
        part = part[part.geometry.notna() & ~part.geometry.is_empty].to_crs(4326)
        parts.append(part[part.geom_type.isin(["Polygon", "MultiPolygon"])])
    polygons = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=4326)
    polygons["geometry"] = polygons.geometry.force_2d().buffer(0)  # valid, and still (multi)polygons
    return polygons[~polygons.geometry.is_empty].reset_index(drop=True)


def polygon_units(polygons, plants, iso2, distance_m, raster):
    """Our plants of ``iso2`` within ``distance_m`` of a polygon, each to the nearest; people in the polygon."""
    ours = plants[plants["iso2"] == iso2]
    crs = polygons.estimate_utm_crs()
    near = gpd.sjoin_nearest(
        ours.to_crs(crs)[["SITE_UID", "geometry"]],
        polygons.reset_index(names="polygon").to_crs(crs)[["polygon", "geometry"]],
        max_distance=distance_m,
        distance_col="distance_m",
    )
    near = near.sort_values("distance_m").drop_duplicates("SITE_UID")
    used = np.sort(near["polygon"].unique())
    units = pd.DataFrame({"unit": used.astype(str), "ref": zonal_people(polygons.geometry.iloc[used], raster)})
    matches = pd.DataFrame({"unit": near["polygon"].astype(str).to_numpy(), "SITE_UID": near["SITE_UID"].to_numpy()})
    return units, matches


def us_units(sv, plants, raster):
    """EPA sourced sewersheds (one CWNS ID, no flag) to our plant with that CWNS ID near its polygon."""
    s = gpd.read_file(f"/vsizip/{sv['paths']['us_sewersheds']}/Version_1_3.gdb", layer="Sewersheds")
    s = s[(s["Method"] == "Sourced") & ~s["CWNS_ID"].str.contains(";") & s["Flag"].isna()].to_crs(4326)
    j = plants[plants["cwns"].notna()].merge(s[["CWNS_ID", "geometry"]], left_on="cwns", right_on="CWNS_ID")
    j = j[j.groupby("CWNS_ID")["CWNS_ID"].transform("size") == 1]
    poly = gpd.GeoSeries(j["geometry_y"], crs=4326).make_valid()
    crs = poly.estimate_utm_crs()  # the sourced states; a few hundred metres of distortion do not matter
    near = (
        poly.to_crs(crs)
        .buffer(float(sv["us_match_distance_m"]))
        .contains(gpd.GeoSeries(j["geometry_x"], crs=4326).to_crs(crs))
    )
    j, poly = j[near.to_numpy()], poly[near.to_numpy()]
    units = pd.DataFrame({"unit": j["CWNS_ID"].to_numpy(), "ref": zonal_people(poly, raster)})
    return units, pd.DataFrame({"unit": units["unit"], "SITE_UID": j["SITE_UID"].to_numpy()})


def build_references(cfg, sv, plants):
    """Units (reference, iso2, unit, ref), matches (reference, unit, SITE_UID) and the status per reference."""
    year = int(cfg["year"])
    tif_dir = cfg["paths"]["pop_tif_dir"]
    jobs = {kind: kind for kind in cfg["plant_reference_kinds"]}
    jobs.update(cfg["plant_reference_polygons"])
    country_of = {"us_sewersheds": "US", "gb_catchments": "GB", "nl_cbs": "NL"}
    units, matches, status = [], [], []
    for name, job in jobs.items():
        iso2 = country_of[job] if isinstance(job, str) else job["iso2"]
        try:
            if job == "us_sewersheds":
                u, m = us_units(sv, plants, raster_for_year("US", tif_dir, year))
            elif job == "gb_catchments":
                u, m, _ = match_gb({**sv, "reference_years": {**sv["reference_years"], "gb": year}}, plants)
            elif job == "nl_cbs":
                u, m = match_nl(sv, plants)
            else:
                path = source_path(cfg, job["source"], job["file"])
                if not os.path.exists(path):
                    raise FileNotFoundError(f"not downloaded: {path}")
                u, m = polygon_units(
                    read_polygons(path),
                    plants,
                    iso2,
                    float(cfg["polygon_match_distance_m"]),
                    raster_for_year(iso2, tif_dir, year),
                )
        except Exception as err:  # one missing or unreadable reference must not hide the others
            status.append({"reference": name, "iso2": iso2, "status": f"{type(err).__name__}: {err}"[:300]})
            logger.warning("%s: %s", name, status[-1]["status"])
            continue
        u = u[u["ref"] > 0].assign(reference=name, iso2=iso2, unit=lambda d: d["unit"].astype(str))
        m = m.assign(reference=name, unit=m["unit"].astype(str))
        m = m[m["unit"].isin(u["unit"])]
        units.append(u[["reference", "iso2", "unit", "ref"]])
        matches.append(m[["reference", "unit", "SITE_UID"]])
        status.append(
            {
                "reference": name,
                "iso2": iso2,
                "status": "ok",
                "units": len(u),
                "plants": int(m["SITE_UID"].nunique()),
                "people": float(u["ref"].sum()),
            }
        )
        logger.info("%s: %d units, %d plants, %.1f M people", name, len(u), len(m), u["ref"].sum() / 1e6)
    columns = (["reference", "iso2", "unit", "ref"], ["reference", "unit", "SITE_UID"])
    units = pd.concat(units, ignore_index=True) if units else pd.DataFrame(columns=columns[0])
    matches = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(columns=columns[1])
    return units, matches, pd.DataFrame(status)


# --------------------------------------------------------------------------- #
# Configurations and the stand-in                                             #
# --------------------------------------------------------------------------- #
def configurations(calibration, exponents, fixed_r10k):
    """The fixed radius and every row of calibration_by_cap at the size measure's exponent."""
    rows = []
    for measure, exponent in exponents.items():
        rows.append(
            {
                "size_measure": measure,
                "radius_exponent": float(exponent),
                "scale_min": 1.0,
                "scale_max": 1.0,
                "cap_km": np.nan,
                "a": np.nan,
                "b": np.nan,
                "fixed_r10k_km": float(fixed_r10k[measure]),
            }
        )
        t = calibration[
            (calibration["size_measure"] == measure)
            & np.isclose(calibration["radius_exponent"], float(exponent))
            & (calibration["cap_km"] > 0)
        ]
        for r in t.itertuples():
            rows.append(
                {
                    "size_measure": measure,
                    "radius_exponent": float(exponent),
                    "scale_min": float(r.scale_min),
                    "scale_max": float(r.scale_max),
                    "cap_km": float(r.cap_km),
                    "a": float(r.a),
                    "b": float(r.b),
                    "fixed_r10k_km": np.nan,
                }
            )
    return pd.DataFrame(rows)


def country_setting(conf, share, connected_people, sum_size):
    """(k, r10k_km) of a country under one configuration."""
    if np.isfinite(conf["fixed_r10k_km"]):
        return 1.0, float(conf["fixed_r10k_km"])
    k = float(np.clip(connected_people / sum_size, conf["scale_min"], conf["scale_max"]))
    return k, float(min(conf["cap_km"], np.exp(conf["a"] + conf["b"] * logit(share))))


def served_by_plant(winner, dist, people, size, k, exponent, r10k_km, max_buffer_m):
    """People each plant serves: cells inside the circle of the plant they go to."""
    radius = np.minimum(r10k_km * (k * size / 1e4) ** exponent * 1000.0, max_buffer_m)
    served = dist <= radius[winner]
    return np.bincount(winner[served], weights=people[served], minlength=len(size))


def parse_dms(text):
    """Degrees of a 'DD°MM'SS.sss" H' string (W and S negative); NaN when it does not parse."""
    m = DMS.search(str(text))
    if not m:
        return np.nan
    value = float(m.group(1)) + float(m.group(2)) / 60 + float(m.group(3)) / 3600
    return -value if m.group(4) in "WS" else value


def read_localities(path, spec):
    """Census localities with coordinates: district, people, dwellings, dwellings with drainage."""
    c = spec["columns"]
    with zipfile.ZipFile(path) as archive, archive.open(spec["member"]) as f:
        t = pd.read_csv(f, dtype=str, encoding="utf-8-sig", usecols=list(c.values()))
    t = t[~t[c["locality"]].isin(spec["aggregate_localities"]) & (t[c["municipality"]] != spec["total_municipality"])]
    out = pd.DataFrame(
        {
            "district": t[c["state"]] + t[c["municipality"]],
            "lon": t[c["lon"]].map(parse_dms),
            "lat": t[c["lat"]].map(parse_dms),
            **{k: pd.to_numeric(t[c[k]], errors="coerce") for k in ("people", "dwellings", "drainage")},
        }
    )
    return out.dropna().reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #
def check(cfg, plants, countries, references, calibration, units, matches, exponents, fixed_r10k):
    """plant_check rows (one per configuration and reference) and district_check rows."""
    year, tif_dir = int(cfg["year"]), cfg["paths"]["pop_tif_dir"]
    max_buffer = float(cfg["max_buffer"])
    confs = configurations(calibration, exponents, fixed_r10k)
    info = countries.set_index("iso2")
    share = references.set_index("iso3")[f"{cfg['fit_reference']}_share"]
    rows, district_rows = [], []

    def setting(conf, iso2):
        c = info.loc[iso2]
        s = float(share[c["iso3"]])
        return country_setting(conf, s, s * float(c["total_people"]), float(c[f"sum_size_{conf['size_measure']}"]))

    for iso2, ref_units in units.groupby("iso2"):
        mine = plants[plants["iso2"] == iso2].reset_index()  # 'index' = SITE_UID (row of plants.csv)
        lon, lat, people = aggregate_cells(raster_for_year(iso2, tif_dir, year), int(cfg["aggregate_cells"]))
        cells, tree = sphere_xyz(lon, lat), cKDTree(sphere_xyz(mine["lon"], mine["lat"]))
        local = pd.Series(np.arange(len(mine)), index=mine["index"])
        for measure in confs["size_measure"].unique():
            size = mine[f"size_{measure}"].to_numpy(dtype=float)
            winner, dist = weighted_nearest(tree, cells, size, int(cfg["candidate_plants"]))
            for conf in confs[confs["size_measure"] == measure].to_dict("records"):
                k, r10k = setting(conf, iso2)
                per_plant = served_by_plant(winner, dist, people, size, k, conf["radius_exponent"], r10k, max_buffer)
                for name, u in ref_units.groupby("reference"):
                    m = matches[(matches["reference"] == name) & matches["SITE_UID"].isin(local.index)]
                    pred = (
                        pd.Series(per_plant[local[m["SITE_UID"]].to_numpy()], index=m["unit"].to_numpy())
                        .groupby(level=0)
                        .sum()
                        .reindex(u["unit"])
                        .fillna(0.0)
                        .to_numpy()
                    )
                    truth = u["ref"].to_numpy(dtype=float)
                    rows.append(
                        {
                            **conf,
                            "reference": name,
                            "iso2": iso2,
                            "k": k,
                            "r10k_km": r10k,
                            "n_units": len(u),
                            "class_balanced_error": class_balanced_error(pred, truth),
                            "total_ratio": float(pred.sum() / truth.sum()),
                        }
                    )
        logger.info("%s: %d configurations checked", iso2, len(confs))

    for name, spec in cfg["district_references"].items():
        iso2 = spec["iso2"]
        try:
            loc = read_localities(source_path(cfg, spec["source"], spec["file"]), spec)
            mine = plants[plants["iso2"] == iso2].reset_index(drop=True)
            tree = cKDTree(sphere_xyz(mine["lon"], mine["lat"]))
            points = sphere_xyz(loc["lon"], loc["lat"])
            district = loc.groupby("district")[["people", "dwellings", "drainage"]].sum()
            drained = (district["drainage"] / district["dwellings"]).clip(upper=1)
            for measure in confs["size_measure"].unique():
                size = mine[f"size_{measure}"].to_numpy(dtype=float)
                winner, dist = weighted_nearest(tree, points, size, int(cfg["candidate_plants"]))
                for conf in confs[confs["size_measure"] == measure].to_dict("records"):
                    k, r10k = setting(conf, iso2)
                    radius = np.minimum(r10k * (k * size / 1e4) ** conf["radius_exponent"] * 1000.0, max_buffer)
                    served = loc["people"].where(dist <= radius[winner], 0.0).groupby(loc["district"]).sum()
                    s = served / district["people"]
                    w = district["people"] / district["people"].sum()
                    district_rows.append(
                        {
                            **conf,
                            "reference": name,
                            "iso2": iso2,
                            "n_districts": len(district),
                            "served_share": float(served.sum() / district["people"].sum()),
                            "drainage_share": float((drained * w).sum()),
                            "above_drainage_pp": float((np.maximum(s - drained, 0) * w).sum() * 100),
                            "districts_above_drainage": float(((s - drained) > 0.05).mean()),
                        }
                    )
        except Exception as err:  # a district file that cannot be read must not stop the plant check
            district_rows.append({"reference": name, "iso2": iso2, "status": f"{type(err).__name__}: {err}"[:300]})
            logger.warning("%s: %s", name, district_rows[-1]["status"])
    return pd.DataFrame(rows), pd.DataFrame(district_rows)


KEY = ["size_measure", "radius_exponent", "scale_min", "scale_max", "cap_km"]


def plant_error(rows, leave_out=None):
    """log10 error per configuration: mean over countries of the mean over their references."""
    rows = rows if leave_out is None else rows[rows["iso2"] != leave_out]
    per_country = rows.groupby(KEY + ["iso2"], dropna=False)["class_balanced_error"].mean()
    return per_country.groupby(level=list(range(len(KEY))), dropna=False).mean()


def choose(rows, calibration, national):
    """Per size measure the chosen configuration, and the same choice with each reference country left out."""
    nat = calibration[calibration["cap_km"] > 0].copy()
    columns = {"jmp": "mean_abs_error_pp"} | {c: f"{c}_mean_abs_error_pp" for c in national if c != "jmp"}
    nat["national_error_pp"] = nat[[columns[c] for c in national]].mean(axis=1)
    out = []
    for leave_out in [None, *sorted(rows["iso2"].unique())]:
        err = plant_error(rows, leave_out).rename("plant_error").reset_index()
        for measure, e in err.groupby("size_measure"):
            fixed = e[e["cap_km"].isna()]
            rules = e[e["cap_km"].notna()].merge(nat, on=KEY)
            ok = rules[rules["plant_error"] <= float(fixed["plant_error"].iloc[0])]
            pick = ok.loc[ok["national_error_pp"].idxmin()] if len(ok) else None
            out.append(
                {
                    "size_measure": measure,
                    "left_out": leave_out or "",
                    "baseline_plants_x": 10 ** float(fixed["plant_error"].iloc[0]),
                    "n_allowed": len(ok),
                    # no rule keeps the plant-level error: the fixed radius without scale stays
                    "choice": "rule" if pick is not None else "fixed radius",
                    **(
                        {
                            "scale_min": pick["scale_min"],
                            "scale_max": pick["scale_max"],
                            "cap_km": pick["cap_km"],
                            "a": pick["a"],
                            "b": pick["b"],
                            "plants_x": 10 ** float(pick["plant_error"]),
                            "national_error_pp": pick["national_error_pp"],
                            **{f"{c}_error_pp": pick[columns[c]] for c in national},
                        }
                        if pick is not None
                        else {"cap_km": np.nan}
                    ),
                }
            )
    return pd.DataFrame(out)


def run(cfg):
    out_dir = cfg["paths"]["output_dir"]

    def read(name):
        return pd.read_csv(
            os.path.join(out_dir, name),
            keep_default_na=False,
            na_values=[""],
            dtype={"uwwCode": str, "cwns": str},  # CWNS IDs keep their leading zeros
        )

    def write(frame, name):
        with atomic_output(os.path.join(out_dir, name)) as tmp:
            frame.to_csv(tmp, index=False)

    plants = read("plants.csv")
    voronoi = load_config(script_name="create_voronoi")
    exponents = {m: voronoi["radius_calibration"][m]["radius_exponent"] for m in cfg["size_measures"]}
    fixed = {m: voronoi["radius_calibration"][m]["radius_at_10k_people_km"] for m in cfg["size_measures"]}
    units, matches, status = build_references(cfg, load_config(script_name="sweep_validation"), plant_points(plants))
    write(units, "plant_references.csv")
    write(status, "plant_check_status.csv")
    calibration = read("calibration_by_cap.csv")
    rows, districts = check(
        cfg,
        plants,
        read("countries.csv"),
        read("reference_table.csv"),
        calibration,
        units,
        matches,
        exponents,
        fixed,
    )
    write(rows, "plant_check.csv")
    write(districts, "district_check.csv")
    choice = choose(rows, calibration, cfg["choice_national"])
    write(choice, "plant_choice.csv")
    print(status.to_string(index=False))
    print(choice.round(3).to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["run"])
    parser.parse_args(argv)
    run(load_config(script_name="connection_radius"))


if __name__ == "__main__":
    configure_logging()
    main()
