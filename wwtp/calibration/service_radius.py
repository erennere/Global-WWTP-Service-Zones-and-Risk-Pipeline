"""Calibrate plant size in people and the service radius against census population.

Two steps, both from published data matched to our own plants. They produce
``footprint_to_people`` and ``radius_calibration`` in ``create_voronoi``
(config.yaml).

1. **People per footprint** (``footprint_to_people``). For every plant with a
   detected footprint F (tank circles + rectangles, no lagoon) and a reported
   size S (``reported_size_columns`` without the skipped values, e.g.
   HydroWASTE ``QUAL_POP`` 4) fit ``log10 S = intercept + slope log10 F``.
   ``unsized_people`` is the median reported size of lagoon plants and plants
   without detections. Country parts (``fit_country_sizes``; run with the
   pipeline's country join by ``wwtp.calibration.connection_radius calibrate``):
   ``country_offsets``, the median of ``log10 S - (intercept + slope log10 F)``
   in countries with enough such plants, and ``unsized_people_by_country``, the
   median reported size of a country's size-free plants or, without enough of
   them, a fixed share of the median size of its plants with a footprint.

2. **Radius** (``radius_calibration``). Census 2020 block-group population
   centroids (USpopcenters, ``block_group2020.rda``) in states where almost
   every plant has a sourced EPA sewershed, so that people outside them are
   unsewered. Each centroid goes to the plant with the smallest distance /
   sqrt(people) among all our plants nearby and stays with it inside the
   plant's circle ``radius_at_10k_people_km * (people / 10000) ** radius_exponent``
   - the pipeline's rule without basins. Evaluation plants: our plants matched
   one-to-one to a sourced sewershed by CWNS ID, plant within 500 m of it.
   For each size measure the grid pair with the smallest mean, over the size
   classes <10k, 10k-100k and >100k people, of the class median
   ``|log10 predicted / census|`` is chosen (large plants count as much as
   small ones; a plant whose area holds nobody counts as a failure), among the
   pairs whose total population is within ``TOTAL_TOLERANCE`` of the census
   total, so that served and unserved totals stay unbiased. A
   leave-one-state-out run reports the out-of-sample error.

The GB catchments are not used: they stay the independent test.

Usage::

    python -m wwtp.calibration.service_radius \\
        --plants data/all_merged_v2.gpkg \\
        --epa-gdb data/validation_data/usa_epa_sewersheds/Version_1_3.gdb \\
        --blockgroups block_group2020.rda --country US --out notes/service_radius_calibration.json
"""

from __future__ import annotations

import argparse
import itertools
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from ..service_areas import calculate_area, plant_size_people, reported_size_people
from ..starter import load_config

US_ALBERS = 5070
DEFAULT_STATES = {"MA": "25", "CT": "09", "RI": "44", "VT": "50", "NH": "33", "UT": "49", "KY": "21"}
EXPONENTS = (0.3, 0.35, 0.4, 0.45, 0.5)
RADII_AT_10K_KM = tuple(np.round(np.arange(2.0, 6.01, 0.25), 2))
SIZE_CLASSES = (1e4, 1e5)  # <10k, 10k-100k, >100k people
CANDIDATE_PLANTS = 64  # nearest plants considered per population point
TOTAL_TOLERANCE = 0.1  # allowed |total predicted / census - 1|


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def load_plants(path):
    """Our plants with lon, lat, footprint_m2, lagoon flag and CWNS ID.

    Accepts the merged GeoPackage (footprint computed exactly as the pipeline
    does) or a CSV with ``lon``, ``lat`` and ``footprint_m2`` columns.
    """
    if path.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(path, low_memory=False, dtype={"CWNS_2022_ID": str})
    else:
        gdf = gpd.read_file(path).to_crs(4326)
        gdf = calculate_area(gdf)
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        pts = gdf.geometry.representative_point()
        df["lon"], df["lat"] = pts.x.values, pts.y.values
    df = df[df["lon"].notna() & df["lat"].notna()].reset_index(drop=True)
    df["footprint_m2"] = pd.to_numeric(df["footprint_m2"], errors="coerce").fillna(0.0)
    lagoon = df["lagoon_detected"] if "lagoon_detected" in df else pd.Series(False, index=df.index)
    df["lagoon"] = lagoon.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})
    df["sized"] = (df["footprint_m2"] > 0) & ~df["lagoon"]
    cwns = df["CWNS_2022_ID"] if "CWNS_2022_ID" in df else pd.Series(np.nan, index=df.index)
    df["cwns"] = cwns.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11).where(cwns.notna())
    return df


def us_matches(plants, gdb, states):
    """Our plants matched one-to-one to EPA sourced sewersheds in ``states``."""
    s = gpd.read_file(gdb, layer="Sewersheds")
    s = s[
        (s["Method"] == "Sourced") & s["STATE_CODE"].isin(states) & ~s["CWNS_ID"].str.contains(";") & s["Flag"].isna()
    ]
    j = plants.reset_index().merge(s[["CWNS_ID", "STATE_CODE", "geometry"]], left_on="cwns", right_on="CWNS_ID")
    j = j[j.groupby("CWNS_ID")["CWNS_ID"].transform("size") == 1].reset_index(drop=True)
    poly = gpd.GeoSeries(j["geometry"], crs=4326).to_crs(US_ALBERS).make_valid()
    pt = gpd.GeoSeries(gpd.points_from_xy(j["lon"], j["lat"]), crs=4326).to_crs(US_ALBERS)
    near = poly.buffer(500).contains(pt).to_numpy()  # plant at its catchment
    return pd.DataFrame({"plant": j["index"], "state": j["STATE_CODE"], "poly": poly.values})[near].reset_index(
        drop=True
    )


# --------------------------------------------------------------------------- #
# Step 1: people per footprint                                                #
# --------------------------------------------------------------------------- #
def fit_footprint_to_people(plants, reported_size_columns, reported_size_skip):
    """``log10 people = intercept + slope log10 footprint`` and the size-free people."""
    reported = reported_size_people(plants, reported_size_columns, reported_size_skip)
    has = np.isfinite(reported)
    fit = plants["sized"].to_numpy() & has
    x, y = np.log10(plants.loc[fit, "footprint_m2"].to_numpy()), np.log10(reported[fit])
    slope, intercept = np.polyfit(x, y, 1)
    unsized = ~plants["sized"].to_numpy() & has
    return {
        "intercept": round(float(intercept), 3),
        "slope": round(float(slope), 3),
        "unsized_people": round(float(np.median(reported[unsized]))),
        "r2": float(np.corrcoef(x, y)[0, 1] ** 2),
        "n_fit": int(fit.sum()),
        "n_unsized": int(unsized.sum()),
    }


def fit_country_sizes(plants, to_people, min_plants):
    """Country parts of the plant size: ``country_offsets`` and ``unsized_people_by_country``.

    Parameters
    ----------
    plants : pandas.DataFrame
        One row per plant: ``country``, ``footprint_m2``, ``sized`` (footprint usable, see
        ``service_areas.detected_footprint``) and ``reported`` (people; NaN where none).
    to_people : dict
        ``intercept`` and ``slope`` of the fit over all countries.
    min_plants : int
        Fewest plants a country value rests on.

    Returns
    -------
    (pandas.DataFrame, float)
        Per country: ``n_plants``; ``n_fit`` and ``offset``, the median of ``log10 reported -
        (intercept + slope * log10 footprint)`` over its sized plants with a reported size (NaN below
        ``min_plants``); ``n_sized`` and ``sized_median``, the median footprint estimate (with the
        offset) of all its sized plants (NaN below ``min_plants``); ``n_unsized_reported`` and
        ``unsized_median``, the median reported size of its size-free plants (NaN below
        ``min_plants``); ``unsized_people``, that median or else ``ratio * sized_median``, and
        ``unsized_from`` ('own', 'ratio' or ''). ``ratio`` is the median, over the countries with both,
        of ``unsized_median / sized_median``.
    """
    sized = plants["sized"].to_numpy(dtype=bool)
    reported = plants["reported"].to_numpy(dtype=float)
    has = np.isfinite(reported) & (reported > 0)
    country = plants["country"]
    footprint = plants["footprint_m2"].to_numpy(dtype=float)
    estimate = np.full(len(plants), np.nan)
    estimate[sized] = float(to_people["intercept"]) + float(to_people["slope"]) * np.log10(footprint[sized])
    residual = pd.Series(np.log10(np.where(has, reported, 1.0)) - estimate, index=plants.index)

    def medians(values, mask):
        grouped = values[mask].groupby(country[mask]).agg(["size", "median"])
        return grouped["size"], grouped["median"].where(grouped["size"] >= min_plants)

    n_fit, offset = medians(residual, sized & has)
    with_offset = pd.Series(estimate, index=plants.index) + country.map(offset).fillna(0.0)
    n_sized, sized_median = medians(np.power(10.0, with_offset), sized)
    n_unsized, unsized_median = medians(pd.Series(reported, index=plants.index), ~sized & has)
    table = pd.DataFrame({"n_plants": country.groupby(country).size()})
    table["n_fit"], table["offset"] = n_fit, offset
    table["n_sized"], table["sized_median"] = n_sized, sized_median
    table["n_unsized_reported"], table["unsized_median"] = n_unsized, unsized_median
    for column in ("n_fit", "n_sized", "n_unsized_reported"):
        table[column] = table[column].fillna(0).astype(int)
    ratio = float((table["unsized_median"] / table["sized_median"]).median())
    table["unsized_people"] = table["unsized_median"].fillna(ratio * table["sized_median"])
    table["unsized_from"] = np.where(
        table["unsized_median"].notna(), "own", np.where(table["unsized_people"].notna(), "ratio", "")
    )
    return table.rename_axis("country").reset_index(), ratio


# --------------------------------------------------------------------------- #
# Step 2: radius                                                              #
# --------------------------------------------------------------------------- #
def _abs_log_error(pred, truth):
    """|log10 pred / truth|; a plant whose area holds nobody counts as infinitely wrong."""
    return np.where(pred > 0, np.abs(np.log10(np.maximum(pred, 1.0) / truth)), np.inf)


def class_balanced_error(pred, truth):
    """Mean over the size classes of the class median |log10 pred / truth|."""
    cls = np.digitize(truth, SIZE_CLASSES)
    err = _abs_log_error(pred, truth)
    return float(np.mean([np.median(err[cls == c]) for c in np.unique(cls)]))


def summary(pred, truth):
    cls = np.digitize(truth, SIZE_CLASSES)
    err = _abs_log_error(pred, truth)
    out = {
        "class_balanced_error_x": 10 ** class_balanced_error(pred, truth),
        "typical_error_x": 10 ** float(np.median(err)),
        "within_factor_2": float(np.mean(err <= np.log10(2))),
        "total_ratio": float(pred.sum() / truth.sum()),
        "n": int(len(truth)),
    }
    for c, label in enumerate(("<10k", "10k-100k", ">100k")):
        if (cls == c).any():
            out[f"ratio {label}"] = float(pred[cls == c].sum() / truth[cls == c].sum())
    return out


def choose_radius(plants, size_people, matches, blockgroups, max_buffer_m, states=DEFAULT_STATES):
    """Pick (radius_exponent, radius_at_10k_people_km) on census population; leave-one-state-out."""
    import pyreadr

    bg = pyreadr.read_r(blockgroups)["block_group2020"]
    bg = bg[bg["STATEFP"].isin(states.values()) & (bg["POPULATION"] > 0)]
    pop = gpd.GeoSeries(gpd.points_from_xy(bg["LONGITUDE"], bg["LATITUDE"]), crs=4326).to_crs(US_ALBERS)
    popv = bg["POPULATION"].to_numpy(dtype=float)
    pxy = np.column_stack([pop.x, pop.y])

    pts = gpd.GeoSeries(gpd.points_from_xy(plants["lon"], plants["lat"]), crs=4326).to_crs(US_ALBERS)
    lo, hi = pxy.min(0) - 100_000, pxy.max(0) + 100_000
    keep = np.flatnonzero(pts.x.between(lo[0], hi[0]).to_numpy() & pts.y.between(lo[1], hi[1]).to_numpy())
    plxy = np.column_stack([pts.x.values[keep], pts.y.values[keep]])
    position = {p: i for i, p in enumerate(keep)}
    cand_d, cand_i = cKDTree(plxy).query(pxy, k=min(CANDIDATE_PLANTS, len(plxy)))
    people = size_people[keep]
    rows = np.arange(len(pxy))

    tree = shapely.STRtree(pop.values)
    ev = matches[matches["plant"].isin(position)].reset_index(drop=True)
    truth = np.array([popv[tree.query(g, predicate="contains")].sum() for g in ev["poly"]])
    ev, truth = ev[truth > 0].reset_index(drop=True), truth[truth > 0]
    ev_idx = np.array([position[p] for p in ev["plant"]])
    state = ev["state"].to_numpy()

    k = np.argmin(cand_d / np.sqrt(people)[cand_i], axis=1)
    winner, dist = cand_i[rows, k], cand_d[rows, k]
    preds = {}
    for e, r10 in itertools.product(EXPONENTS, RADII_AT_10K_KM):
        radius = np.minimum(r10 * (people / 1e4) ** e * 1000.0, max_buffer_m)
        owner = np.where(dist <= radius[winner], winner, -1)
        m = owner >= 0
        preds[(e, float(r10))] = np.bincount(owner[m], weights=popv[m], minlength=len(plxy))[ev_idx]

    def best(mask):
        unbiased = [key for key in preds if abs(preds[key][mask].sum() / truth[mask].sum() - 1) <= TOTAL_TOLERANCE]
        if not unbiased:
            raise RuntimeError("no radius on the grid keeps the total population within TOTAL_TOLERANCE")
        return min(unbiased, key=lambda key: class_balanced_error(preds[key][mask], truth[mask]))

    every = np.ones(len(ev), bool)
    chosen = best(every)
    folds = []
    for st in sorted(set(state)):
        train = state != st
        key = best(train)
        folds.append(
            {
                "held_out": st,
                "radius_exponent": key[0],
                "radius_at_10k_people_km": key[1],
                **summary(preds[key][~train], truth[~train]),
            }
        )
    by_exponent = {}
    for e in EXPONENTS:
        keys = [k for k in preds if k[0] == e and abs(preds[k].sum() / truth.sum() - 1) <= TOTAL_TOLERANCE]
        if keys:
            key = min(keys, key=lambda k: class_balanced_error(preds[k], truth))
            by_exponent[e] = {"radius_at_10k_people_km": key[1], **summary(preds[key], truth)}
    return {
        "states": sorted(states),
        "n_plants": int(len(ev)),
        "people_in_sewersheds": float(truth.sum()),
        "radius_exponent": chosen[0],
        "radius_at_10k_people_km": chosen[1],
        "in_sample": summary(preds[chosen], truth),
        "best_per_exponent": by_exponent,
        "leave_one_state_out": folds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plants", required=True, help="Merged plant GeoPackage or CSV with lon/lat/footprint_m2")
    parser.add_argument("--epa-gdb", required=True, help="EPA National Sewershed Dataset Version_1_3.gdb")
    parser.add_argument("--blockgroups", required=True, help="USpopcenters block_group2020.rda")
    parser.add_argument(
        "--country", required=True, help="ISO 3166 alpha-2 of the calibration states' country; its sizes apply"
    )
    parser.add_argument("--out", default=None, help="Write the full result as JSON")
    args = parser.parse_args()

    cfg = load_config(script_name="create_voronoi")
    plants = load_plants(args.plants)
    to_people = fit_footprint_to_people(plants, cfg["reported_size_columns"], cfg["reported_size_skip"])
    print(
        f"people per footprint: log10 S = {to_people['intercept']} + {to_people['slope']} log10 F "
        f"(n {to_people['n_fit']}, R2 {to_people['r2']:.2f}); size-free plants {to_people['unsized_people']} people "
        f"(n {to_people['n_unsized']})"
    )
    matches = us_matches(plants, args.epa_gdb, DEFAULT_STATES)
    result = {"footprint_to_people": to_people, "radius_calibration": {}}
    # The states lie in one country, so its country values of the size apply to every plant that can
    # take people from them; the national size scale comes after this step (connection_radius).
    country_column = cfg["country_output_column"]
    plants[country_column] = args.country
    sizing = {
        **to_people,
        "country_offsets": cfg["footprint_to_people"]["country_offsets"],
        "unsized_people_by_country": cfg["footprint_to_people"]["unsized_people_by_country"],
    }
    for measure in ("reported", "footprint"):
        size = plant_size_people(
            plants, measure, sizing, cfg["reported_size_columns"], cfg["reported_size_skip"], country_column, {}
        )
        res = choose_radius(plants, size, matches, args.blockgroups, cfg["max_buffer"])
        result["radius_calibration"][measure] = res
        print(
            f"\n== {measure}: radius_exponent {res['radius_exponent']}, radius_at_10k_people_km "
            f"{res['radius_at_10k_people_km']} ({res['n_plants']} plants)"
        )
        print(pd.DataFrame(res["best_per_exponent"]).T.round(3).to_string())
        print(pd.DataFrame(res["leave_one_state_out"]).round(3).to_string(index=False))
    print("\nconfig.yaml (create_voronoi):")
    print(
        f"  footprint_to_people: {{intercept: {to_people['intercept']}, slope: {to_people['slope']}, "
        f"unsized_people: {to_people['unsized_people']}}}"
    )
    print("  radius_calibration:")
    for measure, res in result["radius_calibration"].items():
        print(
            f"    {measure}: {{radius_exponent: {res['radius_exponent']}, "
            f"radius_at_10k_people_km: {res['radius_at_10k_people_km']}}}"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1, default=float)


if __name__ == "__main__":
    main()
