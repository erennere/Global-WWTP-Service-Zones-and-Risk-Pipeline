"""Validate the sweep against people per plant from real catchments, and choose a candidate.

A candidate is one sweep variant (``wwtp.workflow.sweep_variants``) with one service-area approach.
Approach 0 does not depend on the basin level, so it is a candidate once (at the configured level).
Each candidate's people per plant (``<year>_zonal_sum`` of its population layer) is compared with:

=========  ==========================================================  ===============================
reference  people per plant (one unit = one or more of our plants)      role
=========  ==========================================================  ===============================
gb         population raster inside Hoffmann et al.'s real catchments  independent; same raster as the
           (GB); our plant: nearest to the catchment's UWWTD plant,     service areas, so only the
           or the only plant inside an unlinked catchment               geometry is tested
nl         CBS inhabitants per RWZI catchment; our plant: uwwCode       independent
           'NL' + RWZI code
us         census 2020 people (``Pop_2020``) of EPA sourced            calibration: the radius rule
           sewersheds; our plant: CWNS ID, one to one, at its polygon   was fitted on them
hw         HydroWASTE POP_SERVED where reported (QUAL_POP 1)           input when the candidate uses
uwwtd      UWWTD capacity x eu_reference_factor (p.e., not only        reported sizes (radius or
           residents); nearest UWWTD plant within ``threshold``         weights), otherwise independent
=========  ==========================================================  ===============================

Groups (the former verification split), fixed once on the configured variant's layer so that every
candidate is scored on the same plants: ``ver`` (basins with more than one plant, at least
``percent_verification`` of them with a detected footprint), ``unver`` (other basins with more than one
plant), ``single`` (the only plant of its basin). A unit whose plants fall in different groups is
``mixed``. Every metric is also given for ``all``.

Metrics per candidate, reference and group (``SIZE_CLASSES`` of the service-radius calibration):
class-balanced error (mean over the size classes <10k, 10k-100k, >100k people of the class median
``|log10 predicted / reference|``), typical error, share within a factor 2, median bias, total ratio,
units without people, and for gb the capture (reference people inside the candidate's service area).
A unit whose service area holds nobody counts as infinitely wrong.

Choice: score = mean over ``decide_on`` of the class-balanced error (log10). Paired bootstrap over the
reference units (``bootstrap_samples``). ``shortlist``: candidates not worse than the best at
``confidence``. ``recommended``: the best candidate if it beats the configured one in every
``decide_on`` reference at ``confidence``, otherwise the configured one.

Subcommands (``python -m wwtp.sensitivity_analysis_scripts.sweep_validation <command>``; settings in
config.yaml, section ``sweep_validation``)::

    references  plants, groups and reference units from the configured variant's population layer
                and the reference files -> <output_dir>/references/
    score       every candidate -> <output_dir>/units.csv.gz (people per unit), scores.csv,
                choice.csv, served_by_country.csv, summary.md
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from exactextract import exact_extract
from scipy.spatial import cKDTree

from .. import workflow
from ..add_pop import find_country_tif_files
from ..calibration.service_radius import SIZE_CLASSES, _abs_log_error
from ..data_merge.merge_seg_results import assign_to_nearest
from ..geo_utils import load_eu_reference_layer
from ..pop_validation_scripts.verification_script import find_verification_watersheds
from ..starter import load_config
from ..utils import atomic_output, configure_logging, reported_population_mask

logger = logging.getLogger(__name__)

REFERENCES = ("gb", "nl", "us", "hw", "uwwtd")
ROLES = {"gb": "independent", "nl": "independent", "us": "calibration", "hw": "input", "uwwtd": "input"}
GROUPS = ("ver", "unver", "single")
CLASS_LABELS = ("<10k", "10k-100k", ">100k")
YEAR_TOKEN = re.compile(r"(?<!\d)(20\d\d)(?!\d)")
GB_CRS = 27700
US_ALBERS = 5070
PLANT_COLUMNS = ["SITE_UID", "WKT_WWTP", "total_area", "uwwCode", "CWNS_2022_ID", "POP_SERVED", "QUAL_POP"]


# --------------------------------------------------------------------------- #
# Candidates                                                                  #
# --------------------------------------------------------------------------- #
def uses_reported_sizes(variant):
    """True when reported sizes set the variant's radius or weights (hw and uwwtd are then inputs)."""
    return variant.size_measure == "reported" and (variant.dynamic_buffering or variant.weight_method == "size")


def candidates():
    """Every sweep variant with every approach it builds; approach 0 only at the configured level."""
    default = workflow.default_variant()
    default_approach = workflow.figure_approach(default)
    rows = []
    for variant in workflow.sweep_variants():
        for approach in workflow.approaches(variant):
            if approach == "0" and variant.level != default.level:
                continue  # the same layer as at the configured level (workflow.approach0_source)
            rows.append(
                {
                    "candidate": f"{variant.tag}__a{approach}",
                    "level": variant.level if approach != "0" else "any",
                    "approach": approach,
                    "buffer": "size" if variant.dynamic_buffering else variant.buffer,
                    "size_measure": variant.size_measure if variant.dynamic_buffering else "",
                    "weight_method": variant.weight_method,
                    "weight_func": variant.weight_func or "none",
                    "uses_reported_sizes": uses_reported_sizes(variant),
                    "configured": variant == default and approach == default_approach,
                    "path": workflow.population_layer(variant, approach),
                }
            )
    table = pd.DataFrame(rows)
    if table["configured"].sum() != 1:
        raise ValueError("the configured variant and approach must be exactly one candidate of the sweep")
    return table


def configured_layer():
    default = workflow.default_variant()
    return workflow.population_layer(default, workflow.figure_approach(default))


# --------------------------------------------------------------------------- #
# References (run once)                                                       #
# --------------------------------------------------------------------------- #
def country_raster(iso2, tif_dir, year):
    """The population raster of ``iso2`` for ``year`` (as add_pop reads it)."""
    files = find_country_tif_files([iso2], tif_dir).get(iso2) or []
    match = [f for f in files if (m := YEAR_TOKEN.search(os.path.basename(f))) and int(m.group(1)) == year]
    if len(match) != 1:
        raise FileNotFoundError(f"need exactly one {year} raster for {iso2} in {tif_dir}, found {match}")
    return match[0]


def zonal_people(polygons, raster):
    """People per polygon (exactextract sum, as add_pop); a GeoSeries in any CRS."""
    if len(polygons) == 0:
        return np.zeros(0)
    with rasterio.open(raster) as src:
        crs = src.crs
    vec = gpd.GeoDataFrame(geometry=polygons.to_crs(crs).values, crs=crs)
    stats = exact_extract(raster, vec, "sum", output="pandas")
    return stats["sum"].fillna(0).clip(lower=0).to_numpy()


def markdown_table(df, digits=3):
    """A pipe table (pandas' to_markdown needs tabulate)."""

    def cell(v):
        return f"{v:.{digits}f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def read_columns(path, columns, **kwargs):
    """``columns`` of a layer; pyogrio leaves out unknown columns silently, so they are checked first."""
    missing = sorted(set(columns) - set(pyogrio.read_info(path)["fields"]))
    if missing:
        raise KeyError(f"{path} lacks the columns {missing}")
    return gpd.read_file(path, columns=list(columns), **kwargs)


def load_plants(cfg):
    """Our plants from the configured layer: SITE_UID, country, group, point, reference attributes."""
    country, basin = cfg["country_output_column"], cfg["basin_column_name"]
    df = read_columns(configured_layer(), PLANT_COLUMNS + [country, basin], ignore_geometry=True)
    if df["SITE_UID"].duplicated().any():
        raise ValueError("the configured layer has several service areas per SITE_UID")
    df = find_verification_watersheds(df, cfg["percent_verification"], watershed_col=basin)
    df["group"] = np.select([df["watersheds_chosen"], df["is_single_points"]], ["ver", "single"], "unver")
    points = gpd.GeoSeries.from_wkt(df["WKT_WWTP"], crs=4326).representative_point()
    plants = gpd.GeoDataFrame(
        {
            "SITE_UID": df["SITE_UID"].astype("int64"),
            "iso2": df[country].astype(str),
            "group": df["group"],
            "uwwCode": df["uwwCode"],
            "cwns": df["CWNS_2022_ID"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(11)
            .where(df["CWNS_2022_ID"].notna()),
            "POP_SERVED": pd.to_numeric(df["POP_SERVED"], errors="coerce"),
            "QUAL_POP": df["QUAL_POP"],
        },
        geometry=points.values,
        crs=4326,
    )
    return plants


def match_nl(cfg, plants):
    """CBS inhabitants per RWZI (Tabel 1, one row per RWZI and municipality) to our uwwCode NL + code."""
    t = pd.read_excel(cfg["paths"]["nl_cbs"], sheet_name="Tabel 1", header=3)
    t = t[t["regio_type"] == "GM"].drop_duplicates("rwzi_code")
    code = t["rwzi_code"].astype(str)
    t = t[code.str.fullmatch(r"\d+") & (code != "99999")]
    units = pd.DataFrame({"unit": "NL" + t["rwzi_code"].astype(int).astype(str).str.zfill(5), "ref": t["inwoners"]})
    matches = plants[["SITE_UID", "uwwCode"]].merge(units, left_on="uwwCode", right_on="unit")[["unit", "SITE_UID"]]
    return units[units["unit"].isin(matches["unit"])], matches


def match_gb(cfg, plants):
    """Hoffmann et al. catchments to our GB plants; reference people from the population raster.

    Linked catchments (Waterbase lookup, every UWWTD plant): our plant nearest to the UWWTD plant's
    coordinates, within ``gb_match_distance_m``. Unlinked catchments: the only plant of ours inside,
    unless it is near a UWWTD plant (it then belongs to its linked catchment). The catchments of one
    plant are merged; the unit is the plant.
    """
    paths, limit = cfg["paths"], float(cfg["gb_match_distance_m"])
    gb = plants[plants["iso2"] == "GB"].to_crs(GB_CRS)
    catch = gpd.read_file(f"/vsizip/{paths['gb_catchments']}/catchments_consolidated.shp").to_crs(GB_CRS)
    catch = catch[catch.geometry.notna()].copy()
    catch["geometry"] = catch.geometry.make_valid()
    catch = catch.dissolve(by="identifier", as_index=False)
    wb = pd.read_csv(paths["gb_waterbase"]).sort_values("year").drop_duplicates("uwwCode", keep="last")
    wb = gpd.GeoDataFrame(wb, geometry=gpd.points_from_xy(wb["uwwLongitude"], wb["uwwLatitude"]), crs=4326)
    wb = wb.to_crs(GB_CRS)
    lookup = pd.read_csv(paths["gb_waterbase_lookup"]).dropna(subset=["identifier"])
    lookup = lookup.merge(wb[["uwwCode", "geometry"]], on="uwwCode")
    xy = np.column_stack([gb.geometry.x, gb.geometry.y])
    tree = cKDTree(xy)
    lx = gpd.GeoSeries(lookup["geometry"], crs=GB_CRS)
    d, i = tree.query(np.column_stack([lx.x, lx.y]))
    plant_of = {ident: int(p) for ident, p, dist in zip(lookup["identifier"], i, d) if dist <= limit}
    near_uwwtd = cKDTree(np.column_stack([wb.geometry.x, wb.geometry.y])).query(xy)[0] <= limit
    inside = gpd.sjoin(gb.reset_index(drop=True)[["geometry"]], catch[["identifier", "geometry"]], predicate="within")
    inside = inside[~inside["identifier"].isin(set(lookup["identifier"]))]
    inside = inside[inside["identifier"].map(inside.groupby("identifier").size()) == 1]
    inside = inside[~near_uwwtd[inside.index]]
    for p, ident in zip(inside.index, inside["identifier"]):
        plant_of.setdefault(ident, int(p))
    poly = catch.set_index("identifier").geometry
    pairs = pd.DataFrame({"identifier": list(plant_of), "plant": list(plant_of.values())})
    pairs = pairs[pairs["identifier"].isin(poly.index)]
    merged = pairs.groupby("plant")["identifier"].apply(lambda ids: shapely.union_all(poly.loc[list(ids)].values))
    uid = gb["SITE_UID"].to_numpy()[merged.index.to_numpy()]
    shapes = gpd.GeoSeries(merged.values, crs=GB_CRS)
    raster = country_raster("GB", paths["pop_tif_dir"], int(cfg["reference_years"]["gb"]))
    units = pd.DataFrame({"unit": [str(u) for u in uid], "ref": zonal_people(shapes, raster)})
    matches = pd.DataFrame({"unit": units["unit"], "SITE_UID": uid})
    catchments = gpd.GeoDataFrame({"unit": units["unit"]}, geometry=shapes.values, crs=GB_CRS)
    keep = units["ref"] > 0
    return units[keep], matches[keep.to_numpy()], catchments[keep.to_numpy()]


def match_us(cfg, plants):
    """EPA sourced sewersheds (one CWNS ID, no flag) to our plant with that CWNS ID, at its polygon."""
    s = gpd.read_file(f"/vsizip/{cfg['paths']['us_sewersheds']}/Version_1_3.gdb", layer="Sewersheds")
    s = s[(s["Method"] == "Sourced") & ~s["CWNS_ID"].str.contains(";") & s["Flag"].isna() & (s["Pop_2020"] > 0)]
    j = plants[plants["cwns"].notna()].merge(s[["CWNS_ID", "Pop_2020", "geometry"]], left_on="cwns", right_on="CWNS_ID")
    j = j[j.groupby("CWNS_ID")["CWNS_ID"].transform("size") == 1]
    poly = gpd.GeoSeries(j["geometry_y"], crs=s.crs).to_crs(US_ALBERS).make_valid()
    pt = gpd.GeoSeries(j["geometry_x"], crs=4326).to_crs(US_ALBERS)
    near = poly.buffer(float(cfg["us_match_distance_m"])).contains(pt).to_numpy()
    j = j[near]
    units = pd.DataFrame({"unit": j["CWNS_ID"].to_numpy(), "ref": j["Pop_2020"].astype(float).to_numpy()})
    return units, pd.DataFrame({"unit": units["unit"], "SITE_UID": j["SITE_UID"].to_numpy()})


def match_hw(plants):
    """Plants whose HydroWASTE POP_SERVED is reported."""
    p = plants[reported_population_mask(plants) & (plants["POP_SERVED"] > 0)]
    units = pd.DataFrame({"unit": p["SITE_UID"].astype(str).to_numpy(), "ref": p["POP_SERVED"].to_numpy()})
    return units, pd.DataFrame({"unit": units["unit"], "SITE_UID": p["SITE_UID"].to_numpy()})


def match_uwwtd(cfg, plants):
    """Nearest UWWTD plant within ``threshold`` (one to one); people = capacity x eu_reference_factor."""
    ref = load_eu_reference_layer(cfg["paths"]["eu_ref_filepath"], cfg["eu_reference_factor"])
    ref = ref[ref["POP_SERVED_EU"] > 0].reset_index(drop=True)
    near = assign_to_nearest(
        plants[["SITE_UID", "geometry"]], ref[["POP_SERVED_EU", "geometry"]], cfg["threshold"], True
    )
    near = near[near["POP_SERVED_EU"].notna()]
    units = pd.DataFrame({"unit": near["SITE_UID"].astype(str).to_numpy(), "ref": near["POP_SERVED_EU"].to_numpy()})
    return units, pd.DataFrame({"unit": units["unit"], "SITE_UID": near["SITE_UID"].to_numpy()})


def build_references(cfg):
    """plants.csv, units.csv (reference, unit, ref, year, group), matches.csv, gb_catchments.gpkg."""
    out = os.path.join(cfg["paths"]["output_dir"], "references")
    os.makedirs(out, exist_ok=True)
    plants = load_plants(cfg)
    gb_units, gb_matches, catchments = match_gb(cfg, plants)
    found = {
        "gb": (gb_units, gb_matches),
        "nl": match_nl(cfg, plants),
        "us": match_us(cfg, plants),
        "hw": match_hw(plants),
        "uwwtd": match_uwwtd(cfg, plants),
    }
    group_of = plants.set_index("SITE_UID")["group"]
    units, matches = [], []
    for ref, (u, m) in found.items():
        m = m.assign(reference=ref, unit=m["unit"].astype(str))
        groups = (
            m.assign(group=m["SITE_UID"].map(group_of))
            .groupby("unit")["group"]
            .agg(lambda g: g.iloc[0] if g.nunique() == 1 else "mixed")
        )
        u = u.assign(reference=ref, unit=u["unit"].astype(str), year=int(cfg["reference_years"][ref]))
        u["group"] = u["unit"].map(groups)
        units.append(u[["reference", "unit", "ref", "year", "group"]])
        matches.append(m[["reference", "unit", "SITE_UID"]])
        logger.info("%s: %d units, %d plants, %.1f M people", ref, len(u), len(m), u["ref"].sum() / 1e6)
    with atomic_output(os.path.join(out, "plants.csv")) as tmp:
        pd.DataFrame(plants.drop(columns="geometry")).to_csv(tmp, index=False)
    with atomic_output(os.path.join(out, "units.csv")) as tmp:
        pd.concat(units, ignore_index=True).to_csv(tmp, index=False)
    with atomic_output(os.path.join(out, "matches.csv")) as tmp:
        pd.concat(matches, ignore_index=True).to_csv(tmp, index=False)
    with atomic_output(os.path.join(out, "gb_catchments.gpkg")) as tmp:
        catchments.to_file(tmp, driver="GPKG")


def read_references(cfg):
    folder = os.path.join(cfg["paths"]["output_dir"], "references")
    units = pd.read_csv(os.path.join(folder, "units.csv"), dtype={"unit": str})
    matches = pd.read_csv(os.path.join(folder, "matches.csv"), dtype={"unit": str})
    catchments = gpd.read_file(os.path.join(folder, "gb_catchments.gpkg"))
    catchments["unit"] = catchments["unit"].astype(str)
    return units, matches, catchments


# --------------------------------------------------------------------------- #
# Scoring (every candidate)                                                   #
# --------------------------------------------------------------------------- #
def zonal_columns(path):
    """``{year: column}`` of the population layer's ``<year>_zonal_sum`` columns."""
    fields = pyogrio.read_info(path)["fields"]
    return {int(f.split("_")[0]): f for f in fields if re.fullmatch(r"20\d\d_zonal_sum", f)}


def candidate_people(path, units, matches, country_col, catchments, gb_raster):
    """People per reference unit in one candidate's service areas, its gb capture, and people per country."""
    columns = zonal_columns(path)
    missing = sorted(set(units["year"]) - set(columns))
    if missing:
        raise KeyError(f"{path} has no zonal sums for {missing}")
    latest = columns[max(columns)]
    layer = read_columns(path, ["SITE_UID", country_col, *columns.values()], ignore_geometry=True)
    per_plant = layer.groupby("SITE_UID")[list(columns.values())].sum(min_count=1).fillna(0.0)
    m = matches.merge(units[["reference", "unit", "year"]], on=["reference", "unit"])
    m["pred"] = 0.0
    for year, col in columns.items():
        sel = m["year"] == year
        m.loc[sel, "pred"] = m.loc[sel, "SITE_UID"].map(per_plant[col]).fillna(0.0).to_numpy()
    pred = m.groupby(["reference", "unit"], as_index=False)["pred"].sum()
    out = units[["reference", "unit"]].merge(pred, on=["reference", "unit"], how="left").fillna({"pred": 0.0})

    # gb capture: reference people inside the candidate's service area of the matched plant
    gb_ids = matches.loc[matches["reference"] == "gb", "SITE_UID"].unique().tolist()
    capture = pd.Series(0.0, index=catchments["unit"].to_numpy())
    if gb_ids:
        areas = gpd.read_file(path, columns=["SITE_UID"], where=f"{country_col} = 'GB'")
        areas = areas[areas["SITE_UID"].isin(gb_ids) & areas.geometry.notna() & ~areas.geometry.is_empty]
        if len(areas):
            areas = areas.to_crs(GB_CRS)
            areas["geometry"] = areas.geometry.make_valid()
            areas = areas.dissolve(by="SITE_UID").reset_index()
            areas["unit"] = areas["SITE_UID"].astype(str)
            both = areas[["unit", "geometry"]].merge(
                pd.DataFrame({"unit": catchments["unit"], "catchment": catchments.geometry.values}), on="unit"
            )
            overlap = gpd.GeoSeries(shapely.intersection(both.geometry.values, both["catchment"].values), crs=GB_CRS)
            nonempty = ~overlap.is_empty.to_numpy()
            values = np.zeros(len(both))
            values[nonempty] = zonal_people(overlap[nonempty].reset_index(drop=True), gb_raster)
            capture.loc[both["unit"].to_numpy()] = values
    out["capture"] = np.where(out["reference"] == "gb", out["unit"].map(capture).fillna(0.0), np.nan)
    served = layer.groupby(country_col)[latest].sum().rename("people").reset_index()
    served = served.rename(columns={country_col: "iso2"}).assign(year=max(columns))
    return out, served


def _score_one(args):
    row, units, matches, country_col, catchments, gb_raster = args
    logger.info("scoring %s", row["candidate"])
    out, served = candidate_people(row["path"], units, matches, country_col, catchments, gb_raster)
    return row["candidate"], out, served


def metrics(pred, ref, capture=None):
    """Error measures of predicted against reference people per unit (see the module docstring)."""
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    err = _abs_log_error(pred, ref)
    signed = np.where(pred > 0, np.log10(np.maximum(pred, 1.0) / ref), -np.inf)
    cls = np.digitize(ref, SIZE_CLASSES)
    out = {
        "n": len(ref),
        "n_without_people": int((pred <= 0).sum()),
        "class_balanced_error_x": 10 ** float(np.mean([np.median(err[cls == c]) for c in np.unique(cls)])),
        "typical_error_x": 10 ** float(np.median(err)),
        "within_factor_2": float(np.mean(err <= np.log10(2))),
        "median_bias_x": 10 ** float(np.median(signed)),
        "total_ratio": float(pred.sum() / ref.sum()),
    }
    for c, label in enumerate(CLASS_LABELS):
        out[f"ratio {label}"] = float(pred[cls == c].sum() / ref[cls == c].sum()) if (cls == c).any() else np.nan
        out[f"n {label}"] = int((cls == c).sum())
    out["capture"] = float(np.sum(capture) / ref.sum()) if capture is not None else np.nan
    return out


def score_table(people, units, cands):
    """Metrics per candidate, reference and group (all, ver, unver, single)."""
    merged = people.merge(units, on=["reference", "unit"])
    rows = []
    for (cand, ref), part in merged.groupby(["candidate", "reference"], sort=False):
        uses = bool(cands.loc[cand, "uses_reported_sizes"])
        role = ROLES[ref] if ROLES[ref] != "input" else ("input" if uses else "independent")
        for group in ("all", *GROUPS):
            sel = part if group == "all" else part[part["group"] == group]
            if sel.empty:
                continue
            cap = sel["capture"].to_numpy() if ref == "gb" else None
            rows.append(
                {
                    "candidate": cand,
                    "reference": ref,
                    "role": role,
                    "group": group,
                    **metrics(sel["pred"], sel["ref"], cap),
                }
            )
    return pd.DataFrame(rows)


def _class_balanced(err, cls, idx):
    """Class-balanced error (log10) of every candidate (rows of err) on the units idx."""
    c = cls[idx]
    return np.mean([np.median(err[:, idx[c == k]], axis=1) for k in np.unique(c)], axis=0)


def choose(people, units, cands, decide_on, samples, seed, confidence):
    """Bootstrap the decision score; shortlist and recommended candidate (see the module docstring)."""
    names = cands.index.to_list()
    configured = cands.index[cands["configured"]][0]
    rng = np.random.default_rng(seed)
    per_ref, boot = {}, {}
    for ref in decide_on:
        u = units[units["reference"] == ref].reset_index(drop=True)
        if u.empty:
            raise ValueError(f"no {ref} units: check the reference files (decide_on)")
        p = people[people["reference"] == ref].pivot(index="candidate", columns="unit", values="pred")
        p = p.loc[names, u["unit"]].to_numpy()
        err = _abs_log_error(p, u["ref"].to_numpy()[None, :])
        cls = np.digitize(u["ref"].to_numpy(), SIZE_CLASSES)
        per_ref[ref] = _class_balanced(err, cls, np.arange(len(u)))
        boot[ref] = np.array([_class_balanced(err, cls, rng.integers(0, len(u), len(u))) for _ in range(int(samples))])
    score = np.mean([per_ref[r] for r in decide_on], axis=0)
    score_boot = np.mean([boot[r] for r in decide_on], axis=0)  # samples x candidates
    best = int(np.argmin(score))
    with np.errstate(invalid="ignore"):
        diff_best = score_boot - score_boot[:, [best]]
    lower = np.nanquantile(diff_best, 1 - confidence, axis=0)
    table = cands.copy()
    table["score_x"] = 10**score
    for ref in decide_on:
        table[f"class_balanced_error_x {ref}"] = 10 ** per_ref[ref]
        lo, hi = np.nanquantile(boot[ref], [(1 - confidence) / 2, (1 + confidence) / 2], axis=0)
        table[f"{ref} low"], table[f"{ref} high"] = 10**lo, 10**hi
    table["shortlist"] = lower <= 0
    conf = names.index(configured)
    better = {ref: bool(np.nanquantile(boot[ref][:, best] - boot[ref][:, conf], confidence) < 0) for ref in decide_on}
    recommended = names[best] if best == conf or all(better.values()) else configured
    table["best"] = [i == best for i in range(len(names))]
    table["recommended"] = [n == recommended for n in names]
    table = table.sort_values("score_x").reset_index().rename(columns={"index": "candidate"})
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    return table, better


def summary_markdown(choice, scores, better, decide_on, confidence):
    """A short readable summary of the choice."""
    top = choice.head(10)
    cols = ["rank", "candidate", "score_x", *[f"class_balanced_error_x {r}" for r in decide_on], "shortlist"]
    rec = choice[choice["recommended"]].iloc[0]
    conf = choice[choice["configured"]].iloc[0]
    best = choice[choice["best"]].iloc[0]
    lines = [
        "# Sweep validation",
        "",
        f"Decision references: {', '.join(decide_on)} (class-balanced error, lower is better; "
        f"bootstrap {confidence:.0%}).",
        f"- best: {best['candidate']} (x{best['score_x']:.3f})",
        f"- configured: {conf['candidate']} (x{conf['score_x']:.3f}, rank {conf['rank']})",
        "- best better than configured in: " + ", ".join(f"{r} {'yes' if ok else 'no'}" for r, ok in better.items()),
        f"- **recommended: {rec['candidate']}**",
        f"- shortlist (not worse than the best): {int(choice['shortlist'].sum())} of {len(choice)} candidates",
        "",
        "## Top 10",
        "",
        markdown_table(top[cols]),
        "",
        "## Recommended and configured, every reference and group",
        "",
    ]
    keep = [
        "candidate",
        "reference",
        "role",
        "group",
        "n",
        "class_balanced_error_x",
        "typical_error_x",
        "within_factor_2",
        "median_bias_x",
        "total_ratio",
        "capture",
    ]
    sel = scores[scores["candidate"].isin({rec["candidate"], conf["candidate"]})][keep]
    lines += [markdown_table(sel), ""]
    return "\n".join(lines)


def score_all(cfg):
    """Score every candidate and choose (outputs in the module docstring)."""
    out_dir = cfg["paths"]["output_dir"]
    units, matches, catchments = read_references(cfg)
    cands = candidates()
    absent = cands.loc[~cands["path"].map(os.path.exists), "path"].to_list()
    if absent:
        raise FileNotFoundError(f"{len(absent)} population layers missing, e.g. {absent[:3]}")
    gb_raster = country_raster("GB", cfg["paths"]["pop_tif_dir"], int(cfg["reference_years"]["gb"]))
    country_col = cfg["country_output_column"]
    jobs = [(row, units, matches, country_col, catchments, gb_raster) for _, row in cands.iterrows()]
    people, served = [], []
    with ProcessPoolExecutor(max_workers=int(cfg["max_workers"])) as pool:
        for cand, out, srv in pool.map(_score_one, jobs):
            people.append(out.assign(candidate=cand))
            served.append(srv.assign(candidate=cand))
    people = pd.concat(people, ignore_index=True)
    cands = cands.set_index("candidate")
    scores = score_table(people, units, cands)
    choice, better = choose(
        people,
        units,
        cands.drop(columns="path"),
        cfg["decide_on"],
        cfg["bootstrap_samples"],
        cfg["bootstrap_seed"],
        cfg["confidence"],
    )
    with atomic_output(os.path.join(out_dir, "units.parquet")) as tmp:
        people.to_parquet(tmp, index=False)
    with atomic_output(os.path.join(out_dir, "served_by_country.csv")) as tmp:
        pd.concat(served, ignore_index=True).to_csv(tmp, index=False)
    with atomic_output(os.path.join(out_dir, "scores.csv")) as tmp:
        scores.to_csv(tmp, index=False)
    with atomic_output(os.path.join(out_dir, "summary.md")) as tmp, open(tmp, "w", encoding="utf-8") as fh:
        fh.write(summary_markdown(choice, scores, better, cfg["decide_on"], cfg["confidence"]))
    with atomic_output(os.path.join(out_dir, "choice.csv")) as tmp:
        choice.to_csv(tmp, index=False)
    return scores, choice


# --------------------------------------------------------------------------- #
# Command line                                                                #
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["references", "score"])
    args = parser.parse_args(argv)
    cfg = load_config(script_name="sweep_validation")
    if args.command == "references":
        build_references(cfg)
    else:
        score_all(cfg)


if __name__ == "__main__":
    configure_logging()
    main()
