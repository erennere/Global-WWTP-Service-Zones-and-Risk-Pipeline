"""Validate the sweep at two levels, plants and countries, and choose a candidate.

A candidate is one sweep variant (``wwtp.workflow.sweep_variants``) with one service-area approach.
Approach 0 does not depend on the basin level, so it is a candidate once (at the configured level).

**Plants.** Each candidate's people per plant (``<year>_zonal_sum`` of its population layer) against:

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
``mixed``. Every metric is also given for ``all``. Measures (``SIZE_CLASSES`` of the service-radius
calibration): class-balanced error (mean over <10k, 10k-100k, >100k people of the class median
``|log10 predicted / reference|``), typical error, share within a factor 2, median bias, total ratio,
units without people (infinitely wrong), and for gb the capture (reference people inside the
candidate's service area).

**Countries and SDG regions.** Every country with at least ``national_min_plants`` plants: people in
the candidate's service areas (``<national_year>_zonal_sum``) / people of its population raster
(``national_year``), against the share of people connected to wastewater treatment:

==========  ====================================================  ============================
reference   share per country (value of the year nearest           role
            ``national_year``, at most ``national_max_year_gap``
            years off)
==========  ====================================================  ============================
jmp         JMP "safely managed via sewer" (sewered and treated)   independent, global; calibration
                                                                    data for size-based candidates
jmp_sewer   JMP sewer connection, treated or not                    upper bound
eurostat    Eurostat env_ww_con, urban and other treatment          independent (Europe)
oecd        OECD public total treatment                             independent (OECD)
==========  ====================================================  ============================

Measures: mean, median and population-weighted absolute error (percentage points), bias, share within
10 pp, the SDG-region error (regional shares, population-weighted) and the world ratio. The radius
scale of size-based candidates was calibrated on the JMP shares (leaving one region out: 10.1 pp,
notes/radius_calibration_2026-11-12.md), so for them JMP is not an independent test. Part of the gap
is common to every candidate (plants missing from our list); the differences between candidates are
what the choice uses.

**Choice.** For every component of ``decision_weights`` (plant references: class-balanced error;
national references: mean absolute error over countries), the candidate's error relative to the
configured candidate's; the score is their weighted mean (configured = 1). Paired bootstrap: in every
sample the units of each component are resampled. ``shortlist``: not worse than the best at
``confidence``. ``recommended``: the best candidate if its score is below 1 at ``confidence`` and no
component is worse than the configured candidate's at ``confidence``; otherwise the configured one.
The summary also names the best by plants only and by countries only.

Subcommands (``python -m wwtp.sensitivity_analysis_scripts.sweep_validation <command>``; settings in
config.yaml, section ``sweep_validation``)::

    references  plants, groups and reference units from the configured variant's population layer
                and the reference files; national references and people per country
                -> <output_dir>/references/
    score       every candidate -> units.parquet (people per unit), scores.csv (plants),
                national_countries.csv, national_regions.csv, national_scores.csv, choice.csv,
                summary.md, and results.zip with all of them but units.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import zipfile
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
from ..utils import atomic_output, configure_logging, get_iso_codes, reported_population_mask

logger = logging.getLogger(__name__)

PLANT_REFERENCES = ("gb", "nl", "us", "hw", "uwwtd")
NATIONAL_REFERENCES = ("jmp", "jmp_sewer", "eurostat", "oecd")
ROLES = {
    "gb": "independent",
    "nl": "independent",
    "us": "calibration",
    "hw": "input",
    "uwwtd": "input",
    "jmp": "independent",
    "jmp_sewer": "upper bound",
    "eurostat": "independent",
    "oecd": "independent",
}
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
                    # its radius scale was calibrated on the JMP shares (radius_calibration.by_connection)
                    "radius_by_connection": variant.dynamic_buffering
                    and workflow.config("create_voronoi", variant)["calculate_buffer_kwargs"]["radius_by_connection"]
                    is not None,
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


def raster_total(path):
    """People of a population raster: the sum of its positive values, block by block."""
    total = 0.0
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            a = src.read(1, window=window, masked=True).astype("float64").filled(0.0)
            total += float(a[np.isfinite(a) & (a > 0)].sum())
    return total


def _country_total(args):
    iso2, tif_dir, year = args
    try:
        raster = country_raster(iso2, tif_dir, year)
    except FileNotFoundError as err:
        logger.warning("%s: no national total (%s)", iso2, err)
        return iso2, "", np.nan
    return iso2, raster, raster_total(raster)


def _nearest_year(frame, year, max_gap):
    """Per iso3 the row whose year is nearest ``year`` (the later on a tie), at most ``max_gap`` off."""
    frame = frame.assign(_d=(frame["year"] - year).abs(), _neg=-frame["year"])
    frame = frame[frame["_d"] <= max_gap]
    return frame.sort_values(["iso3", "_d", "_neg"]).drop_duplicates("iso3").drop(columns=["_d", "_neg"])


def read_national_references(cfg):
    """Connected share (0-1) and its year per country (iso3) for every national reference, and the SDG region.

    JMP: sheet 'san', columns ``national_series`` (percent). Eurostat (SDMX TSV, 'freq,ww_tp,unit,geo'):
    ww_tp = the series, unit PC; flags after the value dropped; ``eurostat_geo_aliases`` maps its
    non-ISO codes. OECD (CSV with labels): MEASURE = the series.
    """
    paths, series = cfg["paths"], cfg["national_series"]
    year, gap = int(cfg["national_year"]), int(cfg["national_max_year_gap"])
    _, alpha_2_to_3, _, _ = get_iso_codes()
    jmp = pd.read_excel(paths["jmp"], sheet_name="san")
    tables = {}
    for ref in ("jmp", "jmp_sewer"):
        t = jmp[["iso3", "year", series[ref]]].rename(columns={series[ref]: "share"}).dropna()
        tables[ref] = t.assign(share=t["share"] / 100)
    raw = pd.read_csv(paths["eurostat"], sep="\t", dtype=str)
    dims = raw.iloc[:, 0].str.split(",", expand=True)
    raw = raw[(dims[1] == series["eurostat"]) & (dims[2] == "PC")]
    geo = dims.loc[raw.index, 3].replace(cfg["eurostat_geo_aliases"])
    rows = []
    for (_, values), code in zip(raw.iloc[:, 1:].iterrows(), geo):
        for column, value in values.items():
            number = str(value).strip().split(" ")[0]
            if number not in (":", "", "nan") and code in alpha_2_to_3:
                rows.append({"iso3": alpha_2_to_3[code], "year": int(column.strip()), "share": float(number) / 100})
    tables["eurostat"] = pd.DataFrame(rows, columns=["iso3", "year", "share"])
    oecd = pd.read_csv(paths["oecd"], low_memory=False)
    oecd = oecd[oecd["MEASURE"] == series["oecd"]][["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].dropna()
    oecd = oecd.rename(columns={"REF_AREA": "iso3", "TIME_PERIOD": "year", "OBS_VALUE": "share"})
    tables["oecd"] = oecd.assign(share=oecd["share"] / 100)
    out = jmp[["iso3", "region_sdg"]].dropna().drop_duplicates("iso3")
    for ref, t in tables.items():
        t = _nearest_year(t, year, gap).rename(columns={"share": f"{ref}_share", "year": f"{ref}_year"})
        out = out.merge(t, on="iso3", how="outer")
    return out


def build_national(cfg, plants):
    """Countries with at least ``national_min_plants`` plants: people (raster), region, references."""
    counts = plants.groupby("iso2").size()
    _, alpha_2_to_3, _, _ = get_iso_codes()
    keep = [c for c in counts.index if counts[c] >= int(cfg["national_min_plants"]) and c in alpha_2_to_3]
    jobs = [(c, cfg["paths"]["pop_tif_dir"], int(cfg["national_year"])) for c in keep]
    with ProcessPoolExecutor(max_workers=int(cfg["max_workers"])) as pool:
        totals = list(pool.map(_country_total, jobs))
    national = pd.DataFrame(totals, columns=["iso2", "raster", "people"])
    national["iso3"] = national["iso2"].map(alpha_2_to_3)
    national["n_plants"] = national["iso2"].map(counts)
    national = national.merge(read_national_references(cfg), on="iso3", how="left")
    logger.info(
        "national: %d countries, %d with a raster, %d with JMP",
        len(national),
        national["people"].notna().sum(),
        national["jmp_share"].notna().sum(),
    )
    return national


def build_references(cfg):
    """plants.csv, units.csv (reference, unit, ref, year, group), matches.csv, gb_catchments.gpkg, national.csv."""
    out = os.path.join(cfg["paths"]["output_dir"], "references")
    os.makedirs(out, exist_ok=True)
    plants = load_plants(cfg)
    national = build_national(cfg, plants)
    with atomic_output(os.path.join(out, "national.csv")) as tmp:
        national.to_csv(tmp, index=False)
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
    national = pd.read_csv(os.path.join(folder, "national.csv"), keep_default_na=False, na_values=[""])
    return units, matches, catchments, national


# --------------------------------------------------------------------------- #
# Scoring (every candidate)                                                   #
# --------------------------------------------------------------------------- #
def zonal_columns(path):
    """``{year: column}`` of the population layer's ``<year>_zonal_sum`` columns."""
    fields = pyogrio.read_info(path)["fields"]
    return {int(f.split("_")[0]): f for f in fields if re.fullmatch(r"20\d\d_zonal_sum", f)}


def candidate_people(path, units, matches, country_col, catchments, gb_raster, national_year):
    """People per reference unit in one candidate's service areas, its gb capture, and people per country."""
    columns = zonal_columns(path)
    missing = sorted((set(units["year"]) | {national_year}) - set(columns))
    if missing:
        raise KeyError(f"{path} has no zonal sums for {missing}")
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
    served = layer.groupby(country_col)[columns[national_year]].sum().rename("served").reset_index()
    served = served.rename(columns={country_col: "iso2"})
    return out, served


def _score_one(args):
    row, units, matches, country_col, catchments, gb_raster, national_year = args
    logger.info("scoring %s", row["candidate"])
    out, served = candidate_people(row["path"], units, matches, country_col, catchments, gb_raster, national_year)
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


def national_long(served, national):
    """Served and reference share per candidate, country and national reference (countries with both)."""
    # a candidate without any service area in a country serves nobody there
    grid = pd.MultiIndex.from_product([served["candidate"].unique(), national["iso2"]], names=["candidate", "iso2"])
    t = grid.to_frame(index=False).merge(served, on=["candidate", "iso2"], how="left").fillna({"served": 0.0})
    t = t.merge(national, on="iso2")
    t = t[t["people"] > 0]
    rows = []
    for ref in NATIONAL_REFERENCES:
        part = t[t[f"{ref}_share"].notna()]
        rows.append(
            pd.DataFrame(
                {
                    "candidate": part["candidate"],
                    "reference": ref,
                    "iso2": part["iso2"],
                    "iso3": part["iso3"],
                    "region_sdg": part["region_sdg"],
                    "people": part["people"],
                    "served_share": part["served"] / part["people"],
                    "ref_share": part[f"{ref}_share"],
                    "ref_year": part[f"{ref}_year"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def national_regions(long):
    """Population-weighted served and reference shares per candidate, reference, SDG region and world."""
    t = long.assign(served_people=long["served_share"] * long["people"], ref_people=long["ref_share"] * long["people"])
    keys = ["candidate", "reference"]
    regions = (
        t[t["region_sdg"].notna()]
        .groupby([*keys, "region_sdg"], as_index=False)[["people", "served_people", "ref_people"]]
        .sum()
    )
    world = t.groupby(keys, as_index=False)[["people", "served_people", "ref_people"]].sum().assign(region_sdg="World")
    out = pd.concat([regions, world], ignore_index=True)
    out["served_share"] = out["served_people"] / out["people"]
    out["ref_share"] = out["ref_people"] / out["people"]
    out["error_pp"] = (out["served_share"] - out["ref_share"]) * 100
    return out


def national_scores(long, regions, cands):
    """Measures per candidate and national reference (see the module docstring)."""
    rows = []
    for (cand, ref), part in long.groupby(["candidate", "reference"], sort=False):
        err = (part["served_share"] - part["ref_share"]).to_numpy() * 100
        w = part["people"].to_numpy()
        reg = regions[(regions["candidate"] == cand) & (regions["reference"] == ref)]
        world = reg[reg["region_sdg"] == "World"].iloc[0]
        rows.append(
            {
                "candidate": cand,
                "reference": ref,
                "role": "calibration" if ref == "jmp" and bool(cands.loc[cand, "radius_by_connection"]) else ROLES[ref],
                "n_countries": len(part),
                "mean_abs_error_pp": float(np.mean(np.abs(err))),
                "median_abs_error_pp": float(np.median(np.abs(err))),
                "people_weighted_abs_error_pp": float(np.sum(np.abs(err) * w) / np.sum(w)),
                "bias_pp": float(np.mean(err)),
                "within_10pp": float(np.mean(np.abs(err) <= 10)),
                "region_mean_abs_error_pp": float(reg.loc[reg["region_sdg"] != "World", "error_pp"].abs().mean()),
                "world_served_share": float(world["served_share"]),
                "world_ref_share": float(world["ref_share"]),
                "world_ratio": float(world["served_people"] / world["ref_people"]),
            }
        )
    return pd.DataFrame(rows)


def _class_balanced(err, cls, idx):
    """Class-balanced error (log10) of every candidate (rows of err) on the units idx."""
    c = cls[idx]
    return np.mean([np.median(err[:, idx[c == k]], axis=1) for k in np.unique(c)], axis=0)


def _components(people, units, long, names, weights):
    """Per decision component: (number of units, error of every candidate on a set of unit indices)."""
    comps = {}
    for ref in weights:
        if ref in PLANT_REFERENCES:
            u = units[units["reference"] == ref].reset_index(drop=True)
            if u.empty:
                raise ValueError(f"no {ref} units: check the reference files (decision_weights)")
            p = people[people["reference"] == ref].pivot(index="candidate", columns="unit", values="pred")
            err = _abs_log_error(p.loc[names, u["unit"]].to_numpy(), u["ref"].to_numpy()[None, :])
            cls = np.digitize(u["ref"].to_numpy(), SIZE_CLASSES)
            comps[ref] = (len(u), lambda idx, err=err, cls=cls: _class_balanced(err, cls, idx))
        else:
            part = long[long["reference"] == ref]
            if part.empty:
                raise ValueError(f"no countries with {ref}: check the national references (decision_weights)")
            shares = part.pivot(index="candidate", columns="iso2", values="served_share").loc[names]
            ref_share = part.drop_duplicates("iso2").set_index("iso2")["ref_share"].loc[shares.columns].to_numpy()
            err = np.abs(shares.to_numpy() - ref_share[None, :])
            comps[ref] = (len(ref_share), lambda idx, err=err: err[:, idx].mean(axis=1))
    return comps


def _component_label(ref):
    return f"class_balanced_error_x {ref}" if ref in PLANT_REFERENCES else f"mean_abs_error_pp {ref}"


def _display(ref, value):
    """Plant errors as factors, national errors in percentage points."""
    return 10**value if ref in PLANT_REFERENCES else value * 100


def choose(people, units, long, cands, weights, samples, seed, confidence):
    """Bootstrap the decision score; shortlist and recommended candidate (see the module docstring)."""
    names = cands.index.to_list()
    conf = names.index(cands.index[cands["configured"]][0])
    rng = np.random.default_rng(seed)
    comps = _components(people, units, long, names, weights)
    w = {k: float(weights[k]) for k in comps}
    point = {k: f(np.arange(n)) for k, (n, f) in comps.items()}
    boot = {k: np.array([f(rng.integers(0, n, n)) for _ in range(int(samples))]) for k, (n, f) in comps.items()}

    def combined(values, keys, configured_values):
        with np.errstate(invalid="ignore", divide="ignore"):
            return sum(w[k] * values[k] / configured_values[k] for k in keys) / sum(w[k] for k in keys)

    keys = list(comps)
    score = combined(point, keys, {k: point[k][conf] for k in keys})
    score_boot = combined(boot, keys, {k: boot[k][:, [conf]] for k in keys})  # samples x candidates
    best = int(np.nanargmin(score))
    with np.errstate(invalid="ignore"):
        lower = np.nanquantile(score_boot - score_boot[:, [best]], 1 - confidence, axis=0)
    better_overall = bool(np.nanquantile(score_boot[:, best] - 1, confidence) < 0)
    worse = {k: bool(np.nanquantile(boot[k][:, best] - boot[k][:, conf], 1 - confidence) > 0) for k in keys}
    recommended = best if best != conf and better_overall and not any(worse.values()) else conf
    by_kind = {}
    for kind, refs in (("plants", PLANT_REFERENCES), ("countries", NATIONAL_REFERENCES)):
        sub = [k for k in keys if k in refs]
        if sub:
            by_kind[kind] = names[int(np.nanargmin(combined(point, sub, {k: point[k][conf] for k in sub})))]
    table = cands.copy()
    table["score"] = score
    for k in keys:
        lo, hi = np.nanquantile(boot[k], [(1 - confidence) / 2, (1 + confidence) / 2], axis=0)
        table[_component_label(k)] = _display(k, point[k])
        table[f"{k} low"], table[f"{k} high"] = _display(k, lo), _display(k, hi)
    table["shortlist"] = lower <= 0
    table["best"] = [i == best for i in range(len(names))]
    table["recommended"] = [i == recommended for i in range(len(names))]
    table = table.sort_values("score").reset_index().rename(columns={"index": "candidate"})
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    decision = {"better_overall": better_overall, "worse": worse, "best_by": by_kind}
    return table, decision


def summary_markdown(choice, scores, nat_scores, regions, decision, weights, confidence):
    """A short readable summary of the choice."""
    rec = choice[choice["recommended"]].iloc[0]
    conf = choice[choice["configured"]].iloc[0]
    best = choice[choice["best"]].iloc[0]
    cols = ["rank", "candidate", "score", *[_component_label(k) for k in weights], "shortlist"]
    worse = [k for k, bad in decision["worse"].items() if bad]
    lines = [
        "# Sweep validation",
        "",
        "Score: weighted mean of each component's error relative to the configured candidate (configured = 1, "
        f"lower is better); weights {dict(weights)}; paired bootstrap {confidence:.0%}.",
        f"- best: {best['candidate']} (score {best['score']:.3f})",
        f"- configured: {conf['candidate']} (rank {conf['rank']})",
        f"- best better than configured overall: {'yes' if decision['better_overall'] else 'no'}; "
        f"worse in: {', '.join(worse) if worse else 'none'}",
        *[f"- best by {kind} only: {name}" for kind, name in decision["best_by"].items()],
        f"- **recommended: {rec['candidate']}**",
        f"- shortlist (not worse than the best): {int(choice['shortlist'].sum())} of {len(choice)} candidates",
        "",
        "## Top 10",
        "",
        markdown_table(choice.head(10)[cols]),
        "",
    ]
    pair = {rec["candidate"], conf["candidate"]}
    keep = ["candidate", "reference", "role", "group", "n", "class_balanced_error_x", "typical_error_x",
            "within_factor_2", "median_bias_x", "total_ratio", "capture"]  # fmt: skip
    nat_keep = ["candidate", "reference", "role", "n_countries", "mean_abs_error_pp", "median_abs_error_pp",
                "people_weighted_abs_error_pp", "bias_pp", "within_10pp", "region_mean_abs_error_pp",
                "world_served_share", "world_ref_share"]  # fmt: skip
    reg = regions[regions["candidate"].isin(pair) & (regions["reference"] == "jmp")]
    lines += [
        "## Recommended and configured: plants, every reference and group",
        "",
        markdown_table(scores[scores["candidate"].isin(pair)][keep]),
        "",
        "## Recommended and configured: countries",
        "",
        markdown_table(nat_scores[nat_scores["candidate"].isin(pair)][nat_keep]),
        "",
        "## Recommended and configured: SDG regions against JMP (shares of the population)",
        "",
        markdown_table(reg[["candidate", "region_sdg", "people", "served_share", "ref_share", "error_pp"]]),
        "",
    ]
    return "\n".join(lines)


def score_all(cfg):
    """Score every candidate and choose (outputs in the module docstring)."""
    out_dir = cfg["paths"]["output_dir"]
    units, matches, catchments, national = read_references(cfg)
    cands = candidates()
    absent = cands.loc[~cands["path"].map(os.path.exists), "path"].to_list()
    if absent:
        raise FileNotFoundError(f"{len(absent)} population layers missing, e.g. {absent[:3]}")
    gb_raster = country_raster("GB", cfg["paths"]["pop_tif_dir"], int(cfg["reference_years"]["gb"]))
    country_col, national_year = cfg["country_output_column"], int(cfg["national_year"])
    jobs = [(row, units, matches, country_col, catchments, gb_raster, national_year) for _, row in cands.iterrows()]
    people, served = [], []
    with ProcessPoolExecutor(max_workers=int(cfg["max_workers"])) as pool:
        for cand, out, srv in pool.map(_score_one, jobs):
            people.append(out.assign(candidate=cand))
            served.append(srv.assign(candidate=cand))
    people = pd.concat(people, ignore_index=True)
    long = national_long(pd.concat(served, ignore_index=True), national)
    regions = national_regions(long)
    cands = cands.set_index("candidate")
    scores = score_table(people, units, cands)
    nat_scores = national_scores(long, regions, cands)
    weights = cfg["decision_weights"]
    choice, decision = choose(
        people,
        units,
        long,
        cands.drop(columns="path"),
        weights,
        cfg["bootstrap_samples"],
        cfg["bootstrap_seed"],
        cfg["confidence"],
    )
    outputs = {
        "units.parquet": people,
        "scores.csv": scores,
        "national_countries.csv": long,
        "national_regions.csv": regions,
        "national_scores.csv": nat_scores,
        "choice.csv": choice,
    }
    for name, frame in outputs.items():
        with atomic_output(os.path.join(out_dir, name)) as tmp:
            if name.endswith(".parquet"):
                frame.to_parquet(tmp, index=False)
            else:
                frame.to_csv(tmp, index=False)
    with atomic_output(os.path.join(out_dir, "summary.md")) as tmp, open(tmp, "w", encoding="utf-8") as fh:
        fh.write(summary_markdown(choice, scores, nat_scores, regions, decision, weights, cfg["confidence"]))
    small = ["summary.md", *(name for name in outputs if name.endswith(".csv"))]
    with (
        atomic_output(os.path.join(out_dir, "results.zip")) as tmp,
        zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z,
    ):
        for name in small:  # everything but the people per unit: the files to send back
            z.write(os.path.join(out_dir, name), name)
    return scores, nat_scores, choice


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
