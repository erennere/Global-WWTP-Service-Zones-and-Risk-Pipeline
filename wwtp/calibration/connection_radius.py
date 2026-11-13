"""Service-radius scale from the national connection rate: served-share curves per country and their fit.

The radius rule ``radius_km = r10k * (people / 10,000) ** e`` was calibrated on the US. The best r10k
differs between countries (GB 3.0 km, US 3.75 km, NL 6.0 km; notes/service_radius_three_countries_
2026-11-11.md) with how many people are connected and how complete our plant list is. This module
tests one global formula ``r10k = exp(a + b * logit(x))``, with x the national connected share (or our
own reported coverage), fitted on some countries and checked on the others.

Subcommands (``python -m wwtp.calibration.connection_radius <command>``; settings in config.yaml,
section ``connection_radius``; Snakemake target ``calibration`` runs references, curves, calibrate,
plant_check and pack)::

    references  the national connection statistics (reference_urls), saved unparsed to
                <output_dir>/references/ with manifest.json (URL, time, bytes, sha256)
    curves      for every country with plants and a WorldPop raster of ``year``: its people, the sum of
                its plant sizes, and the people inside service areas for every size measure, radius
                exponent and r10k of the grid -> countries.csv, served_share_curves.csv; and the plants
                with their country, footprint and sizes -> plants.csv. Sizes are those before the
                national size scale (``create_voronoi.size_scale``), which ``calibrate`` derives.
    fit         the connected shares per country from the references (reference_table.csv) and the
                cross-validated fit -> fit_summary.csv, fit_countries.csv
    calibrate   what the pipeline uses, from the outputs of ``curves``:
                - the country parts of the plant size (``footprint_to_people.country_offsets`` and
                  ``unsized_people_by_country``) -> plant_size_countries.csv, next to the config values;
                - the national size scale, k = connected people / sum of plant sizes within the limits
                  of ``calibration_size_scale_limits`` ([1, 1] = none), and the radius rule
                  r10k = min(cap, exp(a + b logit(C))), a and b per pair of limits and cap, tested by
                  leaving one SDG region out -> calibration_by_cap.csv, calibration_countries.csv
                A factor k on every size of a country is a factor k ** e on its radii, so the curves
                of ``curves`` serve every k.
    pack        the tables above, those of ``wwtp.calibration.plant_check`` (every configuration on the
                real catchments and census districts, and the choice) and the results of
                ``national_data`` in one file -> calibration_results.zip

Stand-in for the service areas, as in ``wwtp.calibration.service_radius`` (approach 1 without basins):
each 1 km population cell goes to the plant of its country with the smallest distance / sqrt(people)
among its ``candidate_plants`` nearest, and counts as served when it lies inside that plant's circle
``min(r10k * (people / 10,000) ** e, max_buffer)``. Distances are chords on a sphere (well below a metre
off at 50 km), so countries of any size need no projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

from ..add_pop import find_country_tif_files
from ..geo_utils import read_located_plants
from ..service_areas.size import connected_share_logit
from ..service_areas import (
    calculate_area,
    detected_footprint,
    download_overture_maps,
    intersects_with_country_db,
    plant_size_people,
    reported_size_people,
)
from ..starter import load_config
from .service_radius import fit_country_sizes
from ..utils import atomic_output, configure_logging, get_iso_codes, requests_session_with_retries

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_008.8
YEAR_TOKEN = re.compile(r"(?<!\d)(20\d\d)(?!\d)")
QUERY_CHUNK = 1_000_000  # population cells per nearest-plant query (memory: chunk x candidates x 16 bytes)


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def radius_grid(grid):
    """The r10k values (km) of the ``radius_grid`` setting: geometric from start to stop."""
    return np.geomspace(float(grid["start"]), float(grid["stop"]), int(grid["count"]))


def sphere_xyz(lon, lat):
    """Points on a sphere of the Earth's mean radius (metres): chord distance ~ ground distance."""
    lon, lat = np.radians(np.asarray(lon, dtype=float)), np.radians(np.asarray(lat, dtype=float))
    return EARTH_RADIUS_M * np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def raster_for_year(iso2, tif_dir, year):
    """The country's WorldPop raster of ``year`` in ``tif_dir/<iso3>/`` (as add_pop finds them), or None."""
    found = find_country_tif_files([iso2], tif_dir)
    files = found[iso2] if iso2 in found else None
    if not files:
        return None
    matches = []
    for path in files:
        token = YEAR_TOKEN.search(os.path.basename(path))
        if token and int(token.group(1)) == int(year):
            matches.append(path)
    if len(matches) > 1:
        raise ValueError(f"{iso2}: several rasters for {year}: {sorted(matches)}")
    return matches[0] if matches else None


def aggregate_cells(path, factor):
    """People per block of ``factor`` x ``factor`` raster cells and the blocks' centre lon/lat.

    Nodata, NaN and negative values count as 0. The raster is read in strips, so a country's 100 m
    raster never sits in memory whole. Only blocks with people are returned.
    """
    with rasterio.open(path) as src:
        if src.crs is None or not src.crs.is_geographic:
            raise ValueError(f"{path}: expected a raster in geographic coordinates, got {src.crs}")
        height, width, transform = src.height, src.width, src.transform
        n_rows, n_cols = math.ceil(height / factor), math.ceil(width / factor)
        sums = np.zeros((n_rows, n_cols))
        strip = factor * max(1, 512 // factor)
        for row0 in range(0, height, strip):
            rows = min(strip, height - row0)
            block = src.read(1, window=Window(0, row0, width, rows), masked=True).filled(0).astype(float)
            block = np.where(np.isfinite(block) & (block > 0), block, 0.0)
            pad_rows, pad_cols = (-rows) % factor, (-width) % factor
            if pad_rows or pad_cols:
                block = np.pad(block, ((0, pad_rows), (0, pad_cols)))
            r = block.shape[0] // factor
            sums[row0 // factor : row0 // factor + r] = block.reshape(r, factor, n_cols, factor).sum(axis=(1, 3))
    col_width = np.minimum(factor, width - np.arange(n_cols) * factor)
    row_height = np.minimum(factor, height - np.arange(n_rows) * factor)
    lon = transform.c + (np.arange(n_cols) * factor + col_width / 2.0) * transform.a
    lat = transform.f + (np.arange(n_rows) * factor + row_height / 2.0) * transform.e
    r_idx, c_idx = np.nonzero(sums)
    return lon[c_idx], lat[r_idx], sums[r_idx, c_idx]


def load_plants(cfg):
    """Our plants (with a location): country, lon/lat, UWWTD code, CWNS ID, footprint and sizes per size measure.

    The country is joined from the same boundary layer, the footprint computed and the sizes taken the
    same way as in create_voronoi, but without the national size scale: ``size_<measure>`` are the sizes
    it is derived from.
    """
    paths = cfg["paths"]
    gdf = read_located_plants(paths["annotated_all_filepath"]).to_crs(4326)
    gdf = calculate_area(gdf)
    if not os.path.exists(paths["overture"]):
        download_overture_maps(paths["overture_s3_url"], paths["overture"])
    country = cfg["country_output_column"]
    if country in gdf.columns:
        gdf = gdf.drop(columns=[country])
    gdf = intersects_with_country_db(
        gdf, paths["overture"], polygon_country_col=cfg["country_boundary_column"], output_country_col=country
    )
    point = gdf.geometry.representative_point()
    out = pd.DataFrame({"iso2": gdf[country].to_numpy(), "lon": point.x.to_numpy(), "lat": point.y.to_numpy()})
    # identifiers the plant references link by (wwtp.calibration.plant_check): UWWTD code, CWNS ID
    out["uwwCode"] = gdf["uwwCode"].to_numpy()
    cwns = gdf["CWNS_2022_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)
    out["cwns"] = cwns.where(gdf["CWNS_2022_ID"].notna()).to_numpy()
    out["footprint_m2"], out["sized"] = detected_footprint(gdf)
    out["reported"] = reported_size_people(gdf, cfg["reported_size_columns"], cfg["reported_size_skip"])
    for measure in cfg["size_measures"]:
        out[f"size_{measure}"] = np.asarray(
            plant_size_people(
                gdf,
                measure,
                cfg["footprint_to_people"],
                cfg["reported_size_columns"],
                cfg["reported_size_skip"],
                country,
                {},
            ),
            dtype=float,
        )
    return out[out["iso2"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Served-share curves                                                         #
# --------------------------------------------------------------------------- #
def weighted_nearest(tree, cells, size, candidates):
    """Per cell: the plant with the smallest distance / sqrt(people) among its nearest, and that distance."""
    k = min(candidates, tree.n)
    winner = np.empty(len(cells), dtype=np.int64)
    dist = np.empty(len(cells))
    root = np.sqrt(size)
    for start in range(0, len(cells), QUERY_CHUNK):
        d, i = tree.query(cells[start : start + QUERY_CHUNK], k=k)
        d, i = d.reshape(len(d), -1), i.reshape(len(i), -1)
        best = np.argmin(d / root[i], axis=1)
        rows = np.arange(len(d))
        winner[start : start + len(d)] = i[rows, best]
        dist[start : start + len(d)] = d[rows, best]
    return winner, dist


def served_people(people, winner, dist, size, exponent, r10k_km, max_buffer_m):
    """People inside the circle of the plant they go to."""
    radius = np.minimum(r10k_km * (size / 1e4) ** exponent * 1000.0, max_buffer_m)
    return float(people[dist <= radius[winner]].sum())


def country_curves(iso2, raster, plants, settings):
    """One country's people and its curve rows (see module docstring)."""
    lon, lat, people = aggregate_cells(raster, settings["aggregate_cells"])
    cells = sphere_xyz(lon, lat)
    tree = cKDTree(sphere_xyz(plants["lon"], plants["lat"]))
    summary = {"total_people": float(people.sum()), "n_cells": int(len(people))}
    rows = []
    for measure in settings["size_measures"]:
        size = plants[f"size_{measure}"].to_numpy()
        winner, dist = weighted_nearest(tree, cells, size, settings["candidate_plants"])
        for exponent in settings["radius_exponents"]:
            for r10k in settings["grid"]:
                rows.append(
                    {
                        "iso2": iso2,
                        "size_measure": measure,
                        "radius_exponent": float(exponent),
                        "radius_at_10k_people_km": float(r10k),
                        "served_people": served_people(
                            people, winner, dist, size, exponent, r10k, settings["max_buffer"]
                        ),
                    }
                )
    return summary, rows


def compute_curves(cfg):
    """countries.csv and served_share_curves.csv for every country with plants, and plants.csv."""
    plants = load_plants(cfg)
    _, alpha_2_to_3, _, _ = get_iso_codes()
    settings = {
        "aggregate_cells": int(cfg["aggregate_cells"]),
        "candidate_plants": int(cfg["candidate_plants"]),
        "size_measures": list(cfg["size_measures"]),
        "radius_exponents": list(cfg["radius_exponents"]),
        "grid": radius_grid(cfg["radius_grid"]),
        "max_buffer": float(cfg["max_buffer"]),
    }
    countries, jobs = [], {}
    for iso2, group in plants.groupby("iso2"):
        raster = raster_for_year(iso2, cfg["paths"]["pop_tif_dir"], cfg["year"])
        row = {
            "iso2": iso2,
            "iso3": alpha_2_to_3[iso2] if iso2 in alpha_2_to_3 else "",
            "year": int(cfg["year"]),
            "raster": os.path.basename(raster) if raster else "",
            "n_plants": int(len(group)),
            "n_plants_reported": int(np.isfinite(group["reported"]).sum()),
            "reported_people": float(np.nansum(group["reported"])),
            **{f"sum_size_{m}": float(group[f"size_{m}"].sum()) for m in settings["size_measures"]},
            "status": "ok" if raster else "no raster",
        }
        countries.append(row)
        if raster:
            jobs[iso2] = (raster, group.reset_index(drop=True))
    logger.info("%d countries with plants, %d with a %s raster", len(countries), len(jobs), cfg["year"])

    by_size = sorted(jobs, key=lambda iso: -os.path.getsize(jobs[iso][0]))  # largest first
    curves, summaries = [], {}
    with ProcessPoolExecutor(max_workers=int(cfg["max_workers"])) as pool:
        futures = {pool.submit(country_curves, iso, *jobs[iso], settings): iso for iso in by_size}
        for done, future in enumerate(as_completed(futures), start=1):
            iso = futures[future]
            summary, rows = future.result()
            summaries[iso] = summary
            curves.extend(rows)
            logger.info("[%d/%d] %s: %.0f people", done, len(futures), iso, summary["total_people"])
    for row in countries:
        if row["iso2"] in summaries:
            row.update(summaries[row["iso2"]])
            row["reported_coverage"] = row["reported_people"] / row["total_people"] if row["total_people"] else np.nan
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with atomic_output(os.path.join(out_dir, "countries.csv")) as tmp:
        pd.DataFrame(countries).to_csv(tmp, index=False)
    with atomic_output(os.path.join(out_dir, "served_share_curves.csv")) as tmp:
        pd.DataFrame(curves).to_csv(tmp, index=False)
    with atomic_output(os.path.join(out_dir, "plants.csv")) as tmp:
        plants.round({"lon": 5, "lat": 5}).to_csv(tmp, index=False)
    return pd.DataFrame(countries), pd.DataFrame(curves)


# --------------------------------------------------------------------------- #
# References                                                                  #
# --------------------------------------------------------------------------- #
def download_references(cfg):
    """Save every ``reference_urls`` file unparsed; manifest.json records each download or its error."""
    folder = os.path.join(cfg["paths"]["output_dir"], "references")
    os.makedirs(folder, exist_ok=True)
    session = requests_session_with_retries()
    manifest, failed = {}, []
    for name, url in cfg["reference_urls"].items():
        try:
            response = session.get(url, timeout=int(cfg["request_timeout_seconds"]))
            response.raise_for_status()
            with atomic_output(os.path.join(folder, name)) as tmp, open(tmp, "wb") as f:
                f.write(response.content)
            manifest[name] = {
                "url": url,
                "final_url": response.url.split("?")[0],
                "bytes": len(response.content),
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            logger.info("%s: %d bytes", name, len(response.content))
        except Exception as err:  # noqa: BLE001 - every source is tried, failures are listed and fail the run
            manifest[name] = {"url": url, "error": repr(err)}
            failed.append(name)
            logger.error("%s: %s", name, err)
    with atomic_output(os.path.join(folder, "manifest.json")) as tmp, open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    if failed:
        raise RuntimeError(f"reference download failed for {failed}; see {folder}/manifest.json")
    return manifest


# --------------------------------------------------------------------------- #
# Fit                                                                         #
# --------------------------------------------------------------------------- #
def logit(share):
    """logit of a share, clipped as in the pipeline (service_areas.size.CONNECTED_SHARE_CLIP)."""
    return connected_share_logit(share)


def share_at(radii, shares, r10k):
    """Served share at ``r10k`` by interpolation in log radius (the curve is monotone)."""
    return float(np.interp(np.log(r10k), np.log(radii), shares))


def needed_radius(radii, shares, target):
    """The r10k at which the served share reaches ``target``, and whether it lies inside the grid."""
    if target <= shares[0]:
        return float(radii[0]), "below grid"
    if target >= shares[-1]:
        return float(radii[-1]), "above grid"
    # first crossing; shares are non-decreasing in the radius
    j = int(np.argmax(shares >= target))
    s0, s1 = shares[j - 1], shares[j]
    t = 0.0 if s1 == s0 else (target - s0) / (s1 - s0)
    return float(np.exp(np.log(radii[j - 1]) + t * (np.log(radii[j]) - np.log(radii[j - 1])))), "ok"


def fit_line(x, log_r):
    """Least squares log r10k = a + b x."""
    b, a = np.polyfit(np.asarray(x, dtype=float), np.asarray(log_r, dtype=float), 1)
    return float(a), float(b)


def cross_validate(table, folds, seed):
    """Out-of-sample served share per country: fit on the other folds, predict r10k from the predictor.

    ``table`` holds per country: ``x`` (logit of the predictor), ``log_r`` (log needed r10k), ``fit_ok``
    (the needed radius lies inside the grid), ``reference`` and the curve (``radii``, ``shares``).
    """
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(np.arange(len(table)) % folds)
    predicted = np.full(len(table), np.nan)
    for k in range(folds):
        train = (fold_of != k) & table["fit_ok"].to_numpy()
        if train.sum() < 3:
            raise ValueError(f"fold {k}: fewer than 3 training countries with a needed radius inside the grid")
        a, b = fit_line(table.loc[train, "x"], table.loc[train, "log_r"])
        for i in np.flatnonzero(fold_of == k):
            r = math.exp(a + b * table["x"].iloc[i])
            predicted[i] = share_at(table["radii"].iloc[i], table["shares"].iloc[i], r)
    return predicted, fold_of


def error_summary(predicted, reference):
    err = np.asarray(predicted) - np.asarray(reference)
    return {
        "n": int(np.isfinite(err).sum()),
        "mean_abs_error_pp": float(np.nanmean(np.abs(err)) * 100),
        "median_abs_error_pp": float(np.nanmedian(np.abs(err)) * 100),
        "bias_pp": float(np.nanmean(err) * 100),
        "within_5pp": float(np.nanmean(np.abs(err) <= 0.05)),
        "within_10pp": float(np.nanmean(np.abs(err) <= 0.10)),
    }


def _nearest_year(frame, key, year, max_gap):
    """Per ``key``: the row whose ``year`` column is nearest ``year`` (the later one on a tie), if at
    most ``max_gap`` years away."""
    frame = frame.assign(_d=(frame["year"] - year).abs(), _neg=-frame["year"])
    frame = frame[frame["_d"] <= max_gap]
    return frame.sort_values([key, "_d", "_neg"]).drop_duplicates(key).drop(columns=["_d", "_neg"])


def read_references(cfg):
    """Connected-to-treatment share (0-1) and its year per country (iso3) and source, from the downloads.

    Columns ``<source>_share`` and ``<source>_year`` for every source in ``reference_files`` and JMP's
    ``region_sdg``; the year is
    the one nearest ``year``, at most ``reference_max_year_gap`` years away. JMP: sheet 'san', column ``reference_series['jmp']`` (percent). Eurostat
    (SDMX TSV, 'freq,ww_tp,unit,geo'): ``ww_tp`` = the series, unit PC; flags after the value are dropped;
    ``eurostat_geo_aliases`` maps its non-ISO codes. OECD (CSV with labels): MEASURE = the series.
    SDG 6.3.1 (WHO / UN-Habitat 2025 summary workbook, ``paths.sdg631``, columns ``sdg631``): household
    wastewater volume in sewers / total household volume x share of the sewer flow delivered to
    treatment, i.e. the share of people whose sewage reaches treatment through a sewer.
    """
    folder = os.path.join(cfg["paths"]["output_dir"], "references")
    files, series, year = cfg["reference_files"], cfg["reference_series"], int(cfg["year"])
    _, alpha_2_to_3, _, _ = get_iso_codes()
    tables = []
    if "jmp" in files:
        jmp = pd.read_excel(os.path.join(folder, files["jmp"]), sheet_name="san")
        regions = jmp[["iso3", "region_sdg"]].dropna().drop_duplicates("iso3")
        jmp = jmp[["iso3", "year", series["jmp"]]].rename(columns={series["jmp"]: "share"}).dropna()
        tables.append(("jmp", jmp.assign(share=jmp["share"] / 100)))
    if "eurostat" in files:
        raw = pd.read_csv(os.path.join(folder, files["eurostat"]), sep="\t", dtype=str)
        dims = raw.iloc[:, 0].str.split(",", expand=True)
        raw = raw[(dims[1] == series["eurostat"]) & (dims[2] == "PC")]
        geo = dims.loc[raw.index, 3].replace(cfg["eurostat_geo_aliases"])
        rows = []
        for (_, values), code in zip(raw.iloc[:, 1:].iterrows(), geo):
            for column, value in values.items():
                number = str(value).strip().split(" ")[0]
                if number not in (":", "", "nan") and code in alpha_2_to_3:
                    rows.append({"iso3": alpha_2_to_3[code], "year": int(column.strip()), "share": float(number) / 100})
        tables.append(("eurostat", pd.DataFrame(rows, columns=["iso3", "year", "share"])))
    if "oecd" in files:
        oecd = pd.read_csv(os.path.join(folder, files["oecd"]), low_memory=False)
        oecd = oecd[oecd["MEASURE"] == series["oecd"]][["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].dropna()
        oecd = oecd.rename(columns={"REF_AREA": "iso3", "TIME_PERIOD": "year", "OBS_VALUE": "share"})
        tables.append(("oecd", oecd.assign(share=oecd["share"] / 100)))
    sdg = cfg["sdg631"]
    book = pd.read_excel(cfg["paths"]["sdg631"], sheet_name=sdg["sheet"])
    # people whose sewage reaches treatment through a sewer, as a share: the household volume in sewers
    # over the total household volume, times the share of the sewer flow delivered to treatment
    share = book[sdg["sewer_volume"]] / book[sdg["total_volume"]] * book[sdg["sewer_delivered_percent"]] / 100
    sdg_table = pd.DataFrame({"iso3": book[sdg["iso3"]], "year": int(sdg["year"]), "share": share})
    tables.append(("sdg631", sdg_table[np.isfinite(sdg_table["share"].astype(float))]))
    out = regions if "jmp" in files else None
    for source, table in tables:
        table = _nearest_year(table, "iso3", year, int(cfg["reference_max_year_gap"])).rename(
            columns={"share": f"{source}_share", "year": f"{source}_year"}
        )
        out = table if out is None else out.merge(table, on="iso3", how="outer")
    return out.reset_index(drop=True)


def fit_connection_radius(cfg, countries, curves, references, current_r10k):
    """Cross-validated fit per size measure, radius exponent and predictor (see module docstring).

    ``references``: ``read_references`` output; ``fit_reference`` is fitted and scored, the other
    sources are scored as checks. ``current_r10k``: {size_measure: (exponent, r10k)} of the configured
    rule, the baseline. Errors are served share minus reference, in percentage points; ``served /
    reference people`` weights countries by their people.
    """
    fit_source = cfg["fit_reference"]
    checks = [c[: -len("_share")] for c in references.columns if c.endswith("_share") and c != f"{fit_source}_share"]
    refs = references.set_index("iso3")
    countries = countries[countries["status"] == "ok"].set_index("iso3")
    summary, per_country = [], []

    def scored(model, predicted, table, extra=None):
        row = {**key, "model": model, **(extra or {}), **error_summary(predicted, table["reference"])}
        people = table["people"].to_numpy()
        row["served / reference people"] = float(
            np.sum(np.asarray(predicted) * people) / np.sum(table["reference"] * people)
        )
        for source in checks:
            has = table[f"{source}_share"].notna().to_numpy()
            if has.any():
                err = np.asarray(predicted)[has] - table.loc[has, f"{source}_share"].to_numpy()
                row[f"check {source}: n"] = int(has.sum())
                row[f"check {source}: mean_abs_error_pp"] = float(np.mean(np.abs(err)) * 100)
                row[f"check {source}: bias_pp"] = float(np.mean(err) * 100)
        summary.append(row)

    for (measure, exponent), curve in curves.groupby(["size_measure", "radius_exponent"]):
        base = []
        for iso2, c in curve.groupby("iso2"):
            info = countries[countries["iso2"] == iso2]
            if info.empty or info.index[0] not in refs.index or pd.isna(refs.loc[info.index[0], f"{fit_source}_share"]):
                continue
            iso3 = info.index[0]
            c = c.sort_values("radius_at_10k_people_km")
            radii = c["radius_at_10k_people_km"].to_numpy()
            shares = np.maximum.accumulate(c["served_people"].to_numpy() / info["total_people"].iloc[0])
            target = float(refs.loc[iso3, f"{fit_source}_share"])
            r_needed, where = needed_radius(radii, shares, target)
            base.append(
                {
                    "iso3": iso3,
                    "people": float(info["total_people"].iloc[0]),
                    "n_plants": int(info["n_plants"].iloc[0]),
                    "reference": target,
                    **{f"{s}_share": refs.loc[iso3, f"{s}_share"] for s in checks},
                    "reported_coverage": float(info["reported_coverage"].iloc[0]),
                    "radii": radii,
                    "shares": shares,
                    "log_r": math.log(r_needed),
                    "needed_r10k": r_needed,
                    "fit_ok": where == "ok",
                    "needed_where": where,
                }
            )
        if len(base) < 2 * int(cfg["fit_folds"]):
            logger.warning("%s e %s: only %d countries with a reference; skipped", measure, exponent, len(base))
            continue
        table = pd.DataFrame(base)
        key = {
            "size_measure": measure,
            "radius_exponent": float(exponent),
            "reference": fit_source,
            "n_countries": len(table),
            "n_needed_inside_grid": int(table["fit_ok"].sum()),
        }
        current = None
        if measure in current_r10k and math.isclose(current_r10k[measure][0], exponent):
            current = [share_at(r, s, current_r10k[measure][1]) for r, s in zip(table["radii"], table["shares"])]
            scored(f"current r10k {current_r10k[measure][1]} km", current, table)
        median_r = float(np.exp(np.median(table.loc[table["fit_ok"], "log_r"])))
        constant = [share_at(r, s, median_r) for r, s in zip(table["radii"], table["shares"])]
        scored(f"one r10k for all ({median_r:.2f} km, in sample)", constant, table)
        for predictor in cfg["fit_predictors"]:
            table["x"] = logit(table[predictor])
            predicted, fold_of = cross_validate(table, int(cfg["fit_folds"]), int(cfg["fit_seed"]))
            ok = table["fit_ok"].to_numpy()
            a, b = fit_line(table.loc[ok, "x"], table.loc[ok, "log_r"])
            scored(f"r10k = exp(a + b logit({predictor})), cross-validated", predicted, table, {"a": a, "b": b})
            for i, row in table.iterrows():
                per_country.append(
                    {
                        "iso3": row["iso3"],
                        "size_measure": measure,
                        "radius_exponent": float(exponent),
                        "predictor": predictor,
                        "n_plants": row["n_plants"],
                        "people": row["people"],
                        "reference": row["reference"],
                        **{f"{s}_share": row[f"{s}_share"] for s in checks},
                        "reported_coverage": row["reported_coverage"],
                        "needed_r10k": row["needed_r10k"],
                        "needed_where": row["needed_where"],
                        "served_share_max_grid": row["shares"][-1],
                        "served_share_current": current[i] if current is not None else np.nan,
                        "fold": int(fold_of[i]),
                        "predicted_r10k_all_data": math.exp(a + b * row["x"]),
                        "served_share_cross_validated": predicted[i],
                    }
                )
    return pd.DataFrame(summary), pd.DataFrame(per_country)


def size_scale(connected_people, sum_size, limits):
    """Factor on every plant size of a country: connected people / sum of its plant sizes, within limits."""
    return np.clip(np.asarray(connected_people, dtype=float) / np.asarray(sum_size, dtype=float), *limits)


def calibrate_by_region(cfg, countries, curves, references):
    """The national size scale and the capped radius rule, tested by leaving one region out.

    For every size measure, radius exponent e, pair of limits of ``calibration_size_scale_limits`` and
    cap of ``calibration_caps_km``:

    - ``k = connected people / sum of plant sizes`` of the country, within the limits (connected people
      = the ``fit_reference`` share C x the people of its raster; limits [1, 1] = no scale). With it
      every radius of the country grows by ``k ** e``: its served share at r10k is its curve at
      ``r10k * k ** e``, and the r10k it needs is the unscaled one / ``k ** e``.
    - ``r10k = min(cap, exp(a + b * logit(C)))``: a and b on all countries (needed radii above the cap
      count as the cap), and the error of the served share when each SDG region is predicted from a
      fit on the other regions. ``cap`` 0 stands for one radius for all (the best on the other
      regions), the baseline.

    Returns one row per combination and the per-country table of the fits.
    """
    source, checks = cfg["fit_reference"], cfg["check_references"]
    refs = references.dropna(subset=[f"{source}_share", "region_sdg"]).set_index("iso3")
    countries = countries[(countries["status"] == "ok") & countries["iso3"].isin(refs.index)]
    rows, detail = [], []
    for (measure, exponent), curve in curves.groupby(["size_measure", "radius_exponent"]):
        table = []
        for iso2, c in curve.groupby("iso2"):
            info = countries[countries["iso2"] == iso2]
            if info.empty:
                continue
            c = c.sort_values("radius_at_10k_people_km")
            radii = c["radius_at_10k_people_km"].to_numpy()
            shares = np.maximum.accumulate(c["served_people"].to_numpy() / info["total_people"].iloc[0])
            target = float(refs.loc[info["iso3"].iloc[0], f"{source}_share"])
            needed, where = needed_radius(radii, shares, target)
            table.append(
                {
                    "iso2": iso2,
                    "iso3": info["iso3"].iloc[0],
                    "region": refs.loc[info["iso3"].iloc[0], "region_sdg"],
                    "people": float(info["total_people"].iloc[0]),
                    "sum_size": float(info[f"sum_size_{measure}"].iloc[0]),
                    "reference": target,
                    **{f"check_{c}": float(refs.loc[info["iso3"].iloc[0], f"{c}_share"]) for c in checks},
                    "radii": radii,
                    "shares": shares,
                    "needed": needed,
                    "inside": where == "ok",
                }
            )
        t = pd.DataFrame(table)
        x, ok, share, people = (
            logit(t["reference"]),
            t["inside"].to_numpy(),
            t["reference"].to_numpy(),
            t["people"].to_numpy(),
        )
        raw_scale = share * people / t["sum_size"].to_numpy()

        def errors(pred, share=share, people=people, t=t):
            err = (pred - share) * 100
            out = {
                "mean_abs_error_pp": float(np.mean(np.abs(err))),
                "median_abs_error_pp": float(np.median(np.abs(err))),
                "people_weighted_abs_error_pp": float(np.sum(np.abs(err) * people) / people.sum()),
                "bias_pp": float(np.mean(err)),
                "within_10pp": float(np.mean(np.abs(err) <= 10)),
            }
            for c in checks:  # the same served shares against references that were not fitted
                value = t[f"check_{c}"].to_numpy()
                known = np.isfinite(value)
                check = (pred[known] - value[known]) * 100
                out[f"{c}_n"] = int(known.sum())
                out[f"{c}_mean_abs_error_pp"] = float(np.mean(np.abs(check))) if known.any() else np.nan
                out[f"{c}_bias_pp"] = float(np.mean(check)) if known.any() else np.nan
            return out

        grid = radius_grid(cfg["radius_grid"])
        for limits in cfg["calibration_size_scale_limits"]:
            scale = size_scale(share * people, t["sum_size"], (limits["min"], limits["max"]))
            factor = scale ** float(exponent)  # on every radius of the country

            def served(r10k, t=t, factor=factor):
                return np.array(
                    [
                        share_at(r, s, min(max(v * f, r[0]), r[-1]))
                        for r, s, v, f in zip(t["radii"], t["shares"], r10k, factor)
                    ]
                )

            key = {
                "size_measure": measure,
                "radius_exponent": float(exponent),
                "scale_min": float(limits["min"]),
                "scale_max": float(limits["max"]),
                "n_countries": len(t),
            }
            needed = t["needed"].to_numpy() / factor
            single = np.zeros(len(t))
            for region in t["region"].unique():
                test = (t["region"] == region).to_numpy()
                best = min(grid, key=lambda r: np.mean(np.abs(served(np.full(len(t), r)) - share)[~test]))
                single[test] = served(np.full(len(t), best))[test]
            rows.append({**key, "cap_km": 0.0, "a": np.nan, "b": np.nan, **errors(single)})
            for cap in cfg["calibration_caps_km"]:
                log_needed = np.log(np.minimum(needed, cap))
                pred = np.zeros(len(t))
                for region in t["region"].unique():
                    test = (t["region"] == region).to_numpy()
                    a, b = fit_line(x[~test & ok], log_needed[~test & ok])
                    pred[test] = served(np.minimum(np.exp(a + b * x), cap))[test]
                a, b = fit_line(x[ok], log_needed[ok])
                rows.append({**key, "cap_km": float(cap), "a": a, "b": b, **errors(pred)})
                detail.append(
                    pd.DataFrame(
                        {
                            **key,
                            "cap_km": float(cap),
                            "iso2": t["iso2"],
                            "iso3": t["iso3"],
                            "region": t["region"],
                            "reference": share,
                            **{c: t[f"check_{c}"] for c in checks},
                            "size_scale_unlimited": raw_scale,
                            "size_scale": scale,
                            "needed_r10k": needed,
                            "r10k": np.minimum(np.exp(a + b * x), cap),
                            "served_share_region_left_out": pred,
                        }
                    )
                )
    return pd.DataFrame(rows), pd.concat(detail, ignore_index=True)


def country_sizes(cfg, plants):
    """The country parts of the plant size from our plants, next to the values in config.yaml."""
    to_people = cfg["footprint_to_people"]
    table, ratio = fit_country_sizes(
        plants.rename(columns={"iso2": "country"}), to_people, int(cfg["size_country_min_plants"])
    )
    table["config_offset"] = table["country"].map(to_people["country_offsets"])
    table["config_unsized_people"] = table["country"].map(to_people["unsized_people_by_country"])
    table["unsized_ratio"] = ratio
    return table


RESULT_FILES = (
    "countries.csv",
    "served_share_curves.csv",
    "plants.csv",
    "reference_table.csv",
    "plant_size_countries.csv",
    "calibration_by_cap.csv",
    "calibration_countries.csv",
    "plant_references.csv",
    "plant_check_status.csv",
    "plant_check.csv",
    "district_check.csv",
    "plant_units.csv.gz",
    "plant_choice.csv",
    "plant_configurations.csv",
)


def pack_results(out_dir, national_zip):
    """calibration_results.zip: the calibration tables and, under national_data/, the national results."""
    with (
        atomic_output(os.path.join(out_dir, "calibration_results.zip")) as tmp,
        zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive,
    ):
        for name in RESULT_FILES:
            archive.write(os.path.join(out_dir, name), name)
        with zipfile.ZipFile(national_zip) as national:
            for member in national.infolist():
                archive.writestr(f"national_data/{member.filename}", national.read(member))


# --------------------------------------------------------------------------- #
# Command line                                                                #
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["references", "curves", "fit", "calibrate", "pack"])
    args = parser.parse_args(argv)
    cfg = load_config(script_name="connection_radius")
    out_dir = cfg["paths"]["output_dir"]

    def read(name):
        return pd.read_csv(os.path.join(out_dir, name), keep_default_na=False, na_values=[""])

    def write(frame, name):
        with atomic_output(os.path.join(out_dir, name)) as tmp:
            frame.to_csv(tmp, index=False)

    if args.command == "references":
        download_references(cfg)
    elif args.command == "curves":
        compute_curves(cfg)
    elif args.command == "calibrate":
        references = read_references(cfg)
        write(references, "reference_table.csv")
        sizes = country_sizes(cfg, read("plants.csv"))
        write(sizes, "plant_size_countries.csv")
        table, detail = calibrate_by_region(cfg, read("countries.csv"), read("served_share_curves.csv"), references)
        write(table, "calibration_by_cap.csv")
        write(detail, "calibration_countries.csv")
        print(sizes.dropna(subset=["offset", "unsized_people"], how="all").round(3).to_string(index=False))
        print(table.round(3).to_string(index=False))
    elif args.command == "pack":
        national = load_config(script_name="national_data")["paths"]["output_dir"]
        pack_results(out_dir, os.path.join(national, "results.zip"))
    else:
        references = read_references(cfg)
        write(references, "reference_table.csv")
        voronoi = load_config(script_name="create_voronoi")
        current = {
            m: (c["radius_exponent"], c["radius_at_10k_people_km"]) for m, c in voronoi["radius_calibration"].items()
        }
        summary, per_country = fit_connection_radius(
            cfg, read("countries.csv"), read("served_share_curves.csv"), references, current
        )
        write(summary, "fit_summary.csv")
        write(per_country, "fit_countries.csv")
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    configure_logging()
    main()
