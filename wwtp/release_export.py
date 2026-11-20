"""A population layer in the release format of HeiGIT_WWTP_LVL9_BFK0_8_LOG_MULT.gpkg (stage release_export).

Settings: config.yaml, section ``release_export``. Command: ``python -m wwtp.release_export`` with the
standard run overrides (``--size-measure footprint`` etc.); Snakemake target ``release_export``.

Reads the population layer of approach ``approach`` of the variant and writes one GeoPackage with two layers
that share their attribute columns (``columns``, in that order; ``renames`` from the layer's names):

- ``layers.areas``: one service-area polygon per plant, with ``WKT_WWTP`` (the plant point as WKT);
- ``layers.points``: one point per plant (from ``WKT_WWTP``), with ``WKT_SERVICE_AREA`` (the polygon as WKT)
  instead.

As the first release (Desktop/log/column_trim.ipynb, Sep 2026), with these changes for plant list v3:

- Every plant can be traced to the dataset its record comes from and its ID there (Eren, 20 Nov 2026):
  ``SOURCE_DS`` names the dataset and the new column ``SOURCE_ID`` the plant's ID in it (``sources``: rules
  in order, a later one overriding an earlier one, UWW over HW as in the first release): UWWTD (uwwCode),
  HydroWASTE (WASTE_ID), US EPA CWNS (CWNS ID), Canada WSER (facility ID), the national registers of
  Australia, Japan and New Zealand (register ID), Thailand (no ID kept), the detections of the location
  model (their image).
- ``HW_WASTE_ID`` is the plant's HydroWASTE ID wherever one is known (``hw_id_columns``, ``hw_id_pattern``):
  from ``OLD_WASTE_ID`` (where national register plants carry their register ID instead) or the HydroWASTE
  ID the US file records (``HW_unique_ID``).
- ``weights`` (``weights_from_size``): the Voronoi weight the run used, sqrt(plant size in people), normalised
  within the plant's basin (``HYBAS_ID``), as the format's documentation describes the column. The layer's
  own ``weights`` hold 1/n for weight_method size. The sizes are recomputed with the variant's settings, and
  the run's radii (``buffer_length``) recomputed from them must match, or the export stops.

Text that stands for a missing value (``missing_text``: '<NA>', left by an earlier stage in uwwCode, WWTP_NAME
and 184 other columns, also in the first release) is a missing value. ``empty_columns`` are part of the format but not computed (2014 population: download_pop.include_2014 false).
Plants without a service area are left out; rows are sorted by ``sort_by`` and ``WASTE_ID`` renumbered 0..n-1.
"""

from __future__ import annotations

import argparse
import logging
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import from_wkt, to_wkt

from .pipelines import create_pop_output_paths
from .service_areas.size import calculate_buffer, plant_size_people
from .starter import add_standard_override_arguments, load_config, parse_config_overrides
from .utils import atomic_output, configure_logging

logger = logging.getLogger(__name__)


def first_value(df, columns):
    """Per row the first of ``columns`` with a value, as text (NaN where none)."""
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for column in columns:
        values = df[column]
        out = out.where(out.notna(), values.where(values.notna()).astype(object))
    return out.where(out.isna(), out.astype(str))


def clean_ids(ids, width):
    """IDs as written in their dataset: integers stored as floats ('25391.0') without '.0', digits padded."""
    ids = ids.where(ids.isna(), ids.astype(str).str.replace(r"^(\d+)\.0+$", r"\1", regex=True))
    if width is not None:
        digits = ids.notna() & ids.astype(str).str.fullmatch(r"\d+")
        ids = ids.where(~digits, ids.astype(str).str.zfill(int(width)))
    return ids


def sources(df, rules):
    """``SOURCE_DS`` and ``SOURCE_ID`` per row: the last rule a row meets (every ``where`` column equal to its
    value, one of ``present`` with a value, the ID matching ``id_pattern``), the ID its first ``id_columns`` (the
    pattern's group where it has one: the dataset's own ID inside a prefixed one, JP_P22_<TreeCode>)."""
    labels = pd.Series(pd.NA, index=df.index, dtype=object)
    ids = pd.Series(pd.NA, index=df.index, dtype=object)
    for rule in rules:
        met = df[list(rule["present"])].notna().any(axis=1)
        for column, value in rule["where"].items():
            met &= df[column].astype(str).eq(str(value))
        rule_ids = clean_ids(first_value(df, rule["id_columns"]), rule["id_width"]) if rule["id_columns"] else None
        if rule["id_pattern"] is not None:
            met &= rule_ids.notna() & rule_ids.astype(str).str.fullmatch(rule["id_pattern"])
            if re.compile(rule["id_pattern"]).groups:  # the dataset's own ID inside our prefixed one
                rule_ids = rule_ids.astype(str).str.extract(f"^(?:{rule['id_pattern']})$", expand=False)
        labels[met.to_numpy()] = rule["label"]
        ids[met.to_numpy()] = rule_ids[met].to_numpy() if rule_ids is not None else pd.NA
    return labels, ids


def hw_ids(df, columns, pattern):
    """The HydroWASTE ID: the first of ``columns`` holding one (``pattern``), as written there."""
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for column in columns:
        values = df[column]
        valid = values.notna() & values.astype(str).str.fullmatch(pattern)
        out = out.where(out.notna(), values.where(valid).astype(object))
    return out


def basin_weights(sizes, basins):
    """sqrt(size) normalised to sum 1 within each basin."""
    root = pd.Series(np.sqrt(np.asarray(sizes, dtype=float)), index=basins.index)
    return (root / root.groupby(basins, dropna=False).transform("sum")).to_numpy()


def build(gdf, cfg, sizes=None):
    """The two layers (service areas, points) of the release format from a population layer.

    ``sizes``: the plant sizes in people, one per row of ``gdf`` (needed with ``weights_from_size``).
    """
    columns, renames = list(cfg["columns"]), dict(cfg["renames"])
    source_of = {**{c: c for c in columns}, **{new: old for old, new in renames.items()}}
    derived = {"SOURCE_DS", "SOURCE_ID", "HW_WASTE_ID", *cfg["empty_columns"]}
    needed = [source_of[c] for c in columns if c not in derived] + list(cfg["hw_id_columns"])
    needed += [c for rule in cfg["sources"] for c in [*rule["present"], *rule["id_columns"], *rule["where"]]]
    missing = sorted({c for c in needed if c not in gdf.columns and c not in derived})
    if missing:
        raise KeyError(f"the population layer has no {missing}")
    keep = (gdf.geometry.notna() & ~gdf.geometry.is_empty).to_numpy()
    gdf = gdf[keep].copy()
    # text written for a missing value by an earlier stage ('<NA>': all_merged_v2 has it in 186 columns)
    for column in gdf.columns.drop(gdf.geometry.name):
        if not pd.api.types.is_numeric_dtype(gdf[column]):
            gdf[column] = gdf[column].mask(gdf[column].isin(cfg["missing_text"]))
    out = pd.DataFrame(index=gdf.index)
    for column in columns:
        if column in cfg["empty_columns"]:
            out[column] = np.nan
        elif column not in derived:
            out[column] = gdf[source_of[column]].to_numpy()
    out["HW_WASTE_ID"] = hw_ids(gdf, cfg["hw_id_columns"], cfg["hw_id_pattern"]).to_numpy()
    # the rules read the export's columns (HW_WASTE_ID as found) and the layer's others (OLD_WASTE_ID, CWNS, ...)
    others = gdf.drop(columns=[gdf.geometry.name, *[c for c in out.columns if c in gdf.columns]])
    out["SOURCE_DS"], out["SOURCE_ID"] = sources(pd.concat([others, out], axis=1), cfg["sources"])
    if cfg["weights_from_size"]:
        out["weights"] = basin_weights(np.asarray(sizes, dtype=float)[keep], out["HYBAS_ID"])
    areas = gpd.GeoDataFrame(out[columns], geometry=gdf.geometry.to_numpy(), crs=gdf.crs)
    areas = areas.sort_values(list(cfg["sort_by"]), kind="stable").reset_index(drop=True)
    areas["WASTE_ID"] = np.arange(len(areas))
    points = pd.DataFrame(areas.drop(columns=[areas.geometry.name, "WKT_WWTP"]))
    points["WKT_SERVICE_AREA"] = to_wkt(areas.geometry.to_numpy())
    points = gpd.GeoDataFrame(points, geometry=from_wkt(areas["WKT_WWTP"].to_numpy()), crs=areas.crs)
    return areas, points


def plant_sizes(gdf, voronoi_cfg):
    """Plant sizes in people as the run computed them, checked against the run's radii (``buffer_length``)."""
    kwargs = voronoi_cfg["calculate_buffer_kwargs"]
    radius = calculate_buffer(gdf, None, **kwargs)
    stored = pd.to_numeric(gdf["buffer_length"], errors="coerce").to_numpy(dtype=float)
    off = ~np.isclose(radius, stored, rtol=1e-6)
    if off.any():
        raise ValueError(
            f"{int(off.sum())} plants: the radius from the settings differs from the layer's buffer_length; "
            "the layer was made with other sizes than the config's"
        )
    return plant_size_people(gdf, **kwargs)


def write(areas, points, cfg, path):
    with atomic_output(path) as tmp:
        areas.to_file(tmp, driver="GPKG", layer=cfg["layers"]["areas"], engine="pyogrio")
        points.to_file(tmp, driver="GPKG", layer=cfg["layers"]["points"], engine="pyogrio", mode="a")


def parse_args():
    parser = argparse.ArgumentParser(description="Run release_export.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="release_export", **overrides)
    source = create_pop_output_paths(cfg)["voronoi"][str(cfg["approach"])]
    target = cfg["paths"]["release_filepath"]
    logger.info("%s -> %s", source, target)
    gdf = gpd.read_file(source, engine="pyogrio")
    sizes = (
        plant_sizes(gdf, load_config(script_name="create_voronoi", **overrides)) if cfg["weights_from_size"] else None
    )
    areas, points = build(gdf, cfg, sizes)
    write(areas, points, cfg, target)
    counts = areas.groupby(areas["SOURCE_DS"].fillna("none"))["SOURCE_ID"].agg(["size", "count"])
    logger.info("%d plants; per SOURCE_DS plants and plants with a SOURCE_ID:\n%s", len(areas), counts.to_string())


if __name__ == "__main__":
    configure_logging()
    main()
