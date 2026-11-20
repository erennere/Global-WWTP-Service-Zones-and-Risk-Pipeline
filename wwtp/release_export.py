"""A population layer in the release format of HeiGIT_WWTP_LVL9_BFK0_8_LOG_MULT.gpkg (stage release_export).

Settings: config.yaml, section ``release_export``. Command: ``python -m wwtp.release_export`` with the
standard run overrides (``--size-measure footprint`` etc.); Snakemake target ``release_export``.

Reads the population layer of approach ``approach`` of the variant and writes one GeoPackage with two layers
that share their attribute columns (``columns``, in that order; ``renames`` from the layer's names):

- ``layers.areas``: one service-area polygon per plant, with ``WKT_WWTP`` (the plant point as WKT);
- ``layers.points``: one point per plant (from ``WKT_WWTP``), with ``WKT_SERVICE_AREA`` (the polygon as WKT)
  instead.

As the first release (Desktop/log/column_trim.ipynb, Sep 2026), and three changes for plant list v3:

- ``HW_WASTE_ID`` (from ``OLD_WASTE_ID``) only where it is a HydroWASTE ID (``hw_id_pattern``): the national
  register plants of v3 carry their register ID there (``AU_GA_996``, ``JP_P22_...``).
- ``SOURCE_DS`` by ``source_ds``, in order, a later rule overriding an earlier one: as before HW where
  ``HW_WASTE_ID``, UWW where ``uwwCode``, and now NAT for the national register plants.
- ``weights`` (``weights_from_size``): the Voronoi weight the run used, sqrt(plant size in people), normalised
  within the plant's basin (``HYBAS_ID``), as the format's documentation describes the column. The layer's
  own ``weights`` hold 1/n for weight_method size. The sizes are recomputed with the variant's settings, and
  the run's radii (``buffer_length``) recomputed from them must match, or the export stops.

``empty_columns`` are part of the format but not computed (2014 population: download_pop.include_2014 false).
Plants without a service area are left out; rows are sorted by ``sort_by`` and ``WASTE_ID`` renumbered 0..n-1.
"""

from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import from_wkt, to_wkt

from .pipelines import create_pop_output_paths
from .service_areas.size import calculate_buffer, plant_size_people
from .starter import add_standard_override_arguments, load_config, parse_config_overrides
from .utils import atomic_output, configure_logging

logger = logging.getLogger(__name__)


def source_labels(df, rules):
    """``SOURCE_DS`` per row: the label of the last rule a row meets (column present, or equal to ``equals``)."""
    labels = pd.Series(pd.NA, index=df.index, dtype=object)
    for rule in rules:
        values = df[rule["column"]]
        met = values.notna() if rule["equals"] is None else values.astype(str).eq(str(rule["equals"]))
        labels[met.to_numpy()] = rule["label"]
    return labels


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
    derived = {"SOURCE_DS", *cfg["empty_columns"]}
    missing = [source_of[c] for c in columns if c not in derived and source_of[c] not in gdf.columns]
    if missing:
        raise KeyError(f"the population layer has no {missing}")
    keep = (gdf.geometry.notna() & ~gdf.geometry.is_empty).to_numpy()
    gdf = gdf[keep]
    out = pd.DataFrame(index=gdf.index)
    for column in columns:
        if column in cfg["empty_columns"]:
            out[column] = np.nan
        elif column != "SOURCE_DS":
            out[column] = gdf[source_of[column]].to_numpy()
    hw = out["HW_WASTE_ID"].notna() & out["HW_WASTE_ID"].astype(str).str.fullmatch(cfg["hw_id_pattern"])
    out["HW_WASTE_ID"] = out["HW_WASTE_ID"].where(hw)
    # the rules read the export's columns (HW_WASTE_ID as cleaned) and the layer's others (origin)
    others = gdf.drop(columns=[gdf.geometry.name, *[c for c in out.columns if c in gdf.columns]])
    out["SOURCE_DS"] = source_labels(pd.concat([others, out], axis=1), cfg["source_ds"])
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
    logger.info("%d plants; SOURCE_DS %s", len(areas), areas["SOURCE_DS"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    configure_logging()
    main()
