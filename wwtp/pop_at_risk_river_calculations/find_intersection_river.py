"""Attach nearest river-system intersection metadata to non-served polygons.

Workflow:
1. Spatially match polygons to nearby rivers within a search distance.
2. Assign each polygon a river-system ID (`MAIN_RIV`, the HydroRIVERS main stem).
3. Compute the common downstream juncture (`NXT_DIS`, the first shared downstream segment ID) from matched rivers.
4. Write enriched polygons to output.
"""

import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import box
from tqdm import tqdm

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides

from .. import provenance
from ..geo_utils import estimate_utm_epsg_for_geom
from ..utils import atomic_output, configure_logging

logger = logging.getLogger(__name__)


def build_graph(df):
    """Build HYRIV_ID -> NEXT_DOWN adjacency map for downstream traversal."""
    return dict(zip(df["HYRIV_ID"], df["NEXT_DOWN"]))


def find_intersection_id(id1, id2, graph):
    """Find first common downstream node between two river IDs."""
    visited = set()
    while id1:
        visited.add(id1)
        id1 = graph.get(id1)
    while id2:
        if id2 in visited:
            return id2
        id2 = graph.get(id2)
    return None


def find_common_intersection(ids, graph):
    """Find common downstream intersection for a list of river IDs."""
    if not ids:
        return None
    current = ids[0]
    for id_ in ids[1:]:
        current = find_intersection_id(current, id_, graph)
        if current is None:
            return None
    return current


def optimize_river_lookup(polygons_gdf, rivers_gdf, x_distance, utm_epsg):
    """Match polygons to nearby rivers with the same basin ID in one UTM zone.

    Parameters
    ----------
    polygons_gdf : geopandas.GeoDataFrame
        Non-served polygons to enrich.
    rivers_gdf : geopandas.GeoDataFrame
        River segments to match against.
    x_distance : float
        Buffer distance used for the candidate search.
    utm_epsg : int
        Projected CRS used for the lookup.

    Returns
    -------
    geopandas.GeoDataFrame
        Polygon layer with a ``river_list`` column.
    """
    # Project rivers once
    rivers_gdf = rivers_gdf.to_crs(utm_epsg)
    polygons_gdf = polygons_gdf.to_crs(utm_epsg)

    poly_temp = polygons_gdf.copy()
    poly_temp["geometry"] = polygons_gdf.geometry.buffer(x_distance)

    joined = gpd.sjoin(
        rivers_gdf[["geometry", "HYBAS_ID", "HYRIV_ID"]],
        poly_temp[["geometry", "HYBAS_ID"]],
        how="inner",
        predicate="intersects",
    )
    matched = joined[joined["HYBAS_ID_left"] == joined["HYBAS_ID_right"]]
    river_lists = matched.groupby("index_right")["HYRIV_ID"].apply(list)
    polygons_gdf["river_list"] = river_lists.reindex(polygons_gdf.index)
    polygons_gdf["river_list"] = polygons_gdf["river_list"].apply(lambda x: x if isinstance(x, list) else [])

    return polygons_gdf


def orchestrate_settlement_river_intersections(polygons_gdf, rivers_gdf, x_distance, max_workers=4):
    """Run spatial river lookup by UTM zone in parallel.

    Parameters
    ----------
    polygons_gdf : geopandas.GeoDataFrame
        Non-served polygons to enrich.
    rivers_gdf : geopandas.GeoDataFrame
        River segments used for matching.
    x_distance : float
        Buffer distance used for the candidate search.
    max_workers : int, default=4
        Maximum number of worker processes.

    Returns
    -------
    geopandas.GeoDataFrame
        Polygon layer with ``river_list`` assignments.
    """
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")

    if "utm" not in polygons_gdf.columns:
        logger.info("Estimating UTM zones")
        polygons_gdf["utm"] = polygons_gdf.geometry.apply(estimate_utm_epsg_for_geom)

    gdfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Wrap UTM zones in tqdm for progress bar
        for utm in tqdm(polygons_gdf["utm"].unique(), desc="Processing UTM zones"):
            poly_subset = polygons_gdf[polygons_gdf["utm"] == utm].copy()
            bbox_geom = box(*poly_subset.total_bounds)

            possible_idx = list(rivers_gdf.sindex.intersection(bbox_geom.bounds))
            river_subset = rivers_gdf.iloc[possible_idx]

            future = executor.submit(optimize_river_lookup, poly_subset, river_subset, x_distance, utm)
            futures[future] = utm

        for future in as_completed(futures):
            utm_zone = futures[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    gdfs.append(result.to_crs(4326))
            except Exception as err:
                logger.exception("Worker failed for UTM %s: %s", utm_zone, err)

    if gdfs:
        logger.info("Spatial matching complete for %s UTM chunks", len(gdfs))
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=4326)

    logger.warning("No river matches were found for any UTM chunk")
    empty = polygons_gdf.copy()
    empty["river_list"] = [[] for _ in range(len(empty))]
    return empty


def assign_main_riv(polygon_gdf, rivers_gdf):
    """Assign MAIN_RIV for each polygon using the first matched river ID."""
    # 1. Create the hash map (O(N) to build, O(1) to look up)
    # Using zip is the most memory-efficient way to build this in Python
    my_dict = dict(zip(rivers_gdf["HYRIV_ID"], rivers_gdf["MAIN_RIV"]))

    # 2. Use .apply with a lambda
    # We use .get(l[0]) to handle cases where the ID might be missing gracefully
    polygon_gdf["MAIN_RIV"] = polygon_gdf["river_list"].apply(
        lambda l: my_dict.get(l[0]) if (isinstance(l, list) and len(l) > 0) else None
    )
    return polygon_gdf


def assign_river_juncture(polygons_batch, rivers_batch):
    """Assign downstream junction ID (NXT_DIS) for one polygon batch."""
    if polygons_batch.empty or rivers_batch.empty:
        polygons_batch["NXT_DIS"] = None
        return polygons_batch

    graph = build_graph(rivers_batch)

    polygons_batch["NXT_DIS"] = polygons_batch["river_list"].apply(
        lambda rivs: find_common_intersection(rivs, graph) if rivs else None
    )

    return polygons_batch


def orchestrate_river_assignment(polygons_gdf, rivers_gdf, max_workers=8):
    """Compute river-juncture assignment grouped by ``MAIN_RIV`` in parallel.

    Parameters
    ----------
    polygons_gdf : geopandas.GeoDataFrame
        Polygon layer with ``river_list`` and ``MAIN_RIV`` assignments.
    rivers_gdf : geopandas.GeoDataFrame
        River segments providing downstream topology.
    max_workers : int, default=8
        Maximum number of worker processes.

    Returns
    -------
    geopandas.GeoDataFrame
        Polygon layer with the downstream-junction column ``NXT_DIS``.
    """
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")

    work_todo = polygons_gdf[polygons_gdf["river_list"].map(len) > 0].copy()
    work_done_empty = polygons_gdf[polygons_gdf["river_list"].map(len) == 0].copy()
    work_done_empty["NXT_DIS"] = None

    gdfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Wrap MAIN_RIV groups in tqdm for progress bar
        for riv, group in tqdm(work_todo.groupby("MAIN_RIV"), desc="Processing MAIN_RIV groups"):
            river_subset = rivers_gdf[rivers_gdf["MAIN_RIV"] == riv].copy()
            future = executor.submit(assign_river_juncture, group.copy(), river_subset)
            futures[future] = riv

        for future in as_completed(futures):
            riv_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    gdfs.append(result)
            except Exception as err:
                logger.exception("Error processing River System %s: %s", riv_id, err)

    if gdfs:
        final = pd.concat(gdfs + [work_done_empty], ignore_index=True)
        logger.info("Assigned NXT_DIS for %s polygons", len(final))
        return gpd.GeoDataFrame(final, crs=polygons_gdf.crs, geometry="geometry")

    return work_done_empty


def main():
    """Load inputs, perform river matching + juncture assignment, and save output.

    Parameters
    ----------
    None
        Runtime inputs are read from the configured files and optional CLI
        overrides.

    Returns
    -------
    None
        The enriched polygon layer is written to the configured output path.

    Notes
    -----
    CLI usage is ``python -m ...find_intersection_river <max_workers> [level]
    [version] [buffer] [weight_method] [weight_func]``.

    ``weight_func`` accepts ``mult``, ``add``, or ``""`` for default.
    """
    parser = argparse.ArgumentParser(description="Attach downstream river metadata to non-served polygons.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum worker processes for the river-intersection steps",
    )
    add_standard_override_arguments(parser)
    args = parser.parse_args()
    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="find_intersection_river", **overrides)
    provenance.write_run_manifest(cfg, "find_intersection_river")

    polygons_path = cfg["paths"]["non_served_above_threshold_outpath"]
    rivers_path = cfg["paths"]["rivershed_output_path"]
    output_path = cfg["paths"]["non_served_nxt_river_outpath"]
    basin_col = cfg["basin_column_name"]
    x_distance = float(cfg["x_distance"])
    max_workers = int(args.max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    logger.info("Loading data")
    polygons_gdf = gpd.read_file(polygons_path)
    river_columns = ["HYRIV_ID", "NEXT_DOWN", "MAIN_RIV", basin_col, "geometry"]
    rivers_gdf = gpd.read_file(rivers_path, columns=river_columns)
    logger.info("Loaded %s polygons and %s river segments", len(polygons_gdf), len(rivers_gdf))

    # Ensure both are in EPSG:4326
    if polygons_gdf.crs is None:
        raise ValueError("polygons_gdf has no CRS defined.")
    if rivers_gdf.crs is None:
        raise ValueError("rivers_gdf has no CRS defined.")

    polygons_gdf = polygons_gdf.to_crs(4326)
    rivers_gdf = rivers_gdf.to_crs(4326)

    if basin_col not in polygons_gdf.columns:
        raise KeyError(
            f"Configured basin column '{basin_col}' missing in polygons; available: {sorted(polygons_gdf.columns)}"
        )
    if basin_col not in rivers_gdf.columns:
        raise KeyError(
            f"Configured basin column '{basin_col}' missing in rivers; available: {sorted(rivers_gdf.columns)}"
        )

    polygons_gdf[basin_col] = pd.to_numeric(polygons_gdf[basin_col], errors="coerce")
    rivers_gdf[basin_col] = pd.to_numeric(rivers_gdf[basin_col], errors="coerce")
    polygons_before = len(polygons_gdf)
    rivers_before = len(rivers_gdf)
    polygons_gdf = polygons_gdf[np.isfinite(polygons_gdf[basin_col])].copy()
    rivers_gdf = rivers_gdf[np.isfinite(rivers_gdf[basin_col])].copy()
    logger.info(
        "Filtered invalid basin IDs on %s: polygons %s -> %s, rivers %s -> %s",
        basin_col,
        polygons_before,
        len(polygons_gdf),
        rivers_before,
        len(rivers_gdf),
    )

    # Internal river-matching code expects HYBAS_ID. Keep data config-driven by
    # projecting the configured basin key into this internal working column.
    if basin_col != "HYBAS_ID":
        polygons_gdf["HYBAS_ID"] = polygons_gdf[basin_col]
        rivers_gdf["HYBAS_ID"] = rivers_gdf[basin_col]

    polygons_gdf["HYBAS_ID"] = polygons_gdf["HYBAS_ID"].astype(np.int64)
    rivers_gdf["HYBAS_ID"] = rivers_gdf["HYBAS_ID"].astype(np.int64)

    logger.info("Running spatial river matching")
    polygons_gdf = orchestrate_settlement_river_intersections(
        polygons_gdf, rivers_gdf, x_distance, max_workers=max_workers
    )

    logger.info("Adding MAIN_RIV to polygons")
    polygons_gdf = assign_main_riv(polygons_gdf, rivers_gdf)

    logger.info("Assigning river junctures")
    polygons_gdf = orchestrate_river_assignment(polygons_gdf, rivers_gdf, max_workers=max_workers)

    logger.info("Writing output to %s", output_path)
    with atomic_output(output_path) as tmp_path:
        polygons_gdf.to_file(tmp_path, driver="GPKG", index=False)
    logger.info("Done")


if __name__ == "__main__":
    configure_logging()
    main()
