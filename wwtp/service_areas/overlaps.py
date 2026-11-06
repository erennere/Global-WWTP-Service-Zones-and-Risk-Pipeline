"""Dissolving overlapping site buffers into groups, and resolving overlaps between regions.

Part of the service-area stage (``python -m wwtp.create_voronoi``).
"""
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import shapely.affinity
import shapely
from ..geo_utils import apply_in_utm_zones, buffer_geometry
from ..utils import default_cpu_workers, ensure_output_dir_for_file

logger = logging.getLogger(__name__)


def dissolve_overlapping_geometries_fast(subdf, radius, convex=False):
    """Dissolve overlapping geometries with a spatial-index-based fast path.

    Parameters
    ----------
    subdf : geopandas.GeoDataFrame
        Input geometries with a ``some_id`` column.
    radius : float
        Buffer radius used for overlap grouping.
    convex : bool, default=False
        Whether to use bounding boxes of original geometries instead of centroid
        buffers.

    Returns
    -------
    tuple[list[set], geopandas.GeoDataFrame | None]
        Connected overlap groups and the prepared GeoDataFrame.

    Notes
    -----
    This variant replaces the original nested-loop dissolve logic with a spatial
    index query plus connected-component extraction.
    """
    if subdf is None or subdf.empty:
        return [], None

    # 1-2. Squares (or bounding boxes) built in metres, each feature in its own
    # UTM zone: one zone per country distorted large countries (Russia spans
    # 30 zones) by far more than the buffer radius.
    subdf = subdf.reset_index(drop=True)
    if subdf.crs is None:
        subdf = subdf.set_crs(4326)
    if convex:
        # Bounding box of the original geometry, in metres
        boxes = apply_in_utm_zones(subdf, lambda geoms: geoms.envelope)
    else:
        # Square of side 2 * radius around the centroid
        boxes = apply_in_utm_zones(
            subdf, lambda geoms: geoms.centroid.buffer(float(radius), cap_style='square')
        )
    subdf = subdf.set_geometry(boxes).to_crs(4326)

    # 3. Spatial Join to find overlaps (The "Sweep" replacement)
    # This finds every pair of geometries that intersect
    sindex = subdf.sindex
    left_indices, right_indices = sindex.query(subdf.geometry, predicate='intersects')

    # 4. Building the Graph (The DFS replacement)
    # We treat each row index as a node and an intersection as an edge
    g = nx.Graph()
    g.add_edges_from(zip(left_indices, right_indices))

    # 5. Extracting Connected Components
    # This identifies "islands" of overlapping shapes
    components = list(nx.connected_components(g))
    
    final_groups = []
    for component in components:
        # Map the row indices back to your 'some_id' column
        group_ids = set(subdf.iloc[list(component)]['some_id'])
        final_groups.append(group_ids)

    # 6. Final buffer(0) to ensure valid geometries as per original code
    subdf['geometry'] = subdf['geometry'].buffer(0)

    return final_groups, subdf


def orchestrate_overlaps(df, max_workers, buffers_filepath, radius, convex=False, country_col='ISO_2'):
    """
    Orchestrate parallel dissolving of overlapping geometries by country.
    
    Processes countries in parallel using ProcessPoolExecutor, applies
    dissolution to each country's geometries, and merges results.
    Caches results to file to avoid recomputation.
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input geometries with a country-grouping column.
    max_workers : int
        Maximum number of worker processes.
    buffers_filepath : str
        Cache path for dissolved geometries.
    radius : float
        Buffer radius used for overlap grouping.
    convex : bool, default=False
        Whether to use convex-style bounding boxes instead of centroid buffers.
    country_col : str, default='ISO_2'
        Column used to partition geometries into country groups for parallel
        overlap dissolution.

    Returns
    -------
    geopandas.GeoDataFrame
        Dissolved geometry groups aggregated across countries.

    Notes
    -----
    Results are cached to ``buffers_filepath`` so subsequent runs can skip the
    expensive dissolve step.
    """
    if not max_workers:
        max_workers = default_cpu_workers()
    logger.info(f"Starting parallel dissolution orchestration for {len(df)} geometries")
    if os.path.exists(buffers_filepath):
        logger.info(f"Loading cached dissolved buffers from {buffers_filepath}")
        return gpd.read_file(buffers_filepath)

    if country_col not in df.columns:
        raise KeyError(f"Country grouping column '{country_col}' not found in input dataframe.")
    
    # Overlap groups stay within one country on purpose: Voronoi regions are
    # built and clipped per country, so a neighbour's buffer across the border
    # must not enlarge a country's clipping polygon. The order is only for load
    # balancing; a fixed seed keeps runs reproducible.
    countries = np.random.default_rng(0).permutation(np.asarray(pd.unique(df[country_col]), dtype=object))
    logger.debug(f"Processing {len(countries)} countries in parallel with {max_workers} workers")
    df['some_id'] = np.arange(0, len(df))
    #subdf = df.copy()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(dissolve_overlapping_geometries_fast,
                                    df[df[country_col] == country].copy(), radius, convex
                                   ) for country in countries]
    
    final_groups = []
    dfs = []
    error_count = 0
    for future in as_completed(futures):
        if future is None:
            continue
        try:
            groups, subdf = future.result()
            if groups:
                final_groups.extend(groups)
            if subdf is not None and not subdf.empty:
                dfs.append(subdf)
        except Exception as err:
            error_count += 1
            logger.error(f'Error in parallel dissolution: {err}', exc_info=True)

    if error_count:
        # A failed country would silently lose every plant's buffer downstream.
        raise RuntimeError(f"Buffer dissolution failed for {error_count} country group(s); see the log")
    if dfs:
        dfs = pd.concat(dfs, ignore_index=True)
    else:
        logger.error("No results from parallel dissolution")
        dfs = pd.DataFrame()
    
    final_dict = {}
    for i, group in enumerate(final_groups):
        for item in group:
            final_dict[int(item)] = i

    dfs['some_id'] = dfs['some_id'].astype(int)
    dfs['group_id'] = dfs['some_id'].map(lambda x: final_dict[x])
    dissolved_buffers = dfs.dissolve(by='group_id').reset_index(drop=True)  # type: ignore[operator]
    logger.debug(f"Dissolved {len(dfs)} geometries into {len(dissolved_buffers)} groups")
    dissolved_buffers['geometry'] = dissolved_buffers['geometry'].buffer(0)
    if 'centroid' in dissolved_buffers:
        dissolved_buffers = dissolved_buffers.drop(labels='centroid', axis=1)
    try:
        ensure_output_dir_for_file(buffers_filepath)
        dissolved_buffers.to_file(buffers_filepath, driver='GPKG', index=False)
        logger.info(f"Saved dissolved buffers to {buffers_filepath}")
    except Exception as err:
        logger.warning(f"Failed to cache dissolved buffers: {err}")
    return dissolved_buffers


def resolve_polygon_overlaps(region_polygons):
    """
    Remove overlapping areas from Voronoi region polygons based on area size.
    
    For each pair of overlapping polygons, removes the intersection area from
    the smaller polygon and keeps it in the larger one. Operates on a copy of
    input geometries to avoid modifying the original GeoDataFrame.
    
    Parameters
    ----------
    region_polygons : geopandas.GeoDataFrame
        Voronoi polygons to clean.

    Returns
    -------
    numpy.ndarray
        Geometry array with overlaps removed.

    Notes
    -----
    The routine uses pairwise area comparisons and removes the shared portion
    from the smaller polygon when overlaps are detected.
    """
    logger.debug(f"Starting polygon overlap resolution for {len(region_polygons)} geometries")

    non_intersecting_polygons = region_polygons.geometry.to_numpy().copy()
    present = [k for k, g in enumerate(non_intersecting_polygons) if g is not None and not g.is_empty]
    if len(present) < 2:
        return non_intersecting_polygons

    # Candidate pairs from the original footprints: differences only shrink
    # polygons, so no pair outside this set can overlap later.
    tree = shapely.STRtree(non_intersecting_polygons[present])
    left, right = tree.query(non_intersecting_polygons[present], predicate="intersects")
    pairs = sorted({(present[i], present[j]) for i, j in zip(left, right) if present[i] < present[j]})

    for i, j in pairs:
        geom_i = non_intersecting_polygons[i]
        geom_j = non_intersecting_polygons[j]
        if geom_i is None or geom_j is None or geom_i.is_empty or geom_j.is_empty:
            continue
        if not geom_i.intersects(geom_j):
            continue
        # The shared area stays with the larger polygon and is cut from the smaller.
        if geom_i.area >= geom_j.area:
            non_intersecting_polygons[j] = buffer_geometry(geom_j.difference(geom_i))
        else:
            non_intersecting_polygons[i] = buffer_geometry(geom_i.difference(geom_j))

    logger.debug(f"Polygon overlap resolution complete: {len(pairs)} candidate pairs checked")
    return non_intersecting_polygons
