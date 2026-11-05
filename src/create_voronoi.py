"""
Plant Capacity Spatial Data Science Module

Comprehensive geospatial utilities for WWTP (Wastewater Treatment Plant) 
capacity analysis, including coordinate transformations, Voronoi diagrams, 
spatial clustering, and watershed integration.

Key Features:
  - Geometry validation and topology fixing (zero-buffer algorithm)
  - Spatial clustering with Union-Find optimization
  - UTM coordinate transformation and projection estimation
  - Weighted Voronoi diagram generation with multiple distance metrics
  - DuckDB-based spatial indexing for watershed and country integration
  - Buffer dissolution and polygon overlap management
  - Multi-process orchestration for large-scale spatial analysis

Dependencies:
  - geopandas, shapely: Spatial geometry and operations
  - duckdb: SQL-based spatial queries
  - scipy: Spatial indexing and distance metrics
  - pyproj: Coordinate reference system transformations
  
Organization:
  The module is organized into 9 functional sections:
    1. Geometry Validation & Manipulation
    2. Coordinate Transformation & Projection
    3. Spatial Clustering
    4. Grid & Distance Utilities
    5. Data Processing & Normalization
    6. DuckDB & External Data Integration
    7. Buffer & Geometry Dissolution
    8. Voronoi Computation & Orchestration
    9. Configuration & Main Execution
"""

import argparse
import os, logging, sys
from typing import Any, cast
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm 
from rasterio.features import shapes

from scipy.spatial import cKDTree  # type: ignore[attr-defined]
from skimage.measure import find_contours
import cv2

from shapely import Point, Polygon, LineString, MultiPolygon, MultiLineString, box, from_wkt, to_wkt, vectorized
from shapely.ops import unary_union
from shapely.geometry import shape
import shapely.affinity
import shapely

try:
    from .geo_utils import (
        UnionFind,
        cluster_point_indices,
        estimate_utm_epsg,
        estimate_utm_epsg_for_geom,
        estimate_utm_crs,
        parse_diameters_to_round_area,
        ensure_duckdb_spatial,
        buffer_geometry,
        safe_to_wkt,
        nearest_neighbor_distances,
        apply_in_utm_zones,
    )
    from .utils import configure_logging, default_cpu_workers, duckdb_connection, ensure_output_dir_for_file, quote_sql_identifier
except ImportError:  # Support running/importing as a top-level module
    from geo_utils import (
        UnionFind,
        cluster_point_indices,
        estimate_utm_epsg,
        estimate_utm_epsg_for_geom,
        estimate_utm_crs,
        parse_diameters_to_round_area,
        ensure_duckdb_spatial,
        buffer_geometry,
        safe_to_wkt,
        nearest_neighbor_distances,
        apply_in_utm_zones,
    )
    from utils import configure_logging, default_cpu_workers, duckdb_connection, ensure_output_dir_for_file, quote_sql_identifier

# Configure module-level logging
_LOG_LEVEL_NAME = os.environ.get(
    "WWTP_SERVICE_PIPELINE_LOG_LEVEL",
    "INFO",
).upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
logger = logging.getLogger(__name__)


def geometry_contains_points(geometry, points):
    """Return a boolean mask for points contained in geometry.

    Uses shapely vectorized predicates when available to avoid creating
    one Point object per grid coordinate.
    """
    if points is None or len(points) == 0:
        return np.array([], dtype=bool)

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    try:
        return np.asarray(shapely.contains_xy(geometry, x_coords, y_coords), dtype=bool)
    except Exception:
        try:  # type: ignore[attr-defined]
            return np.asarray(vectorized.contains(geometry, x_coords, y_coords), dtype=bool)
        except Exception:
            return np.fromiter(
                (geometry.contains(Point(x, y)) for x, y in points),
                dtype=bool,
                count=len(points),
            )


################################################################################
# SECTION 1: GEOMETRY VALIDATION & MANIPULATION
################################################################################

def normalize_plane(a, b):
    """Normalize 2-D coordinates by one isotropic scale (the larger extent).

    Both axes are shifted by their minimum and divided by the SAME scalar,
    the larger of the x and y extents, so Euclidean distances keep their
    shape: the result is ``(points - min) / max(width, height)``. Scaling the
    axes separately (the previous behaviour) stretched the shorter axis and
    moved Voronoi boundaries whenever the grid was not square.

    Parameters
    ----------
    a : numpy.ndarray
        Array of shape ``(n, 2)`` with ``(x, y)`` coordinates.
    b : tuple or array-like
        Single ``(x, y)`` coordinate pair.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(a_normalized, b_normalized)``; the longer axis spans ``[0, 1]``.
        A degenerate extent of 0 is treated as 1.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    all_points = np.vstack([a, b.reshape(1, 2)])
    min_vals = np.min(all_points, axis=0)
    extent = float(np.max(np.max(all_points, axis=0) - min_vals))
    scale = extent if extent > 0 else 1.0
    return (a - min_vals) / scale, (b - min_vals) / scale

def is_valid_geom(geom):
    """Return ``True`` if a geometry is non-``None``, topologically valid, and has finite coordinates.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry or None
        Geometry to validate.

    Returns
    -------
    bool
        ``True`` when all checks pass, ``False`` otherwise.
    """
    try:
        if geom is None:
            logger.debug("Geometry is None")
            return False
        if not geom.is_valid:
            logger.debug(f"Invalid geometry topology: {geom.geom_type}")
            return False
        # Only Point/LineString expose .coords, so the previous hasattr check
        # skipped this validation entirely for the Polygon/MultiPolygon geometries
        # this pipeline actually handles. get_coordinates covers every type.
        coords = shapely.get_coordinates(geom)
        if coords.size and not np.isfinite(coords).all():
            logger.debug(f"Non-finite coordinates in {geom.geom_type}")
            return False
        return True
    except Exception as e:
        logger.debug(f"Exception during geometry validation: {e}")
        return False
    
def drop_duplicates(df, col):
    """Remove duplicate rows by column while keeping all ``NaN`` rows.

    Parameters
    ----------
    df : pandas.DataFrame or None
        Input dataframe.
    col : str
        Column to deduplicate on.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with non-``NaN`` duplicates removed; ``NaN`` rows are
        always preserved.
    """
    if df is not None and not df.empty:
        nans = df[df[col].isna()]
        uniques = df[df[col].notna()].drop_duplicates(subset=[col], keep='first')
        df = pd.concat([uniques, nans], ignore_index=True)
    return df

def create_centroid_points(geom):
    """Extract a representative point from a geometry.

    Returns ``Point`` geometries as-is; returns ``centroid`` for polygons and
    lines; returns ``None`` for unsupported or invalid geometries.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry or None
        Geometry to extract a centroid from.

    Returns
    -------
    shapely.geometry.Point or None
        Valid centroid or original point, or ``None`` when unavailable.
    """
    if pd.isna(geom):
        logger.debug("Geometry is NaN")
        return None
    if isinstance(geom, Point):
        return geom
    elif isinstance(geom, (Polygon, LineString, MultiLineString, MultiPolygon)):
        centroid = geom.centroid
        if centroid.is_valid and not centroid.is_empty:
            return centroid
        else:
            logger.debug(f"Invalid centroid for {geom.geom_type} geometry")
            return None
    else:
        logger.debug(f"Unsupported geometry type for centroid: {type(geom).__name__}")
        return None

################################################################################
# SECTION 2: COORDINATE TRANSFORMATION & PROJECTION
################################################################################
    
################################################################################
# SECTION 3: SPATIAL CLUSTERING
################################################################################

def cluster_points(df, threshold):
    """
    Cluster nearby points and assign cluster IDs to DataFrame.
    
    Groups geometries that are within threshold distance using Union-Find
    algorithm after computing pairwise centroid distances. Assigns cluster
    ID to each point based on membership.
    
    Args:
        df (pd.GeoDataFrame): Points with 'geometry' column
        threshold (float): Maximum distance for clustering in same CRS units
        
    Returns:
        pd.GeoDataFrame: One row per cluster: the most complete record, with
        ``weights`` summed, ``POP_SERVED`` the cluster maximum (never the sum)
        and ``n_merged_records`` the number of records merged.

    Notes:
        Uses centroid-to-centroid distance for clustering
        Union-Find ensures transitive closure (A~B and B~C → A~C)
    """

    df = df.copy()
    df['num_missing'] = df.isnull().sum(axis=1)

    cluster_sets = cluster_point_indices(df['geometry'], threshold)
    data = []

    for cluster_set in cluster_sets:
        sub_df = df.iloc[list(cluster_set)]
        if len(sub_df) == 1:
            data.append(sub_df.assign(n_merged_records=1))
        else:
            weights = sub_df['weights'].sum()
            # Get the row(s) with the fewest NaNs
            min_missing = sub_df['num_missing'].min()
            best_rows = sub_df[sub_df['num_missing'] == min_missing]
            best_row = best_rows.iloc[0].copy()  # Choose the first one if there's a tie
            best_row['weights'] = weights
            if 'POP_SERVED' in sub_df.columns:
                # Records this close are the same plant reported by several
                # sources; adding their populations double-counts it.
                best_row['POP_SERVED'] = pd.to_numeric(sub_df['POP_SERVED'], errors='coerce').max()
            best_row['n_merged_records'] = len(sub_df)
            data.append(pd.DataFrame([best_row]))
    df = pd.concat(data, ignore_index=True)
    df = df.drop(columns=['num_missing'])
    return df
        
################################################################################
# SECTION 4: GRID & DISTANCE UTILITIES
################################################################################

def create_ranges(x, y, step, min_step=100):
    """Grid coordinates from ``min(x, y)`` in steps of exactly ``step``.

    The last coordinate is the first one at or beyond ``max(x, y)``, so the
    grid covers the whole range and the spacing is exactly ``step``. Contour
    extraction scales pixel indices by ``step``; a grid whose real spacing
    differed from it (the previous ``linspace``) shifted and stretched every
    extracted polygon.

    Args:
        x, y (float): Boundary coordinates
        step (float): Grid spacing in CRS units (metres)
        min_step (float): Kept for call compatibility; unused.

    Returns:
        np.ndarray: ``min + step * [0, 1, ..., n]``
    """
    min_val = min(x, y)
    max_val = max(x, y)
    n_range = max_val - min_val
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if n_range == 0:
        return np.array([x, y])
    n_steps = max(1, int(np.ceil(n_range / step - 1e-9)))
    return min_val + step * np.arange(n_steps + 1, dtype=float)

def auto_weight_scale(points):
    """
    Compute automatic weight scaling factor based on median nearest neighbor distance.
    
    Calculates the median distance between each point and its nearest neighbor.
    This is used to normalize weights in distance-based weighting functions.
    
    Args:
        points (list or np.ndarray): Array of shape (n, 2) with point coordinates
        
    Returns:
        float: Median of minimum inter-point distances
        
    Notes:
        Used for weight scaling in additive distance functions.
    """
    points = [(x, y) for x, y in points if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    # nanmedian, not nanmean: the name, the docstring and every caller's use of
    # the value all say median, and the O(n^2) pdist/squareform matrix this
    # replaced was only ever reduced to each row's minimum.
    return float(np.nanmedian(nearest_neighbor_distances(points, neighbors=1)))

def default_distance_additive(a, b, weight, factor):
    """
    Additive weighted distance function for Voronoi weighting.
    
    Returns the power distance in the normalized plane, ``|a - b|^2 - w^2``
    (Laguerre / power-diagram weighting): a site with weight ``w`` claims a
    disc of radius ``w`` (a fraction of the grid's longer extent) outright and
    competes beyond it. The value can be negative inside that disc; it is only
    ever compared, never used as a length.
    
    Args:
        a (np.ndarray): Grid points array shape (n, 2)
        b (tuple): Single point coordinate (x, y)
        weight (float): Weight parameter (affects distance directly)
        factor (float): Optional scale normalization factor. When > 0,
            effective additive weight is computed as ``weight / factor``.
        
    Returns:
        np.ndarray: Power distances for all points in a (smaller = closer)

    Notes:
        The previous ``sqrt(max(d^2 - w^2, 0.01))`` replaced every point inside
        a site's own disc by 0.1, so those points ranked *behind* points just
        outside the disc. The signed power distance is monotonic in true
        distance for every site.
    """
    a, b = normalize_plane(a, b)
    effective_weight = weight / factor if factor and factor > 0 else weight
    return np.sum((a - b) ** 2, axis=-1) - effective_weight ** 2

def default_distance_multiplicative(a, b, weight, factor):
    """
    Multiplicative weighted distance function for Voronoi weighting.
    
    Computes normalized Euclidean distance scaled inversely by weight.
    weight > 1 contracts the Voronoi region, weight < 1 expands it.
    
    Args:
        a (np.ndarray): Grid points array shape (n, 2)
        b (tuple): Single point coordinate (x, y)
        weight (float): Weight parameter (affects scaling - 1/weight)
        factor (float): Unused in multiplicative metric
        
    Returns:
        np.ndarray: Weighted distances for all points in a
        
    Notes:
        Division by weight means weight=0 causes division error (not handled).
    """
    a, b = normalize_plane(a, b)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))/weight

################################################################################
# SECTION 5: DATA PROCESSING & NORMALIZATION
################################################################################

def calculate_area(df, only_round=False):
    """
    Calculate Voronoi region areas from assigned points.
    
    Computes area (in m²) for each Voronoi region based on point assignments.
    Optionally rounds values before calculating to handle overlapping regions.
    
    Args:
        df (pd.GeoDataFrame): Points with 'geometry' (Point), 'weight', and a site ID column
        only_round (bool): Round weight values before area calculation (default False)
        
    Returns:
        pd.DataFrame: The provided input dataframe with area-derived columns,
        including ``base_values`` required by ``create_weights``.
        
    Notes:
        Assumes 1 point per cell in Voronoi grid
        Area = point_count × cell_size (depends on Voronoi grid resolution)
        Rounding reduces numeric precision but may handle overlaps better
    """

    if df is None or df.empty:
        logger.warning("Input dataframe is empty, returning as-is")
        return df
    logger.debug(f"Calculating area for {len(df)} WWTP facilities")
    if 'wwtp_area_rect' in df:
        df['wwtp_area_rect_2'] = df['wwtp_area_rect'].apply(
            lambda x: np.sum([
                float(i) for i in str(x).strip().strip('[]').split()
                if i and i.lower() != 'none'
            ]) if pd.notnull(x) else 0)
        df['round_area'] = df['diameters'].apply(parse_diameters_to_round_area)
        # The service radius is calibrated on the full detected footprint, so it
        # does not change with only_round (which only affects the weights).
        df['footprint_m2'] = df['round_area'] + df['wwtp_area_rect_2']
        if only_round:
            df['total_area'] = df['round_area']
            logger.debug(f"Using round areas only for {len(df)} facilities")
        else:
            df['total_area'] = df['footprint_m2']
            logger.debug(f"Combined round + building areas for {len(df)} facilities")
    else:
        logger.warning("No 'wwtp_area_rect' column found, using default area=1")
        df['total_area'] = 1
        df['footprint_m2'] = 0.0

    # Build detection-derived capacity proxy once so downstream weighting
    # logic can reuse these fields without recomputing.
    rect_counts = pd.to_numeric(
        df.get('num_detection_rect', pd.Series(0, index=df.index)),
        errors='coerce',
    ).fillna(0)
    circle_counts = pd.to_numeric(
        df.get('num_detection_circle', pd.Series(0, index=df.index)),
        errors='coerce',
    ).fillna(0)
    num_detections = (rect_counts.astype(int) + circle_counts.astype(int)).clip(lower=0)
    df['num_detections'] = num_detections
    df['capacity_proxy'] = df['total_area'] * np.sqrt(df['num_detections'])
    fallback_mean = df['capacity_proxy'].mean()
    df['base_values'] = df['capacity_proxy'].replace(0.0, np.nan).fillna(fallback_mean)

    return df

def normalize_column_to_rounded_str(series):
    """
    Convert numeric column values to rounded string IDs for grouping.
    
    Rounds numeric values and converts to strings for use as group keys.
    Handles NaN values by returning NaN. Used for grouping country/buffer IDs.
    
    Args:
        series (pd.Series): Numeric values to normalize
        
    Returns:
        pd.Series: String values rounded to nearest integer, NaN preserved
        
    Notes:
        Potential precision loss when rounding floating-point IDs
    """

    logger.debug(f"Normalizing {len(series)} values to rounded strings")
    numeric = pd.to_numeric(series, errors='coerce')
    rounded = numeric.round(0).astype('Int64')  # Use 'Int64' to allow NaNs
    nans_preserved = rounded.isna().sum()
    logger.debug(f"Conversion complete: {nans_preserved} NaN values preserved")
    return rounded.astype(str)

################################################################################
# SECTION 6: DUCKDB & EXTERNAL DATA INTEGRATION
################################################################################

def download_overture_maps(url, filepath):
    """
    Download and extract Overture Maps data from S3 URL.
    
    Downloads gzip-compressed Overture Maps GeoParquet file and extracts it.
    Creates parent directories if needed. Logs download progress.
    
    Args:
        url (str): S3 URL to gzip-compressed parquet file
        filepath (str): Local path to save extracted parquet file
        
    Returns:
        None (saves file to disk)
        
    Logs:
        INFO: Download start, progress, completion
        WARNING: Connection/download errors
        
    Raises:
        Exception: If download or extraction fails
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    logger.info(f"Starting download of Overture Maps from {url}")
    download_query  = f"""
    COPY(
        SELECT * -- REPLACE(ST_AsWKB(geometry)) as geometry
        FROM read_parquet('{url}', filename=true, hive_partitioning=1)
        WHERE subtype = 'country'
    )
    TO '{filepath}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    try:
        # duckdb_connection owns the scratch database: unique name, always removed.
        with duckdb_connection() as con:
            logger.debug("Installing and loading DuckDB SPATIAL extension")
            ensure_duckdb_spatial(con)
            logger.debug(f"Executing download query to save parquet at {filepath}")
            con.sql(download_query)
        logger.info(f"Successfully downloaded Overture Maps country boundaries to {filepath}")
    except Exception as err:
        logger.warning(f"Error downloading country polygons from Overture Maps: {err}")

def process_centroid(args):
    """
    Worker function for parallel polygon centroid matching.
    
    Given a single point centroid, finds which polygon contains it
    using spatial index query. Returns the value from the target column.
    
    Args:
        args (tuple): (centroid, spatial_index, polygon_gdf, column_name)
            - centroid: shapely.Point
            - spatial_index: rtree.index.Index
            - polygon_gdf: pd.GeoDataFrame with polygons
            - column_name: str, column to extract value from
            
    Returns:
        value: Value from polygon_gdf[column_name] or None if no match found
        
    Notes:
        Used in parallel workers for intersect_with_polygon_sindex()
        Returns None if centroid not in any polygon
    """

    centroid, sidx, polygons, col = args

    if centroid is None or centroid.is_empty or not centroid.is_valid:
        logger.debug(f"Skipping invalid centroid")
        return None

    possible_matches_index = list(sidx.intersection(centroid.bounds))
    if not possible_matches_index:
        logger.debug(f"No spatial index matches found for centroid at {centroid.x:.4f}, {centroid.y:.4f}")
        return None

    possible_matches = polygons.iloc[possible_matches_index]
    possible_matches = possible_matches[possible_matches.is_valid & ~possible_matches.is_empty]

    try:
        precise_matches = possible_matches[possible_matches.intersects(centroid)]
    except Exception:
        precise_matches = gpd.GeoDataFrame(columns=possible_matches.columns)

    if not precise_matches.empty:
        match_value = precise_matches.iloc[0][col]
        logger.debug(f"Found polygon intersection for centroid: {col}={match_value}")
        return match_value
    else:
        logger.debug(f"No precise polygon intersection found for centroid")
        return None       

def intersect_with_polygon_sindex(df, polygons, col, concurrency=False):
    """
    Intersect dataframe centroids with polygons using spatial indexing.
    
    Finds which polygon contains each point centroid using R-tree
    spatial index for efficiency. Optionally parallelizes via threads.
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input features to intersect.
    polygons : geopandas.GeoDataFrame
        Polygons providing the output attribute.
    col : str
        Polygon column to transfer to ``df``.
    concurrency : bool, default=False
        Whether to use a thread pool for centroid lookup.

    Returns
    -------
    geopandas.GeoDataFrame
        Input features with the transferred polygon column.

    Notes
    -----
    Invalid geometries are separated before intersection and concatenated back
    into the result after processing.
    """
    if df is None or df.empty:
        return df

    try:
        # Separate rows with invalid or missing geometry
        nans = df[df['geometry'].isna() | (~df['geometry'].is_valid) | (df['geometry'].is_empty)].copy()
        df = df[df['geometry'].notna() & df['geometry'].is_valid & ~df['geometry'].is_empty].copy()
        utm = df.crs
        # Create centroids safely
        polygons['geometry'] = polygons['geometry'].apply(buffer_geometry)
        df['geometry'] = df['geometry'].apply(buffer_geometry)
        df['centroid'] = df['geometry'].centroid

        # Build spatial index on polygon layer
        sidx = polygons.sindex
        matched_col_values = []
        args_list = [(centroid, sidx, polygons, col) for centroid in df['centroid']]
        if concurrency:
            logger.info(f"Intersecting {len(args_list)} centroids with polygons using ThreadPoolExecutor")
            with ThreadPoolExecutor() as executor:
                matched_col_values = list(executor.map(process_centroid, args_list))
        else:
            logger.info(f"Intersecting {len(args_list)} centroids with polygons (sequential)")
            matched_col_values = [process_centroid(args) for args in args_list]
        matched_count = sum(1 for v in matched_col_values if v is not None)
        logger.debug(f"Successfully matched {matched_count}/{len(args_list)} centroids to polygons")
        df[col] = matched_col_values

        df['geometry'] = df['geometry'].apply(buffer_geometry)
        df = df.drop(columns=['centroid'])
        df = pd.concat([df, nans], ignore_index=True)
        return gpd.GeoDataFrame(df, geometry='geometry', crs=utm)
    except Exception as err:
        logger.error(f"Error during spatial-index polygon intersection on column '{col}': {err}")
        raise

def intersect_with_polygons_db(df, polygons, cols, df_join_col='ISO_2', polygon_join_col='ISO_2'):
    """Intersect features with a polygon layer in a single UTM zone using DuckDB.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input features to intersect.
    polygons : geopandas.GeoDataFrame
        Polygon layer containing the attributes to transfer.
    cols : list[str] | tuple[str, ...] | str
        Polygon columns to transfer onto ``df``.
    df_join_col : str, default='ISO_2'
        Join column on ``df`` used to restrict polygon matches.
    polygon_join_col : str, default='ISO_2'
        Join column on ``polygons`` used to restrict polygon matches.

    Returns
    -------
    geopandas.GeoDataFrame
        Input features with the requested polygon columns added.

    Notes
    -----
    This variant assumes all input geometries already belong to the same UTM
    zone and uses DuckDB spatial SQL for the join.
    """
    if isinstance(cols, str):
        cols = [cols]
    cols = list(cols)
    select_cols = ",\n        ".join(
        f"b.{quote_sql_identifier(col)} AS {quote_sql_identifier(col)}" for col in cols
    )
    quoted_df_join_col = quote_sql_identifier(df_join_col)
    quoted_polygon_join_col = quote_sql_identifier(polygon_join_col)
    query2 = f"""
    WITH
    data AS (
        SELECT * REPLACE(ST_GeomFromText(centroid)) AS centroid
        FROM df
    ),
    polygons AS(
        SELECT * REPLACE(ST_GeomFromText(geometry)) AS geometry
        FROM polygons
    )
    SELECT
        a.* REPLACE(ST_AsText(a.centroid)) AS centroid, 
        {select_cols}
        FROM data a
        LEFT JOIN polygons b ON a.{quoted_df_join_col} = b.{quoted_polygon_join_col} 
        AND ST_IsValid(a.centroid)
        AND ST_IsValid(b.geometry)
        AND ST_Intersects(a.centroid, b.geometry)
    """
    if df is None or df.empty or polygons is None or polygons.empty:
        return df
    if df_join_col not in df.columns:
        raise KeyError(f"Join column '{df_join_col}' not found in df.")
    if polygon_join_col not in polygons.columns:
        raise KeyError(f"Join column '{polygon_join_col}' not found in polygons.")
    if df.crs is None:
        # set_crs assigns a missing CRS; to_crs reprojects and needs one already.
        df = df.set_crs(4326)
    if polygons.crs is None:
        polygons = polygons.set_crs(4326)

    utm = df['utm'].mode()[0]
    df = df.to_crs(utm)
    polygons = polygons.to_crs(utm)

    nans = df[df['geometry'].isna() | (~df['geometry'].is_valid) | (df['geometry'].is_empty)].copy()
    df = df[df['geometry'].notna() & df['geometry'].is_valid & ~df['geometry'].is_empty].copy()
    
    df['centroid'] = df['geometry'].apply(create_centroid_points)
    df['centroid'] = df['centroid'].map(safe_to_wkt)
    df['geometry'] = df['geometry'].map(safe_to_wkt)
    polygons['geometry'] = polygons['geometry'].map(safe_to_wkt)
    
    logger.info(f"Starting DuckDB polygon intersection for {len(df)} points")
    try:
        # duckdb_connection owns the scratch database: unique name, always removed.
        with duckdb_connection() as con:
            ensure_duckdb_spatial(con)
            df = con.execute(query2).df()
            logger.debug(f"Query returned {len(df)} results")

            df = df.drop(labels=['centroid'], axis=1)
            df['geometry'] = df['geometry'].map(lambda x: from_wkt(x) if not pd.isna(x) else None)
            df = pd.concat([df, nans], ignore_index=True)
            df['geometry'] = df['geometry'].map(buffer_geometry)
            return gpd.GeoDataFrame(df, geometry='geometry', crs=utm).to_crs(4326)
    except Exception as err:
        # Never return `df` here: at this point its geometry column holds WKT
        # strings, its null rows have been split off, and the requested `cols`
        # are absent - the caller would wrap that in a GeoDataFrame and carry on.
        logger.error(f'Error during DuckDB polygon intersection: {err}')
        raise

def intersect_with_polygons_parallelized(df, polygons, cols, use_duckdb=False, max_workers=16, df_join_col='ISO_2', polygon_join_col='ISO_2'):
    """
    Parallel polygon intersection with automatic UTM zone partitioning.
    
    Partitions data by UTM projection zone, processes each zone in parallel
    using either spatial indexing (default) or DuckDB spatial SQL, then
    concatenates results. Handles invalid geometries separately.
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input features to intersect.
    polygons : geopandas.GeoDataFrame
        Polygon layer with the requested attribute columns.
    cols : list[str] | tuple[str, ...] | str
        Polygon columns to transfer to ``df``.
    use_duckdb : bool, default=False
        Whether to use the DuckDB per-zone implementation instead of the spatial
        index implementation.
    max_workers : int, default=16
        Maximum number of worker processes for zone-based processing.
    df_join_col : str, default='ISO_2'
        Join column on ``df`` used when ``use_duckdb=True``.
    polygon_join_col : str, default='ISO_2'
        Join column on ``polygons`` used when ``use_duckdb=True``.

    Returns
    -------
    geopandas.GeoDataFrame
        Input features with the transferred polygon columns.

    Notes
    -----
    The workflow partitions both inputs by estimated UTM zone, processes each
    zone independently, and restores invalid or missing geometries at the end.
    """
    if isinstance(cols, str):
        cols = [cols]
    cols = list(cols)
    nans = df[df['geometry'].isna() | (~df['geometry'].is_valid) | (df['geometry'].is_empty)].copy().reset_index(drop=True) 
    df = df[df['geometry'].notna() & df['geometry'].is_valid & ~df['geometry'].is_empty].copy().reset_index(drop=True)  
    
    df['utm'] = df['geometry'].apply(estimate_utm_epsg_for_geom)
    polygons['utm'] = polygons['geometry'].apply(estimate_utm_epsg_for_geom)
    
    data = []
    unique_utms = set(df['utm'].unique()).union(polygons['utm'].unique())
    func = intersect_with_polygon_sindex if not use_duckdb else intersect_with_polygons_db
    if not use_duckdb:
        for utm in unique_utms:
            sub_df = df[df['utm'] == utm].copy()
            sub_polygons = polygons[polygons['utm'] == utm].copy()
            if sub_df.empty:
                continue
            for col in cols:
                sub_df = func(sub_df, sub_polygons, col)
            if sub_df is not None:
                data.append(sub_df)
    else:
        for utm in unique_utms:
            result = intersect_with_polygons_db(
                df[df['utm'] == utm].copy(),
                polygons[polygons['utm'] == utm].copy(),
                cols,
                df_join_col=df_join_col,
                polygon_join_col=polygon_join_col,
            )
            if result is not None:
                data.append(result)
    data.append(nans)
    return gpd.GeoDataFrame(pd.concat(data, ignore_index=True), geometry='geometry', crs=4326)

def intersects_with_country_db(df, filepath, polygon_country_col='country', output_country_col='ISO_2'):
    """
    Intersect point geometries with country boundaries using DuckDB.
    
    Performs spatial join to find country ISO_2 codes for each point
    using bounding box filtering followed by precise intersection test.
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input features to enrich with country codes.
    filepath : str
        Path to the country-boundary parquet file.
    polygon_country_col : str, default='country'
        Country-code column in the boundary parquet.
    output_country_col : str, default='ISO_2'
        Output column added to ``df`` for the matched country code.

    Returns
    -------
    geopandas.GeoDataFrame
        Input features with the configured output country column.

    Notes
    -----
    The current implementation serializes geometries as WKT before executing the
    DuckDB spatial join.
    """
    quoted_polygon_country_col = quote_sql_identifier(polygon_country_col)
    quoted_output_country_col = quote_sql_identifier(output_country_col)
    logger.info(f"Starting DuckDB country boundary intersection for {len(df)} points")
    # One country per input row. A plain LEFT JOIN returned one row per
    # matching polygon, so a point inside a country stored as both a land and
    # a maritime polygon, or a basin crossing a border, came back duplicated;
    # and requiring the feature's bbox to lie inside the country's bbox left
    # cross-border features unmatched. Rows with several candidate countries
    # take the one with the largest overlap (ties: lowest code).
    query2 = f"""
    WITH
    data AS (
        SELECT *,
        ST_XMax(geometry) AS LON_MAX,
        ST_XMin(geometry) AS LON_MIN,
        ST_YMax(geometry) AS LAT_MAX,
        ST_YMin(geometry) AS LAT_MIN
        FROM (
            SELECT * REPLACE(ST_GeomFromText(geometry)) AS geometry
            FROM df
        )
    ),
    countries AS (
        SELECT {quoted_polygon_country_col} AS __country, geometry,
        ST_XMax(geometry) AS LON_MAX,
        ST_XMin(geometry) AS LON_MIN,
        ST_YMax(geometry) AS LAT_MAX,
        ST_YMin(geometry) AS LAT_MIN
        FROM ( SELECT * REPLACE(ST_GeomFromWKB(geometry)) AS geometry
            FROM read_parquet('{filepath}')
        )
    ),
    -- No geometry here: carrying the country polygon per matched row copied a
    -- whole country boundary for every feature (out of memory on detailed
    -- boundaries). Multi-country rows re-read it below.
    matches AS (
        SELECT a.__rid, b.__country
        FROM data a
        JOIN countries b ON
            a.LON_MIN <= b.LON_MAX
            AND a.LON_MAX >= b.LON_MIN
            AND a.LAT_MIN <= b.LAT_MAX
            AND a.LAT_MAX >= b.LAT_MIN
            AND ST_Intersects(a.geometry, b.geometry)
    ),
    distinct_matches AS (SELECT DISTINCT __rid, __country FROM matches),
    counts AS (SELECT __rid, count(*) AS __n FROM distinct_matches GROUP BY __rid),
    single AS (
        SELECT d.__rid, d.__country FROM distinct_matches d JOIN counts c USING (__rid) WHERE c.__n = 1
    ),
    multi AS (
        SELECT m.__rid, m.__country, sum(ST_Area(ST_Intersection(a.geometry, b.geometry))) AS __overlap
        FROM distinct_matches m
        JOIN counts c USING (__rid)
        JOIN data a USING (__rid)
        JOIN countries b ON b.__country = m.__country
            AND a.LON_MIN <= b.LON_MAX AND a.LON_MAX >= b.LON_MIN
            AND a.LAT_MIN <= b.LAT_MAX AND a.LAT_MAX >= b.LAT_MIN
        WHERE c.__n > 1
        GROUP BY m.__rid, m.__country
    ),
    multi_best AS (
        SELECT __rid, __country FROM (
            SELECT *, row_number() OVER (PARTITION BY __rid ORDER BY __overlap DESC, __country) AS __rn
            FROM multi
        ) WHERE __rn = 1
    ),
    chosen AS (SELECT * FROM single UNION ALL SELECT * FROM multi_best)
    SELECT
        a.* EXCLUDE (__rid, LON_MAX, LON_MIN, LAT_MAX, LAT_MIN) REPLACE (ST_AsText(a.geometry) AS geometry),
        ch.__country AS {quoted_output_country_col}
    FROM data a
    LEFT JOIN chosen ch USING (__rid)
    ORDER BY a.__rid
    """
    if df is None or df.empty:
        logger.warning("Input dataframe is empty, returning as-is")
        return df

    try:
        if df.crs is None:
            df = df.set_crs(4326)
        elif df.crs.to_epsg() != 4326:
            logger.debug(f"Transforming from {df.crs.to_epsg()} to EPSG:4326 for intersection")
            df = df.to_crs(epsg=4326)

        logger.debug(f"Converting {len(df)} geometries to WKT format")
        df = pd.DataFrame(df)
        df['geometry'] = df['geometry'].map(safe_to_wkt)
        df['__rid'] = np.arange(len(df))
        # duckdb_connection owns the scratch database: unique name, always removed.
        with duckdb_connection() as con:
            ensure_duckdb_spatial(con)
            logger.debug("Executing DuckDB spatial intersection query")
            df = con.sql(query2).df()
        iso_matched = df[output_country_col].notna().sum()
        logger.info(f"DuckDB intersection complete: {iso_matched}/{len(df)} points matched to countries")
        df['geometry'] = df['geometry'].map(lambda x: from_wkt(x) if not pd.isna(x) else None)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
        df['geometry'] = df['geometry'].map(buffer_geometry)
        return df
    except Exception as err:
        logger.error(f"Error during DuckDB country boundary intersection using {filepath}: {err}")
        raise

################################################################################
# SECTION 7: BUFFER & GEOMETRY DISSOLUTION
################################################################################

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

################################################################################
# SECTION 8: VORONOI COMPUTATION & ORCHESTRATION
################################################################################

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

def extract_site_coordinates(df, centroid_points):
    """
    Extract Voronoi site coordinates from dataframe geometries.
    
        Derives site coordinates from input geometries with centroid fallback.
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input sites with geometry information.
    centroid_points : bool
        Whether to derive coordinates from feature centroids.
    Returns
    -------
    list[tuple[float | None, float | None]]
        Site coordinates used by the Voronoi solver.

    Notes
    -----
    Coordinates are derived from feature geometries and are suitable for
    Voronoi assignment.
    """
    logger.debug(f"Computing centroids from {len(df)} site geometries")
    points = []
    for geom in df['geometry'].apply(create_centroid_points):
        if isinstance(geom, Point):
            points.append((geom.x, geom.y))
        elif isinstance(geom, (LineString, MultiLineString, Polygon, MultiPolygon)):
            points.append((geom.centroid.x, geom.centroid.y))
        else:
            points.append((None, None))
    logger.debug(f"Computed {len(points)} site centroids")
    
    return points

LAGOON_COLUMN = 'lagoon_detected'
SIZE_MEASURES = ('reported', 'footprint')
_WARNED_MISSING_SIZE_COLUMNS = set()  # warn once per process, not once per basin group


def _truthy(series):
    return series.astype(str).str.strip().str.lower().isin({'true', '1', '1.0', 'yes'})


def reported_size_people(df, columns, skip_quality):
    """Reported plant size in people: the first positive value among ``columns``.

    Parameters
    ----------
    df : pandas.DataFrame
        Plant records.
    columns : sequence of str
        Columns in order of preference (e.g. ``POP_SERVED``, ``EPA POP 2022``,
        ``uwwLoadEnteringUWWTP``). Missing columns are skipped.
    skip_quality : dict
        ``{column: {quality_column: [codes]}}``: values of ``column`` are ignored
        on rows whose ``quality_column`` is one of ``codes`` (HydroWASTE
        ``QUAL_POP`` 4: people within 11 km of the plant, not a plant record).

    Returns
    -------
    numpy.ndarray
        People per row; NaN where no column holds a positive value.
    """
    for column in columns:
        if column not in df.columns and column not in _WARNED_MISSING_SIZE_COLUMNS:
            _WARNED_MISSING_SIZE_COLUMNS.add(column)
            logger.warning("Reported size column %r is not in the plant data; those plants use the footprint estimate", column)
    values = {column: pd.to_numeric(df[column], errors='coerce') for column in columns if column in df.columns}
    for column, rules in skip_quality.items():
        for quality_column, codes in rules.items():
            if column in values and quality_column in df.columns:
                quality = pd.to_numeric(df[quality_column], errors='coerce')
                values[column] = values[column].where(~quality.isin([float(c) for c in codes]))
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for column in columns:
        if column in values:
            out = out.fillna(values[column].where(values[column] > 0))
    return out.to_numpy(dtype=float)


def plant_size_people(df, size_measure, footprint_to_people, reported_size_columns, reported_size_skip, **_):
    """Plant size in people, used for both the service radius and the Voronoi weight.

    Parameters
    ----------
    df : pandas.DataFrame
        Plant records with ``footprint_m2`` (from ``calculate_area``; falls back to
        ``total_area``) and optionally ``lagoon_detected`` and the reported columns.
    size_measure : {'reported', 'footprint'}
        ``reported``: the reported size where there is one (see
        ``reported_size_people``), the footprint estimate elsewhere.
        ``footprint``: the footprint estimate for every plant.
    footprint_to_people : dict
        ``intercept`` and ``slope`` of ``log10 people = intercept + slope * log10
        footprint_m2`` and ``unsized_people`` for lagoon plants and plants
        without detections, whose footprint says nothing about their size.
    reported_size_columns, reported_size_skip
        Passed to ``reported_size_people`` (used with ``reported`` only).

    Returns
    -------
    numpy.ndarray
        People per row (always > 0).
    """
    if size_measure not in SIZE_MEASURES:
        raise ValueError(f"size_measure must be one of {SIZE_MEASURES}, got {size_measure!r}")
    if 'footprint_m2' in df.columns:
        footprint = pd.to_numeric(df['footprint_m2'], errors='coerce')
    elif 'total_area' in df.columns:
        footprint = pd.to_numeric(df['total_area'], errors='coerce')
    else:
        footprint = pd.Series(0.0, index=df.index)
    footprint = np.nan_to_num(footprint.to_numpy(dtype=float), nan=0.0)
    lagoon = _truthy(df[LAGOON_COLUMN]).to_numpy() if LAGOON_COLUMN in df.columns else np.zeros(len(df), bool)
    sized = (footprint > 0) & ~lagoon
    people = np.where(
        sized,
        np.power(10.0, float(footprint_to_people['intercept'])
                 + float(footprint_to_people['slope']) * np.log10(np.where(sized, footprint, 1.0))),
        float(footprint_to_people['unsized_people']),
    )
    if size_measure == 'reported':
        reported = reported_size_people(df, reported_size_columns, reported_size_skip)
        people = np.where(np.isfinite(reported), reported, people)
    return people


def service_radius_m(size_people, radius_at_10k_people_km, exponent, max_m):
    """Service radius (m): ``radius_at_10k_people_km * (people / 10,000) ** exponent``, at most ``max_m``."""
    people = np.asarray(size_people, dtype=float)
    radius_km = float(radius_at_10k_people_km) * np.power(people / 1e4, float(exponent))
    return np.minimum(radius_km * 1000.0, float(max_m))


def calculate_buffer(df, weights, *args, **kwargs):
    """Per-site service radius in metres.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input sites (see ``plant_size_people`` for the columns used).
    weights : numpy.ndarray
        Voronoi weights; unused, kept for the ``calculate_buffer_fn`` signature.
    **kwargs
        ``buffer`` (m), ``dynamic_buffering``, ``max_buffer`` (m), and with
        dynamic buffering ``size_measure``, ``footprint_to_people``,
        ``reported_size_columns``, ``reported_size_skip``, ``radius_exponent``
        and ``radius_at_10k_people_km`` (starter picks the latter from
        ``radius_calibration`` for the size measure).

    Returns
    -------
    numpy.ndarray
        Radius in metres for each row of ``df``.

    Notes
    -----
    With ``dynamic_buffering`` false every site gets ``buffer`` (at most
    ``max_buffer``). Otherwise the radius grows with plant size in people. The
    exponent and the radius at 10,000 people were chosen on US census
    population in EPA sourced sewersheds (``src.calibration.service_radius``);
    the GB catchments are the independent test.
    """
    max_buffer = kwargs['max_buffer']
    if not kwargs['dynamic_buffering']:
        fixed = float(min(kwargs['buffer'], max_buffer))
        return np.full(len(df), fixed, dtype=float)
    people = plant_size_people(df, **kwargs)
    radius = service_radius_m(people, kwargs['radius_at_10k_people_km'], kwargs['radius_exponent'], max_buffer)
    logger.debug("Service radius %.0f-%.0f m (median %.0f m) for %d sites",
                 radius.min(), radius.max(), np.median(radius), len(df))
    return radius

def initialize_voronoi_weights(df, distance_fn, scale_weights, points):
    """
    Initialize weight parameters for Voronoi computation based on distance function type.
    
    Handles weight setup differently depending on whether additive or multiplicative
    distance metrics are used. Optionally applies weight scaling based on nearest neighbors.
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input sites with a ``weights`` column.
    distance_fn : callable
        Distance function used by the Voronoi solver.
    scale_weights : bool
        Whether to scale weights using nearest-neighbor spacing.
    points : list | numpy.ndarray
        Site coordinates used for optional scaling.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Weight array and the applied scaling factor.

    Notes
    -----
    Additive and multiplicative distance functions use different default weight
    initializations, so the returned array depends on the selected metric.
    """
    weights = df['weights'].values.astype(float)
    factor = 0
    
    logger.debug(f"Initializing weights for {len(df)} sites (distance_fn={distance_fn.__name__}, scale_weights={scale_weights})")
    
    if distance_fn == default_distance_additive:
        if scale_weights:
            logger.debug("Additive distance: Computing weight scaling factor")
            #factor = auto_weight_scale(points)
            #if factor is None or pd.isna(factor) or factor <= 0:
            #    factor = 1.0
            factor = 1.0
            weights = weights * factor
            logger.debug(f"Additive distance: Applied scaling factor {factor:.4f}")
        else:
            logger.debug("Additive distance: Using zero weights (standard Euclidean)")
            weights = np.zeros(len(df))
    elif distance_fn == default_distance_multiplicative:
        if not scale_weights:
            logger.debug("Multiplicative distance: Using equal weights (standard Voronoi)")
            weights = np.ones(len(df))
        else:
            logger.debug("Multiplicative distance: Using provided weights)")
    logger.debug(f"Weight initialization complete: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
    return weights, factor

def extract_contours_scipy(region_mask_2d, n_points, grid_minx, grid_miny):
    """
    Extract polygon contours from a region mask using scipy.measure.find_contours.
    
    Uses scipy's marching squares algorithm to find contours in a 2D binary mask,
    then converts contours to polygon coordinates in the original coordinate system.
    
    Parameters
    ----------
    region_mask_2d : numpy.ndarray
        Boolean mask indicating region membership.
    n_points : int
        Grid spacing multiplier.
    grid_minx : float
        Minimum x coordinate of the grid origin.
    grid_miny : float
        Minimum y coordinate of the grid origin.

    Returns
    -------
    list[shapely.geometry.Polygon]
        Valid polygons extracted from the mask.
    """
    polygons = []
    contours = find_contours(region_mask_2d, level=0.5, fully_connected='low', positive_orientation='low')
    logger.debug(f"scipy.find_contours extracted {len(contours)} contours")
    
    for contour in contours:
        contour = np.array(contour)
        # Convert from grid indices to actual coordinates
        contour_x = contour[:, 1] * n_points + grid_minx
        contour_y = contour[:, 0] * n_points + grid_miny
        poly_coords = np.stack([contour_x, contour_y], axis=-1).reshape(-1, 2)
        
        # Ensure polygon is closed
        if not np.array_equal(poly_coords[0], poly_coords[-1]):
            poly_coords = np.vstack([poly_coords, poly_coords[0]])
        
        # Create and validate polygon
        poly = Polygon(poly_coords).buffer(0)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    
    logger.debug(f"scipy contour extraction created {len(polygons)} valid polygons")
    return polygons

def extract_contours_cv2(region_mask_2d, n_points, grid_minx, grid_miny):
    """
    Extract polygon contours from a region mask using OpenCV's contour finding.
    
    Uses cv2.findContours with external contour retrieval to extract boundaries
    from a binary mask, then converts contours to polygon coordinates.
    
    Parameters
    ----------
    region_mask_2d : numpy.ndarray
        Boolean mask indicating region membership.
    n_points : int
        Grid spacing multiplier.
    grid_minx : float
        Minimum x coordinate of the grid origin.
    grid_miny : float
        Minimum y coordinate of the grid origin.

    Returns
    -------
    list[shapely.geometry.Polygon]
        Valid polygons extracted from the mask.
    """
    polygons = []
    mask_uint8 = (region_mask_2d.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"cv2.findContours extracted {len(contours)} contours")
    
    for contour in contours:
        contour = contour.squeeze()
        # Skip invalid contours
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        
        # Convert from grid indices to actual coordinates
        contour_x = contour[:, 0] * n_points + grid_minx
        contour_y = contour[:, 1] * n_points + grid_miny
        poly_coords = np.stack([contour_x, contour_y], axis=-1).reshape(-1, 2)
        
        # Ensure polygon is closed
        if not np.array_equal(poly_coords[0], poly_coords[-1]):
            poly_coords = np.vstack([poly_coords, poly_coords[0]])
        
        # Create and validate polygon
        poly = Polygon(poly_coords).buffer(0)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    
    logger.debug(f"cv2 contour extraction created {len(polygons)} valid polygons")
    return polygons

def extract_contours_rasterio(region_mask_2d, n_points, grid_minx, grid_miny):
    """
    Extract polygon contours from a region mask using rasterio's feature extraction.
    
    Uses rasterio.features.shapes to convert a raster mask to vector polygons,
    then applies affine transformations to map grid coordinates to actual space.
    
    Parameters
    ----------
    region_mask_2d : numpy.ndarray
        Boolean mask indicating region membership.
    n_points : int
        Grid spacing multiplier.
    grid_minx : float
        Minimum x coordinate of the grid origin.
    grid_miny : float
        Minimum y coordinate of the grid origin.

    Returns
    -------
    list[shapely.geometry.Polygon]
        Valid polygons extracted from the mask.
    """
    polygons = []
    results = shapes(region_mask_2d.astype(np.uint8), mask=region_mask_2d > 0)
    for geom, val in results:
        poly = shape(geom)
        # Pixel (i, j) covers index space [i, i+1] x [j, j+1] while its grid
        # node sits at index i: shift by half a cell so each cell is centred
        # on its node, then scale by the grid step.
        poly = shapely.affinity.scale(poly, xfact=n_points, yfact=n_points, origin=(0, 0))
        poly = shapely.affinity.translate(poly, xoff=grid_minx - n_points / 2.0, yoff=grid_miny - n_points / 2.0)
        poly = poly.buffer(0)
        
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    
    logger.debug(f"rasterio contour extraction created {len(polygons)} valid polygons")
    return polygons

def finalize_gdf(df_list, cols):
    """
    Finalize and concatenate list of GeoDataFrames into single output.
    
    Concatenates multiple GeoDataFrames from parallel Voronoi workers,
    applies topology normalization, and returns single combined GeoDataFrame.
    
    Parameters
    ----------
    df_list : list
        GeoDataFrames to concatenate.
    cols : pandas.Index
        Column names for the empty fallback frame.

    Returns
    -------
    geopandas.GeoDataFrame
        Concatenated GeoDataFrame in EPSG:4326.
    """
    if len(df_list) > 0:
        df = pd.concat(df_list, ignore_index=True)
    else:
        # fallback: empty GeoDataFrame with geometry column
        df = pd.DataFrame(columns=cols)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='epsg:4326')
    if len(df) > 0:
        df['geometry'] = df['geometry'].map(buffer_geometry)
    return df

def assign_sites_streaming(valid_points, points, weights, distance_fn, factor):
    """
    Assign each grid point to nearest Voronoi site using weighted distance.
    
    Computes distance from each grid point to all sites using the provided
    distance function, tracking minimum distance site for streaming computation.
    Supports multiplicative and additive weighting schemes.
    
    Parameters
    ----------
    valid_points : numpy.ndarray
        Candidate grid points to assign.
    points : numpy.ndarray
        Site locations.
    weights : numpy.ndarray
        Weight value for each site.
    distance_fn : callable
        Distance function used for assignment.
    factor : float | None
        Optional scaling factor passed into ``distance_fn``.

    Returns
    -------
    numpy.ndarray
        Site assignment index for each valid grid point.

    Notes
    -----
    The streaming implementation avoids materializing a full distance matrix for
    large grids.
    """

    n_points = valid_points.shape[0]
    best_distances = np.full(n_points, np.inf)
    assignments = np.full(n_points, -1, dtype=int)
    logger.debug(f"Assigning {n_points} grid points to {len(points)} sites")

    for idx, (site, weight) in enumerate(zip(points, weights)):
        dist = distance_fn(valid_points, site, weight, factor)
        mask = dist < best_distances
        best_distances[mask] = dist[mask]
        assignments[mask] = idx
    
    assigned_count = np.sum(assignments >= 0)
    logger.debug(f"Assignment complete: {assigned_count}/{n_points} points assigned")
    return assignments

def weighted_voronoi(df, col, country_clip, scale_weights=False, clipping=None, n_points=100, distance_fn=default_distance_multiplicative,
                      scipy_true=False, cv2_true=False, centroid_points=False, buffering=False, threshold=500,
                      calculate_buffer_fn=calculate_buffer, buffer_fn_kwargs=None, site_id_col='WASTE_ID',
                      size_weights=False):
    """
    Generate weighted Voronoi diagram from point sites with multiple contour methods.
    
    Creates Voronoi regions using weighted distance metrics. Generates grid of points,
    assigns each to nearest site, then extracts contours using scipy/cv2/rasterio helper
    functions (extract_contours_scipy, extract_contours_cv2, extract_contours_rasterio).
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input sites with geometry, weights, and identifier columns.
    col : str
        Column used to identify sites in the output.
    country_clip : geopandas.GeoDataFrame | None
        Country boundary used for final clipping.
    scale_weights : bool, default=False
        Whether to scale weights by local spacing.
    clipping : geopandas.GeoDataFrame | None, default=None
        Optional additional clipping boundary.
    n_points : int, default=100
        Grid resolution used for the rasterized Voronoi assignment.
    distance_fn : callable, default=default_distance_multiplicative
        Distance function used by the weighted Voronoi solver.
    scipy_true : bool, default=False
        Whether to extract contours with SciPy.
    cv2_true : bool, default=False
        Whether to extract contours with OpenCV.
    centroid_points : bool, default=False
        Whether to derive site coordinates from centroids.
    buffering : bool, default=False
        Whether to intersect regions with local feature buffers.
    threshold : float | int, default=500
        Clustering threshold applied before Voronoi generation.
    calculate_buffer_fn : callable, default=calculate_buffer
        Buffer-length function used to compute per-site ``buffer_length`` values.
        Signature must start with
        ``(df, weights, *args, **kwargs)``.
    buffer_fn_kwargs : dict | None, default=None
        Keyword arguments forwarded to ``calculate_buffer_fn``.
    site_id_col : str, default='WASTE_ID'
        Column on ``df`` used as the unique site identifier for region assembly.
    size_weights : bool, default=False
        Weight each site by the square root of its size in people
        (``plant_size_people`` with ``buffer_fn_kwargs``), computed after
        clustering. Requires the multiplicative distance.

    Returns
    -------
    tuple
        Tuple ``(region_polygons, point_df)`` containing the generated Voronoi
        regions and the subset of points represented in the final output.

    Notes
    -----
    For a single-site group, the function returns the clipped service area
    derived from buffering and clipping inputs rather than a mathematically
    meaningful Voronoi partition. ``clipping`` and ``country_clip`` refer to
    geometric clipping boundaries, not an ML CLIP model.
    """

    if df is None or df.empty:
        logger.warning("Input dataframe for weighted_voronoi is empty")
        return
    if site_id_col not in df.columns:
        raise KeyError(f"Site identifier column '{site_id_col}' not found in weighted_voronoi input dataframe.")
    logger.info(f"Starting weighted Voronoi generation for {len(df)} sites (n_points={n_points})")
    
    # === PHASE 1: CRS VALIDATION & PROJECTION ===
    # Ensure all geometries have a proper coordinate system
    # The Projection to UTM must have happened in the previous steps, 
    # but we check again here to be safe
    if df.crs is None:
        df = df.set_crs('epsg:4326')
        utm = estimate_utm_crs(df)
        df = df.to_crs(utm)
        logger.debug(f"Projected to UTM CRS: {utm}")
    crs = df.crs
    
    # === PHASE 2: SITE PREPROCESSING ===
    # Cluster nearby points to avoid creating duplicate Voronoi regions
    # Points within 'threshold' distance are merged together with aggregate properties
    df = gpd.GeoDataFrame(cluster_points(df, threshold), geometry='geometry', crs=crs)
    logger.debug(f"After clustering: {len(df)} sites")

    # === PHASE 3: SITE COORDINATES & WEIGHT INITIALIZATION ===
    # For multi-site groups, extract point locations and initialize weights.
    # Dynamic buffering also computes per-site buffer lengths here.
    points = extract_site_coordinates(df, centroid_points)
    weights, factor = initialize_voronoi_weights(df, distance_fn, scale_weights, points)
    if buffer_fn_kwargs is None:
        buffer_fn_kwargs = {}
    fn_kwargs = dict(buffer_fn_kwargs)
    buffer_lengths = calculate_buffer_fn(
        df,
        weights,
        **fn_kwargs,
    )
    buffer_lengths = np.asarray(buffer_lengths, dtype=float)
    if len(buffer_lengths) != len(df):
        raise ValueError(
            f"calculate_buffer_fn returned {len(buffer_lengths)} buffer lengths for {len(df)} rows"
        )
    df['buffer_length'] = buffer_lengths
    if size_weights:
        if distance_fn is not default_distance_multiplicative:
            raise ValueError("size weights need the multiplicative distance (weight_func 'mult')")
        # A multiplicatively weighted site's area grows with the square of its
        # weight, so sqrt(people) gives areas in proportion to people at equal
        # density. Computed here, after clustering, from the merged records.
        weights = np.sqrt(plant_size_people(df, **fn_kwargs))
    # === PHASE 4: BUFFERED EXTENT FOR GRID DOMAIN ===
    # The clipping input may contain multiple geometries, so we unify it into
    # a single geometry and ensure it uses the same CRS as the sites.
    # If a clipping geometry is provided, we use it directly.
    # Otherwise, we fall back to a buffered bounding box around the sites.
    # We use this same buffered extent to build the Voronoi grid because
    # using a much larger clipping geometry can increase computation cost
    # significantly (approximately O(nm) for grid dimensions n and m).
    minx, miny, maxx, maxy = df.geometry.buffer(np.max(buffer_lengths)).total_bounds
    
    # === PHASE 5: CLIPPING GEOMETRY PREPARATION ===
    # Use provided clipping geometry when available (CRS-aligned), otherwise use
    # the buffered extent bounds as the clipping object.
    actual_clipping_object = None
    if clipping is not None and not clipping.empty:
        if clipping.crs is None:
            clipping = clipping.set_crs('epsg:4326')
        if clipping.crs != crs:
            clipping = clipping.to_crs(crs)
        actual_clipping_object = buffer_geometry(unary_union(clipping.geometry))
    else:
        actual_clipping_object = buffer_geometry(box(minx, miny, maxx, maxy))
    
    # === PHASE 6: COUNTRY CRS ALIGNMENT ===
    # Ensure country clipping geometry is aligned with site CRS
    # This is critical for accurate clipping during final boundary operations
    if country_clip is not None:
        if country_clip.crs is None:
            country_clip = country_clip.set_crs('epsg:4326')
        if  country_clip.crs != crs:
            country_clip = country_clip.to_crs(crs)
    
    if len(df) == 1:
        # === PHASE 7: SPECIAL CASE (SINGLE SITE) ===
        # When only one site exists, create Voronoi region from clipping boundary
        # This is NOT a true Voronoi diagram, but the site's service area
        region_polygons = df.copy()
        point_df = df.copy().reset_index(drop=True)
        region_geom = df.iloc[0]['geometry'].buffer(df.iloc[0]['buffer_length'])
        geom = df.iloc[0]['geometry']

        if isinstance(region_geom, (Point, Polygon, MultiPolygon, LineString, MultiLineString)):
            region_geom = buffer_geometry(region_geom)  # type: ignore[index]

        # Optionally intersect region with buffer around site point
        if buffering:
            point_buffer = geom.centroid.buffer(df.iloc[0]['buffer_length'])
            region_geom = region_geom.intersection(point_buffer).buffer(0)  # type: ignore[union-attr, index]
        region_polygons.at[region_polygons.index[0], 'geometry'] = cast(Any, region_geom)  # type: ignore[index]

        # Clip region to actual clipping geometry to ensure it does not exceed bounds
        region_polygons = gpd.clip(region_polygons, actual_clipping_object)  # type: ignore[arg-type]
        region_polygons['geometry'] = region_polygons['geometry'].map(buffer_geometry)
        
        # Apply country boundary clipping if provided
        if country_clip is not None and not country_clip.empty:
            region_polygons = gpd.clip(region_polygons, country_clip)
        region_polygons['geometry'] = region_polygons['geometry'].map(buffer_geometry)

        # Same 'area' column the multi-site branch emits, computed in the metric
        # CRS before reprojecting - without it downstream consumers see the
        # column only for multi-site groups.
        region_polygons['area'] = region_polygons.geometry.area

        # Convert all outputs to WGS84 for standard output format
        region_polygons = region_polygons.to_crs(4326)
        point_df = point_df.to_crs(4326)
        return region_polygons, point_df

    # === PHASE 8: GRID GENERATION ===
    # Use adaptive step sizing to ensure reasonable coverage
    x_coords = create_ranges(minx, maxx, n_points)
    y_coords = create_ranges(miny, maxy, n_points)
    # Create 2D mesh grid
    xv, yv = np.meshgrid(x_coords, y_coords)
    # Flatten to list of (x, y) coordinate pairs
    grid_points = np.column_stack((xv.ravel(), yv.ravel()))
    del xv, yv
    grid_minx = x_coords[0]
    grid_miny = y_coords[0]
  
    # === PHASE 9: GRID MASKING & SITE ASSIGNMENT ===
    # Filter grid points to include only those within the clipping boundary to optimize assignment.
    # Extract only points that are inside the clipping boundary
    # Assign each grid point to its nearest weighted site
    # This is the core Voronoi computation step
    mask = geometry_contains_points(actual_clipping_object, grid_points)
    valid_points = grid_points[mask]
    valid_flat_indices = np.flatnonzero(mask)
    del grid_points

    assignments = assign_sites_streaming(valid_points, points, weights, distance_fn, factor)
    assignment_grid = np.full(mask.shape, -1, dtype=np.int32)
    assignment_grid[valid_flat_indices] = assignments
    assignment_grid_2d = assignment_grid.reshape(len(y_coords), len(x_coords))
    del assignment_grid, valid_flat_indices, valid_points
    
    # === PHASE 10: REGION BOUNDARY EXTRACTION ===
    # Build Voronoi region polygon for each site
    region_polygons = []
    df.reset_index(drop=True, inplace=True)
    for point, (i, row) in zip(points, df.iterrows()):
        assigned_to_site = assignments == i
        if not np.any(assigned_to_site):
            # No points assigned to this site: create empty region placeholder
            region_polygons.append({site_id_col: row[site_id_col], 'geometry': None})
            continue

        # === CONTOUR EXTRACTION ===
        # Create 2D binary mask indicating which grid points belong to this site
        # Reshape 1D mask back to 2D grid for contour detection
        region_mask_2d = assignment_grid_2d == i
        
        # Extract contours (region boundaries) from binary mask using selected method:
        # Different contour extraction algorithms have different speed/accuracy tradeoffs
        if scipy_true:
            # scipy marching squares: smooth contours, good for analysis
            polygons = extract_contours_scipy(region_mask_2d, n_points, grid_minx, grid_miny)
        elif cv2_true:
            # OpenCV contours: fast, good edge detection
            polygons = extract_contours_cv2(region_mask_2d, n_points, grid_minx, grid_miny)
        else:
            # rasterio shapes (default): standard raster-to-vector conversion
            polygons = extract_contours_rasterio(region_mask_2d, n_points, grid_minx, grid_miny)

        if polygons:
            polygons = buffer_geometry(unary_union(polygons))
            # Optionally intersect region with buffer around site for local influence zone
            if buffering:
                point_buffer = None
                point_buffer = Point(point).buffer(row['buffer_length'])
                polygons = polygons.intersection(point_buffer).buffer(0)
            region_polygons.append({site_id_col: row[site_id_col], 'geometry': polygons})
        else:
            # No contours found for this site
            region_polygons.append({site_id_col: row[site_id_col], 'geometry': None})

    del assignment_grid_2d, assignments, mask

    # === PHASE 11: GEODATAFRAME CONVERSION & DEDUPLICATION ===
    # Convert region list to DataFrame for further processing
    region_polygons = pd.DataFrame(region_polygons)
    region_polygons = pd.merge(region_polygons, df.drop(['geometry'], axis=1), on=[site_id_col])
    region_polygons = gpd.GeoDataFrame(region_polygons, geometry='geometry', crs=crs)
    region_polygons['geometry'] = region_polygons['geometry'].map(buffer_geometry)
    region_polygons = drop_duplicates(region_polygons, site_id_col)
    
    # === PHASE 12: OVERLAP RESOLUTION ===
    # Remove overlapping areas between adjacent Voronoi regions
    # Each region intersection is assigned to larger polygon via area comparison
    non_intersecting_polygons = resolve_polygon_overlaps(region_polygons)
    region_polygons['geometry'] = non_intersecting_polygons 
    region_polygons['geometry'] = region_polygons['geometry'].map(buffer_geometry)

    # === PHASE 13: FINAL BOUNDARY CLIPPING ===
    # Clip regions to computed bounding box
    region_polygons = gpd.clip(region_polygons, actual_clipping_object)  # type: ignore[arg-type]
    region_polygons['geometry'] = region_polygons['geometry'].map(buffer_geometry)

    # Clip regions to country boundary if provided (second clipping operation)
    if country_clip is not None and not country_clip.empty:
        region_polygons = gpd.clip(region_polygons, country_clip)
        region_polygons['geometry'] = region_polygons['geometry'].map(buffer_geometry)

    # Area is measured on the final (clipped) polygons, in the metric CRS,
    # so it matches the geometry that is actually written out.
    region_polygons = region_polygons[
        region_polygons.geometry.notna() & ~region_polygons.geometry.is_empty
    ].copy()
    region_polygons['area'] = region_polygons.geometry.area

    # Keep the sites whose region survived clipping. Matching is by site id:
    # matching on the weight column would keep/drop sites that merely share a
    # weight value with a surviving region.
    point_df = df[df[site_id_col].isin(region_polygons[site_id_col])].reset_index(drop=True)

    # === PHASE 14: CRS STANDARDIZATION & RETURN ===
    # Convert all outputs to WGS84 for standard geographic format
    region_polygons = region_polygons.to_crs(4326)
    point_df = point_df.to_crs(4326)
    return region_polygons, point_df

def voronoi_worker(args):
    """
    Worker function for parallel Voronoi generation.
    
    Unpacks tuple of arguments and calls weighted_voronoi function.
    Designed for use with multiprocessing.Pool.map().
    
    Args:
        args (tuple): Packed arguments for weighted_voronoi function
        
    Returns:
        tuple: (region_polygons, point_df) from weighted_voronoi
        
    Notes:
        Catches and prints exceptions during unpacking.
    """
    try:
        (sub_df, col, country_clip, scale_weights, clipping, n_points, distance_fn,
        scipy_true, cv2_true, centroid_points, buffering, threshold, calculate_buffer_fn,
        buffer_fn_kwargs, site_id_col, size_weights) = args
        logger.debug(f"voronoi_worker: Unpacked arguments for {len(sub_df)} sites")
    except Exception as err:
        logger.error(f"voronoi_worker: Error unpacking arguments: {err}")
        raise
    return weighted_voronoi(
        sub_df, col, country_clip, scale_weights, clipping, n_points, distance_fn,
        scipy_true, cv2_true, centroid_points, buffering, threshold, calculate_buffer_fn,
        buffer_fn_kwargs, site_id_col, size_weights)

def create_weights(sub_df, sigma=3, percent_threshold=10, method='linear'):
    """Calculate normalized site weights from a detection-adjusted area proxy.

    Parameters
    ----------
    sub_df : pandas.DataFrame | geopandas.GeoDataFrame
        Input records with ``total_area`` plus detection count columns
        ``num_detection_rect`` and ``num_detection_circle``. Must already
        include a ``base_values`` column (produced by a custom function  'area_fn').
    sigma : float, default=3
        Standard-deviation multiplier used for upper clipping.
    percent_threshold : float, default=10
        Divisor used to derive the lower clipping threshold from the median.
    method : {'linear', 'logarithmic', 'square_root', 'sigmoid', 'size'}, default='linear'
        Transformation applied before normalization. ``size`` returns equal
        placeholder weights: ``weighted_voronoi`` sets sqrt(people) after
        clustering (``size_weights``).

    Returns
    -------
    pandas.DataFrame
        Copy of ``sub_df`` with added ``capacity_proxy`` and normalized
        ``weights`` columns.
    """
    df = sub_df.copy()
    if method == 'size':
        df['weights'] = 1.0 / len(df)
        return df

    # 1. Base signal must be precomputed in area_fn.
    if 'base_values' not in df.columns:
        raise KeyError("Missing 'base_values'. Run area_fn before create_weights.")
    base_values = pd.to_numeric(df['base_values'], errors='coerce')
    
    # If everything is still NaN (empty or all zeros), fallback to equal distribution
    if base_values.isnull().all() or base_values.sum() == 0:
        df['weights'] = 1.0 / len(df)
        return df

    # 2. Apply the chosen transformation method
    if method == 'logarithmic':
        df['weights'] = np.log1p(base_values)
        
    elif method == 'square_root':
        df['weights'] = np.sqrt(base_values)
        
    elif method == 'sigmoid':
        # Normalize to Z-scores so the sigmoid center (0) aligns with the data mean
        base_std = base_values.std()
        if pd.isna(base_std) or base_std == 0:
            df['weights'] = pd.Series(1.0, index=df.index, dtype=float)
        else:
            z = (base_values - base_values.mean()) / base_std
            df['weights'] = 1 / (1 + np.exp(-z))
    else: # Default to 'linear'
        df['weights'] = base_values
    raw_total = df['weights'].sum()
    if pd.isna(raw_total) or raw_total == 0:
        df['weights'] = 1.0 / len(df)
        return df
    # 3. Initial Normalization to sum=1
    df['weights'] = df['weights'] / df['weights'].sum()
    # 4. Outlier Clipping
    sub_std = df['weights'].std()
    sub_median = df['weights'].median()
    if not pd.isna(sub_std) and sub_std > 0:
        # Upper bound clipping
        upper_limit = sub_median + (sub_std * sigma)
        # Lower bound clipping
        lower_limit = sub_median / percent_threshold
        
        df['weights'] = df['weights'].clip(lower=lower_limit, upper=upper_limit)
    # 5. Final Re-normalization 
    # Necessary because clipping changes the total sum
    total_w = df['weights'].sum()
    if total_w > 0:
        df['weights'] = df['weights'] / total_w
    return df

def orchestrate_voronoi_weights(df, col, country_df, workers=12, scale_weights=False, clipping=None, n_points=100, distance_fn=default_distance_multiplicative,
                                scipy_true=False, cv2_true=False, centroid_points=False,
                                buffering=False, threshold=500, sigma=3, percent_threshold=10,
                                area_fn=None, area_fn_kwargs=None,
                                method='linear', output_path=None, overwrite=False, flush_size=None,
                                calculate_buffer_fn=calculate_buffer, buffer_fn_kwargs=None,
                                site_country_col='ISO_2', country_boundary_col='country', site_id_col='WASTE_ID'):
    """
    Orchestrate parallel Voronoi generation across data groups.
    
    Groups dataframe by column, processes each group in parallel with
    weighted Voronoi generation, then concatenates results.
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input locations with geometry, weights, and country codes.
    col : str
        Column used to group features before parallel processing.
    country_df : geopandas.GeoDataFrame
        Country boundaries used for clipping.
    workers : int, default=12
        Number of worker processes.
    scale_weights : bool, default=False
        Whether to scale weights before region generation.
    clipping : geopandas.GeoDataFrame | None, default=None
        Optional clipping geometry layer.
    n_points : int, default=100
        Grid resolution used for Voronoi generation.
    distance_fn : callable, default=default_distance_multiplicative
        Distance function used by the weighted Voronoi solver.
    scipy_true : bool, default=False
        Whether to use the SciPy contour extractor.
    cv2_true : bool, default=False
        Whether to use the OpenCV contour extractor.
    centroid_points : bool, default=False
        Whether to derive site coordinates from centroids.
    buffering : bool, default=False
        Whether to intersect generated regions with local buffers.
    threshold : float | int, default=500
        Clustering threshold applied before region generation.
    sigma : float, default=3
        Standard-deviation multiplier used during weight clipping.
    percent_threshold : float, default=10
        Divisor used to derive the lower clipping threshold from the median.
    area_fn : callable | None, default=None
        Area preprocessing function to apply before weight creation. The
        function must return the provided dataframe with a ``base_values``
        column added.
    area_fn_kwargs : dict | None, default=None
        Keyword arguments forwarded to ``area_fn``.
    method : str, default='linear'
        Weight-transformation method passed to ``create_weights``.
    output_path : str | None, default=None
        Optional output file path used for logging/traceability of this run.
    overwrite : bool, default=False
        If ``True``, existing checkpoint/output files are replaced.
    flush_size : int | None, default=None
        Number of completed worker results to buffer before flushing to temp.
    calculate_buffer_fn : callable, default=calculate_buffer
        Buffer-length function forwarded to ``weighted_voronoi``.
    buffer_fn_kwargs : dict | None, default=None
        Keyword arguments forwarded to ``calculate_buffer_fn``.
    site_country_col : str, default='ISO_2'
        Country-code column on ``df`` used to select relevant boundary clips.
    country_boundary_col : str, default='country'
        Country-code column on ``country_df`` used to match country clips.
    site_id_col : str, default='WASTE_ID'
        Site identifier column used by workers to preserve feature identity in
        Voronoi outputs.

    Returns
    -------
    tuple | bool
        When ``output_path`` is ``None``, returns
        ``(region_df_final, point_df_final)``.
        When ``output_path`` is provided, returns ``True`` on success and
        ``False`` on failure after checkpoint/rename handling.

    Notes
    -----
    ``clipping`` is normalized to the same grouping key as ``df`` so each
    worker receives only the geometry relevant to its current group.
    """
    if area_fn is None:
        raise ValueError("area_fn must be provided to orchestrate_voronoi_weights")
    if area_fn_kwargs is None:
        area_fn_kwargs = {}
    if buffer_fn_kwargs is not None:
        buffer_fn_kwargs = dict(buffer_fn_kwargs)
    # weight_method 'size': weighted_voronoi weights each site by sqrt(people)
    size_weights = method == 'size'

    # Group both df and clipping by the same column
    if output_path:
        logger.info(f"Starting orchestrate_voronoi_weights for {len(df)} sites with {workers} workers (target={output_path})")
    else:
        logger.info(f"Starting orchestrate_voronoi_weights for {len(df)} sites with {workers} workers")
    if flush_size is None:
        flush_size = max(1, workers)
    temp_output_path = os.path.join(os.path.dirname(output_path), f"temp_{os.path.basename(output_path)}") if output_path else None
    
    df = df[~df[col].isna()].reset_index(drop=True)
    df[col] = normalize_column_to_rounded_str(df[col])

    if temp_output_path and os.path.exists(temp_output_path):
        if not overwrite:
            try:
                processed = gpd.read_file(temp_output_path, columns=[col])
                if col in processed.columns:
                    processed_keys = set(normalize_column_to_rounded_str(processed[col]).astype(str))
                    before = len(df)
                    df = df[~df[col].isin(processed_keys)]
                    logger.info(
                        "Resuming from temp checkpoint %s: skipping %s already-processed groups",
                        temp_output_path,
                        before - len(df),
                    )
            except Exception as err:
                logger.warning("Could not read temp checkpoint %s (%s). Continuing without resume.", temp_output_path, err)
        else:
            try:
                os.remove(temp_output_path)
                logger.info("overwrite=True: removed existing temp checkpoint %s", temp_output_path)
            except Exception as err:
                logger.warning("Failed to remove temp checkpoint %s: %s", temp_output_path, err)

    logger.debug(f"After NaN filtering/resume: {len(df)} sites in {len(df[col].unique())} groups")

    if clipping is not None:
        clipping = clipping[~clipping[col].isna()].reset_index(drop=True)
        clipping[col] = normalize_column_to_rounded_str(clipping[col])

    df_groups = {str(k): v for k, v in df.groupby(col)}
    clip_groups = {str(k): v for k, v in clipping.groupby(col)} if clipping is not None else {}

    task_stats = {
        'generated': 0,
    }

    def iter_voronoi_args(batch_size):
        batch = []

        for key, sub_df in df_groups.items():
            # For each group, perform weight normalization and outlier clipping before Voronoi generation
            # Each group will be projected to the appropriate UTM CRS for accurate area calculation and Voronoi generation
            # the clipping geometry for each group will also be projected to the same UTM CRS for accurate clipping during Voronoi generation
            if sub_df is None or sub_df.empty or site_country_col not in sub_df:
                continue

            # Calculate area for weight normalization and outlier clipping
            sub_df = area_fn(sub_df, **area_fn_kwargs)
            sub_df = create_weights(sub_df, sigma, percent_threshold, method)

            # Estimate UTM CRS for this group based on geometry centroid for accurate distance calculations in Voronoi
            # and apply it to both the sites and the clipping geometry for this group
            utm_crs = estimate_utm_crs(sub_df)
            sub_df = sub_df.to_crs(utm_crs)
            logger.debug(f"Group {key}: {len(sub_df)} sites after area calculation and CRS conversion")
            if sub_df is None or sub_df.empty:
                continue

            # Get corresponding clipping geometry for this group if available
            sub_clip = clip_groups.get(key, None)
            if sub_clip is not None and not sub_clip.empty:
                if sub_clip.crs is None:
                    sub_clip = sub_clip.set_crs(4326)
                sub_clip = sub_clip.to_crs(utm_crs)

            # Get corresponding country boundary for this group if available
            country_iso_2 = []
            country_clip = None
            if not sub_df.empty and site_country_col in sub_df:
                iso2_series = sub_df[site_country_col].dropna()
                if not iso2_series.empty:
                    unique_vals = iso2_series.unique().tolist()
                    if unique_vals:
                        country_iso_2 = unique_vals

            if len(country_iso_2) > 0:
                country_clip = country_df[country_df[country_boundary_col].isin(country_iso_2)]
                if country_clip is not None and not country_clip.empty:
                    if country_clip.crs is None:
                        country_clip = country_clip.set_crs(4326)
                    country_clip = country_clip.to_crs(utm_crs)

            # If no country clipping is needed, pass the entire sub_df to the worker.
            # Otherwise, create separate tasks for each country within the group.
            if country_clip is None:
                task_stats['generated'] += 1
                task = (sub_df, col, country_clip, scale_weights,
                        sub_clip, n_points, distance_fn, scipy_true,
                        cv2_true, centroid_points, buffering,
                    threshold, calculate_buffer_fn, buffer_fn_kwargs, site_id_col, size_weights)
                batch.append(task)
            else:
                for country in country_iso_2:
                    country_sub_df = sub_df[sub_df[site_country_col] == country].copy().reset_index(drop=True)
                    if country_sub_df.empty:
                        continue
                    country_sub_clip = country_clip[country_clip[country_boundary_col] == country].copy().reset_index(drop=True)
                    task_stats['generated'] += 1
                    task = (country_sub_df, col, country_sub_clip,
                            scale_weights, sub_clip, n_points, distance_fn,
                            scipy_true, cv2_true, centroid_points,
                            buffering, threshold, calculate_buffer_fn, buffer_fn_kwargs, site_id_col, size_weights)
                    batch.append(task)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    region_df_all = []
    point_df_all = []

    def flush_results(force=False):
        nonlocal region_df_all, point_df_all
        if not region_df_all:
            return
        if not force and len(region_df_all) < flush_size:
            return

        region_chunk = finalize_gdf(region_df_all, df.columns)

        # Persist region chunks to temp file (overwrite decision was already
        # applied at startup — from here we always append if the file exists).
        if temp_output_path is not None:
            ensure_output_dir_for_file(temp_output_path)
            if os.path.exists(temp_output_path):
                region_chunk.to_file(temp_output_path, mode='a', driver='GPKG', index=False)
            else:
                region_chunk.to_file(temp_output_path, driver='GPKG', index=False)

        region_df_all = []
        point_df_all = []

    processed_results = 0
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for batch in iter_voronoi_args(batch_size=workers):
                futures = [executor.submit(voronoi_worker, task) for task in batch]
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        continue
                    region_df, point_df = result
                    region_df_all.append(region_df)
                    if output_path:
                        # Points are only returned in tuple mode; keeping them
                        # here just grew memory until the next flush dropped them.
                        processed_results += 1
                        flush_results()
                    else:
                        point_df_all.append(point_df)
                        processed_results += 1
        if output_path:
            flush_results(force=True)
    except Exception as err:
        logger.error("Error during Voronoi orchestration: %s", err, exc_info=True)
        return False if output_path else (finalize_gdf([], df.columns), finalize_gdf([], df.columns))

    if task_stats['generated'] == 0:
        logger.warning("No Voronoi tasks were generated")
    else:
        logger.debug(
            "Processed %s Voronoi task results (generated=%s)",
            processed_results,
            task_stats['generated'],
        )

    # Output-path mode: finalize temp checkpoint into final file and return bool.
    if output_path:
        if temp_output_path is not None and os.path.exists(temp_output_path):
            try:
                ensure_output_dir_for_file(output_path)
                if os.path.exists(output_path):
                    if overwrite:
                        os.remove(output_path) 
                    else:
                        logger.warning("Output already exists and overwrite=False: %s", output_path)
                        return True
                os.replace(temp_output_path, output_path)
                logger.info("Orchestrate Voronoi complete: processed=%s (generated=%s) -> %s",
                            processed_results, task_stats['generated'], output_path)
                return True
            except Exception as err:
                logger.error("Failed to finalize temp checkpoint %s -> %s: %s", temp_output_path, output_path, err)
                return False
        logger.warning("No temp checkpoint produced for %s", output_path)
        return False

    # Legacy tuple-return mode when no output path is provided.
    region_df_final = finalize_gdf(region_df_all, df.columns)
    point_df_final = finalize_gdf(point_df_all, df.columns)
    logger.info(
        f"Orchestrate Voronoi complete: "
        f"{len(region_df_final)} regions, "
        f"{len(point_df_final)} points"
    )
    return region_df_final, point_df_final

################################################################################
# SECTION 9: CONFIGURATION & MAIN EXECUTION
################################################################################

"""
MAIN EXECUTION WORKFLOW
=======================

This section orchestrates the complete Voronoi spatial allocation pipeline.

1. Configuration: Loads YAML config and initializes output paths
2. Data Preparation: Loads WWTP, watershed, and country boundary data
3. Approach Execution: Runs the selected approach(es)
4. Output: Saves results to GeoPackage per approach_id

WORKFLOW:
    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="create_voronoi", **overrides)                 # Load YAML configuration
  → create_output_paths(cfg)              # Create output directory structure
  → prepare_data(cfg)                     # Load input spatial data
  → run_voronoi_approach()                # Execute requested approach(es)
  → Save results to cfg['paths']['voronoi_dir']

APPROACH VARIANTS:

  Approach 0: WWTP buffer Voronoi (no watersheds)
    - Creates buffers around WWTP facilities and generates Voronoi regions
    - Weighted/unweighted and mult/add distance are read from weight_func in config

  Approach 1: Watershed-constrained Voronoi
    - Uses watershed boundaries to constrain regions
    - Weighted/unweighted and mult/add distance are read from weight_func in config

  Approach 2: City Voronoi
    - Uses major cities instead of WWTP facilities
    - Weighted/unweighted and mult/add distance are read from weight_func in config

  --only-round flag (applies to approaches 0 and 1):
    - When set, uses only round-area weights instead of all points

See pipelines.run_voronoi_approach() for parameter documentation.
"""

def _filter_requested_approaches(requested_approaches, cfg, paths_dict, only_round=False):
    """Return runnable approaches plus skip reasons for existing outputs or disabled features.

    Parameters
    ----------
    requested_approaches : list[str]
        Normalized approach identifiers requested by the caller.
    cfg : dict
        Runtime config dictionary from ``starter.load_config``.
    paths_dict : dict
        Output-path mapping from ``create_output_paths``.
    only_round : bool, default=False
        Whether approaches 0/1 should use the ``*_only_round`` output key.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``(runnable, skipped_existing, skipped_disabled)``.
    """
    runnable = []
    skipped_existing = []
    skipped_disabled = []

    for approach_id in requested_approaches:
        if approach_id == '2' and not cfg['city_voronoi']:
            skipped_disabled.append(approach_id)
            continue

        path_key = f"{approach_id}_only_round" if only_round and approach_id in {'0', '1'} else approach_id
        output_path = paths_dict['voronoi'][path_key]
        if os.path.exists(output_path) and not cfg['overwrite_existing']:
            skipped_existing.append(approach_id)
        else:
            runnable.append(approach_id)

    return runnable, skipped_existing, skipped_disabled

########################### Global Variables ####################################
if __name__ == '__main__':
    configure_logging()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run Voronoi spatial allocation approach(es) individually',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
EXAMPLES:
  # Run approach 0 (WWTP, no watersheds)
  python -m src.create_voronoi --approach 0

  # Run approaches 0 and 1 with only-round weights
  python -m src.create_voronoi --approach 0 1 --only-round

    # Run all approaches with config overrides
    python -m src.create_voronoi --level 8 --version 2 --buffer 15000 --weight-method square_root --weight-func mult --dynamic-buffering true --size-measure reported

  # Run with verbose logging
  python -m src.create_voronoi --approach 1 --verbose

  # Run all approaches (default)
  python -m src.create_voronoi
        '''
    )
    parser.add_argument('--approach', nargs='+', type=str, default=None,
                       help='Approach(es) to run: 0 (WWTP no watersheds), 1 (WWTP with watersheds), 2 (cities). Default: all')
    parser.add_argument('--only-round', action='store_true',
                       help='Use only round-area weights (default: all points)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    try:
        from .starter import add_standard_override_arguments
    except ImportError:
        from starter import add_standard_override_arguments
    add_standard_override_arguments(parser)

    args = parser.parse_args()
    
    # Validate and normalize approach names
    VALID_APPROACHES = ['0', '1', '2']
    if args.approach:
        requested = [str(a).lower() for a in args.approach]
        invalid = [a for a in requested if a not in VALID_APPROACHES]
        if invalid:
            parser.error(f"Invalid approach(es): {', '.join(invalid)}. Valid: {', '.join(VALID_APPROACHES)}")
        approaches_to_run = requested
    else:
        approaches_to_run = VALID_APPROACHES
    
    # Setup paths and logging
    try:
        from .starter import add_standard_override_arguments, load_config, parse_config_overrides
        from . import pipelines as _pipelines_module
        from .pipelines import create_output_paths, prepare_data, run_voronoi_approach, _resolve_configured_callable
    except ImportError:  # Support running as a top-level script
        from starter import add_standard_override_arguments, load_config, parse_config_overrides
        import pipelines as _pipelines_module
        from pipelines import create_output_paths, prepare_data, run_voronoi_approach, _resolve_configured_callable
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        overrides = parse_config_overrides(args=args)
    except ValueError as exc:
        parser.error(str(exc))

    cfg = load_config(script_name="create_voronoi", **overrides)
    paths_dict = create_output_paths(cfg)

    # Ensure output directory exists
    os.makedirs(cfg['paths']['voronoi_dir'], exist_ok=True)

    # Derive execution parameters from config
    only_round = args.only_round
    scale_weights = cfg['weight_func'] in {'mult', 'add'}
    country_output_col = cfg['country_output_column']
    site_id_col = cfg['site_id_column']

    requested_approaches = approaches_to_run.copy()
    logger.info(f"Running approaches: {', '.join(requested_approaches)}")
    logger.info(f"  weight_func={cfg['weight_func']!r}, weight_method={cfg['weight_method']!r}, only_round={only_round}, scale_weights={scale_weights}")
    print("=" * 80)
    print(f"VORONOI ALLOCATION - APPROACH EXECUTION")
    print(f"Requested: {', '.join(requested_approaches)}")
    print(f"weight_func={cfg['weight_func']!r}  weight_method={cfg['weight_method']!r}  only_round={only_round}")
    print("=" * 80)

    approaches_to_run, skipped_approaches, skipped_disabled_approaches = _filter_requested_approaches(
        requested_approaches,
        cfg,
        paths_dict,
        only_round=only_round,
    )
    if skipped_approaches:
        logger.info(f"Skipping completed approaches (output exists): {', '.join(skipped_approaches)}")
    if skipped_disabled_approaches:
        logger.info(
            "Skipping approach(es) disabled by config (city_voronoi=False): %s",
            ', '.join(skipped_disabled_approaches),
        )

    if not approaches_to_run:
        reasons = []
        if skipped_approaches:
            reasons.append("output files already exist")
        if skipped_disabled_approaches:
            reasons.append("approach 2 requires city_voronoi=true")
        message = "No requested approaches remain to run"
        if reasons:
            message += f" ({'; '.join(reasons)})"
        logger.info(message)
        print(message)
        sys.exit(0)

    data = _resolve_configured_callable(
        cfg['prepare_data_fn'], prepare_data, 'prepare_data_fn', _pipelines_module,
    )(cfg)

    if not isinstance(data, dict):
        raise TypeError("prepare_data_fn must return a dict with gdf_bbox, basin_gdf, and country_df")

    gdf_bbox = data['gdf_bbox']
    basin_gdf = data['basin_gdf']
    country_df = data['country_df']
    
    # Lazily-computed shared data structures
    dissolved_site_buffers = None
    dissolved_buffers_cities = None
    gdf_0 = None
    gdf_4 = None
    gdf_2 = None
    basin_gdf_2 = None
    
    # Execute requested approaches
    for approach_id in approaches_to_run:
        try:
            # Path key encodes the only_round variant
            path_key = f"{approach_id}_only_round" if only_round and approach_id in {'0', '1'} else approach_id

            # === APPROACH 0: WWTP buffer Voronoi (no watersheds) ===
            if approach_id == '0':
                logger.info("Starting Approach 0: WWTP buffer Voronoi")
                if dissolved_site_buffers is None:
                    dissolved_site_buffers = orchestrate_overlaps(gdf_bbox, cfg['max_workers'], 
                                                                 paths_dict['buffers']['WWTP'], cfg['buffer'],
                                                                 country_col=country_output_col)
                    dissolved_site_buffers = drop_duplicates(drop_duplicates(dissolved_site_buffers, site_id_col), 'geometry')
                    dissolved_site_buffers['buffer_id'] = np.arange(len(dissolved_site_buffers))
                
                if gdf_0 is None:
                    gdf_0 = gdf_bbox.copy()
                    gdf_0 = intersect_with_polygon_sindex(gdf_0, dissolved_site_buffers, 'buffer_id', 
                                                       concurrency=cfg['sindex_concurrency'])
                    gdf_0 = drop_duplicates(drop_duplicates(gdf_0, site_id_col), 'geometry')
                
                run_voronoi_approach('0', gdf_0, dissolved_site_buffers, country_df, cfg, cfg['distance_fn'],
                                    paths_dict['voronoi'][path_key], buffer_id_col='buffer_id',
                                    scale_weights=scale_weights, only_round=only_round, buffering=False,
                                    method=cfg['weight_method'])
                print("✓ Approach 0 completed")
            
            # === APPROACH 1: Watershed-constrained Voronoi ===
            elif approach_id == '1':
                logger.info("Starting Approach 1: Watershed-constrained Voronoi")
                if gdf_2 is None:
                    gdf_2 = gdf_bbox.copy()
                    gdf_2['buffer_id'] = gdf_2['HYBAS_ID']
                    basin_gdf_2 = basin_gdf.copy()
                    basin_gdf_2['buffer_id'] = basin_gdf_2['HYBAS_ID']
                
                run_voronoi_approach('1', gdf_2, basin_gdf_2, country_df, cfg, cfg['distance_fn'],
                                    paths_dict['voronoi'][path_key], buffer_id_col='buffer_id',
                                    scale_weights=scale_weights, only_round=only_round, buffering=True,
                                    method=cfg['weight_method'])
                print("✓ Approach 1 completed")
            
            # === APPROACH 2: City Voronoi ===
            elif approach_id == '2':
                logger.info("Starting Approach 2: City Voronoi")
                if dissolved_buffers_cities is None:
                    df_cities = pd.read_csv(cfg['paths']['cities'])
                    df_cities = gpd.GeoDataFrame(df_cities, 
                                               geometry=gpd.GeoSeries([from_wkt(geom) if isinstance(geom, str) else geom 
                                                                        for geom in df_cities['geometry']]), 
                                               crs='epsg:4326')
                    df_cities['geometry'] = df_cities['geometry'].map(buffer_geometry)
                    
                    country_output_col = cfg['country_output_column']
                    country_boundary_col = cfg['country_boundary_column']
                    if country_output_col not in df_cities.columns:
                        if not os.path.exists(cfg['paths']['overture']):
                            download_overture_maps(cfg['paths']['overture_s3_url'], cfg['paths']['overture'])
                        df_cities = intersects_with_country_db(
                            df_cities,
                            cfg['paths']['overture'],
                            polygon_country_col=country_boundary_col,
                            output_country_col=country_output_col,
                        )
                    
                    dissolved_buffers_cities = orchestrate_overlaps(df_cities, cfg['max_workers'], 
                                                                   paths_dict['buffers']['city'], cfg['buffer'],
                                                                   country_col=country_output_col)
                    dissolved_buffers_cities = drop_duplicates(dissolved_buffers_cities, 'geometry')
                    dissolved_buffers_cities['geometry'] = dissolved_buffers_cities['geometry'].map(buffer_geometry)
                    dissolved_buffers_cities['buffer_id'] = np.arange(len(dissolved_buffers_cities))
                    
                    gdf_4 = gdf_bbox.copy()
                    gdf_4 = intersect_with_polygons_parallelized(
                        gdf_4,
                        dissolved_buffers_cities,
                        ['buffer_id'],
                        use_duckdb=cfg['duckdb_cond'],
                        max_workers=cfg['max_workers'],
                        df_join_col=country_output_col,
                        polygon_join_col=country_output_col,
                    )
                    gdf_4 = drop_duplicates(drop_duplicates(gdf_4, site_id_col), 'geometry')
                    gdf_4['geometry'] = gdf_4['geometry'].map(buffer_geometry)
                
                run_voronoi_approach('2', gdf_4, dissolved_buffers_cities, country_df, cfg, cfg['distance_fn'],
                                    paths_dict['voronoi']['2'], buffer_id_col='buffer_id',
                                    scale_weights=scale_weights, only_round=only_round, buffering=False,
                                    method=cfg['weight_method'])
                print("✓ Approach 2 completed")
        
        except Exception as e:
            logger.error(f"Error executing approach {approach_id}: {e}", exc_info=True)
            print(f"✗ Approach {approach_id} FAILED: {e}")
            sys.exit(1)
    
    print("=" * 80)
    print(f"SUCCESS: All requested approaches completed ({', '.join(approaches_to_run)})")
    print("=" * 80)
    logger.info(f"Voronoi generation completed for approaches: {', '.join(approaches_to_run)}")
