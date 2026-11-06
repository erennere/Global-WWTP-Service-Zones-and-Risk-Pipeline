"""Spatial joins of sites with basins, buffers and country boundaries (GeoPandas, DuckDB).

Part of the service-area stage (``python -m wwtp.create_voronoi``).
"""
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point, Polygon, LineString, MultiPolygon, MultiLineString, from_wkt
import shapely.affinity
import shapely
from ..geo_utils import buffer_geometry, ensure_duckdb_spatial, estimate_utm_epsg_for_geom, safe_to_wkt
from ..utils import duckdb_connection, quote_sql_identifier

logger = logging.getLogger(__name__)


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
