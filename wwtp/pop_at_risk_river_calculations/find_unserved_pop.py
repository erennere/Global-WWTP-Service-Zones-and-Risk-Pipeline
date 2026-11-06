"""Filter non-served population polygons above a threshold and export to GeoPackage.

This script reads a CSV of non-served polygons (WKT geometry), filters rows by
`pop_sum > threshold`, and writes the result to a GPKG file.
"""
import os
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import pandas as pd
from shapely import from_wkt, to_wkt

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides

from .. import provenance
from ..geo_utils import buffer_meters, ensure_duckdb_spatial
from ..utils import configure_logging, duckdb_connection, ensure_output_dir_for_file
from .find_pop_in_danger_pop import finding_tiles

logger = logging.getLogger(__name__)

def read_urban_areas(filepath):
    try:
        return gpd.read_file(filepath)
    except Exception as e:
        logger.error("Failed to read urban areas file: %s", e)
        raise Exception(f"Failed to read urban areas file: {filepath}") from e
    
def add_bbox(gdf, zoom_level, max_workers, is_parallel):
    def chunk_find_bbox(chunk, zoom_level):
        chunk['tiles'] = chunk['geometry'].map(lambda geom: finding_tiles(geom, zoom_level))
        return chunk

    if is_parallel:
        gdf['indx'] = range(len(gdf))
        chunks = [gdf[['indx', 'geometry']].iloc[i:i + max_workers] for i in range(0, len(gdf), max_workers)]
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(chunk_find_bbox, chunk, zoom_level) for chunk in chunks]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error("Error processing chunk: %s", e)
                    raise

        results = pd.concat(results).drop(columns=['geometry'])
        gdf = gdf.merge(results, on='indx', how='left').drop(columns=['indx'])
    else:
        gdf['tiles'] = gdf['geometry'].map(lambda geom: finding_tiles(geom, zoom_level))
    return gdf

def add_buffers_to_WWTP(gdf, wwtp_filepath, wwtp_buffer, zoom_level,
                        max_workers, is_parallel, wwtp_country_col):
    """Keep the non-served polygons that lie within ``wwtp_buffer`` metres of a plant.

    Each plant is buffered by ``wwtp_buffer`` metres in its own UTM zone and
    the polygons that intersect any buffer are returned once each, with
    their ORIGINAL geometry and attributes.

    The previous DuckDB version buffered both sides in EPSG:3857 (so the test
    was 2 x ``wwtp_buffer``, shrunk by cos(latitude)), returned the polygons
    with the 50 km buffer still applied, missed matches across zoom-8 tile
    edges, and wrote a polygon once per country of the nearby plants, which
    double-counted its population.

    ``zoom_level``, ``max_workers`` and ``is_parallel`` are accepted for call
    compatibility; the spatial join does not need tiles.
    """
    logger.info(
        "Starting add_buffers_to_WWTP: input_rows=%s wwtp_path=%s wwtp_buffer=%s m",
        len(gdf), wwtp_filepath, wwtp_buffer,
    )
    wwtp_gdf = gpd.read_file(wwtp_filepath)
    if 'geometry' not in wwtp_gdf.columns:
        raise ValueError(f"WWTP file has no geometry column: {wwtp_filepath}")
    if wwtp_country_col not in wwtp_gdf.columns:
        raise KeyError(f"WWTP file is missing configured country column '{wwtp_country_col}': {wwtp_filepath}")

    before = len(wwtp_gdf)
    wwtp_gdf = wwtp_gdf[wwtp_gdf.geometry.notna() & wwtp_gdf.geometry.is_valid & ~wwtp_gdf.geometry.is_empty]
    logger.info("WWTP rows kept for buffering: %s of %s", len(wwtp_gdf), before)
    if gdf.empty or wwtp_gdf.empty:
        return gdf.iloc[0:0].copy() if wwtp_gdf.empty else gdf

    if gdf.crs is None:
        raise ValueError("Non-served polygons need a CRS")
    wwtp_zones = gpd.GeoDataFrame(geometry=buffer_meters(wwtp_gdf, float(wwtp_buffer)), crs=wwtp_gdf.crs).to_crs(gdf.crs)

    probe = gpd.GeoDataFrame(geometry=gdf.geometry.values, crs=gdf.crs)
    probe['_row'] = range(len(probe))
    hits = gpd.sjoin(probe, wwtp_zones, how='inner', predicate='intersects')
    keep = sorted(set(hits['_row'].tolist()))
    result = gdf.iloc[keep].copy()
    logger.info("Non-served polygons within %s m of a plant: %s of %s", wwtp_buffer, len(result), len(gdf))
    return result


def orchestrate_urban_intersection(unserved_gdf, urban_filepath, zoom_level=8, max_workers=32, is_parallel=True,
                                   urban_buffer=10000, tolerance=0.0001):
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")

    if unserved_gdf.empty:
        return unserved_gdf

    unserved_gdf['idx'] = range(len(unserved_gdf))

    unserved_gdf = add_bbox(unserved_gdf, zoom_level=zoom_level, max_workers=max_workers, is_parallel=is_parallel)
    urban_areas_gdf = read_urban_areas(urban_filepath)
    # Buffer in metres in each area's UTM zone (EPSG:3857 shrank it by cos(lat)).
    urban_areas_gdf = urban_areas_gdf.to_crs(4326)
    urban_areas_gdf['geometry'] = buffer_meters(urban_areas_gdf, float(urban_buffer))
    urban_areas_gdf = add_bbox(urban_areas_gdf, zoom_level=zoom_level, max_workers=max_workers, is_parallel=False)

    urban_areas_gdf = urban_areas_gdf.to_crs(3857)
    urban_areas_gdf['geometry'] = urban_areas_gdf['geometry'].map(to_wkt)
    unserved_gdf = unserved_gdf.to_crs(3857)
    unserved_gdf['geometry'] = unserved_gdf['geometry'].map(to_wkt)

    duckdb_query = f"""
        WITH urban_areas AS (
            SELECT *
            FROM urban_areas_gdf
        ),
        unserved AS (
            SELECT *, UNNEST(tiles) AS tile
            FROM unserved_gdf
        ),
        clipped AS (
            SELECT
                a.idx,
                ST_Intersection(
                    ST_GeomFromText(a.geometry),
                    ST_GeomFromText(b.geometry)
                ) AS geom
            FROM unserved a
            INNER JOIN urban_areas b
                ON list_contains(b.tiles, a.tile)
               AND ST_Intersects(
                    ST_GeomFromText(a.geometry),
                    ST_GeomFromText(b.geometry)
               )
        ),
        -- One tile per polygon, the lowest-numbered one: reproducible, and DuckDB 2
        -- rejects RANDOM() in a window ORDER BY.
        first_tile AS (
            SELECT idx, tile
            FROM (
                SELECT
                    idx,
                    tile,
                    ROW_NUMBER() OVER (PARTITION BY idx ORDER BY tile) AS row_num
                FROM unserved
            )
            WHERE row_num = 1
        ),
        aggregated AS (
            SELECT
                idx,
                ST_AsText(
                    ST_SimplifyPreserveTopology(
                        ST_Union_Agg(geom),
                        {tolerance}
                    )
                ) AS geometry
            FROM clipped
            GROUP BY idx
        )
        SELECT u.* REPLACE (a.geometry AS geometry), t.tile
        FROM unserved_gdf u
        INNER JOIN aggregated a
            ON u.idx = a.idx
        INNER JOIN first_tile t
            ON u.idx = t.idx
    """

    try:
        with duckdb_connection() as conn:
            ensure_duckdb_spatial(conn)
            df = conn.execute(duckdb_query).df()
            if df.empty:
                empty_columns = [col for col in unserved_gdf.columns if col not in {'idx', 'tiles'}] + ['tile']
                return gpd.GeoDataFrame(columns=empty_columns, geometry='geometry', crs=4326)
            df = df.sort_values('idx').drop(columns=['tiles', 'idx'])
            df['geometry'] = df['geometry'].map(from_wkt)
            return gpd.GeoDataFrame(df, geometry='geometry', crs=3857).to_crs(4326)
    except Exception as e:
        logger.error("Error during urban intersection: %s", e)
        raise Exception("Error during urban intersection") from e

def parse_args():
    """Parse optional config override flags for this step."""
    parser = argparse.ArgumentParser(
        description="Filter non-served polygons above threshold and export to GPKG."
    )
    add_standard_override_arguments(parser)
    return parser.parse_args()

def create_unserved_pop(filepath, threshold, output_filepath,
                        urban_filepath, wwtp_filepath,
                        max_workers=32,
                        is_parallel=True,
                        zoom_level=8,
                        urban_buffer=10000,
                        wwtp_buffer=10000,
                        wwtp_country_col='ISO_2',
                        tolerance=0.0001):
    """Create a GeoPackage with non-served polygons above a population threshold.

    Parameters
    ----------
    filepath : str
        Input CSV path containing at least `pop_sum` and `geometry` columns.
    threshold : int
        Minimum population sum to keep a polygon.
    output_filepath : str
        Output GPKG path.

    Returns
    -------
    int
        Number of output rows written.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input CSV not found: {filepath}")

    try:
        threshold_value = int(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold must be an integer-like value, got: {threshold}") from exc
    if threshold_value < 0:
        raise ValueError("threshold must be >= 0")

    logger.info("Filtering non-served polygons from %s with threshold=%s", filepath, threshold_value)

    query = f"""
    SELECT *
    FROM read_csv('{filepath}',
        header=True,
        all_varchar=True,
        max_line_size=10000000
        )
    WHERE TRY_CAST(pop_sum AS BIGINT) > {threshold_value}
    """
    # duckdb_connection owns the scratch database: unique name, always removed.
    with duckdb_connection() as conn:
        df = conn.sql(query).df()

    if df.empty:
        logger.warning("No rows matched threshold=%s. Writing empty output file.", threshold)
    if "geometry" not in df.columns:
        raise ValueError("Input CSV is missing required 'geometry' column")

    df['geometry'] = df.geometry.apply(from_wkt)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
    df = orchestrate_urban_intersection(
        df,
        urban_filepath,
        zoom_level=zoom_level,
        max_workers=max_workers,
        is_parallel=is_parallel,
        urban_buffer=urban_buffer,
        tolerance=tolerance
    )
    df = add_buffers_to_WWTP(df, 
                            wwtp_filepath=wwtp_filepath,
                            wwtp_buffer=wwtp_buffer,
                            zoom_level=zoom_level,
                            max_workers=max_workers,
                            is_parallel=is_parallel,
                            wwtp_country_col=wwtp_country_col)

    ensure_output_dir_for_file(output_filepath)
    df.to_file(output_filepath, driver='GPKG', index=False)
    logger.info("Wrote %s polygons to %s", len(df), output_filepath)
    return len(df)

def main():
    """Load config and run threshold-based non-served population extraction."""
    args = parse_args()
    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="find_unserved_pop", **overrides)
    provenance.write_run_manifest(cfg, "find_unserved_pop")

    threshold = int(cfg["threshold_value"])
    max_workers = int(cfg["max_workers"])
    is_parallel = str(cfg["is_parallel"]).strip().lower() in {"1", "true", "yes", "y", "on"}
    zoom_level = int(cfg["zoom_level"])
    urban_buffer = int(cfg["urban_buffer"])
    tolerance = float(cfg["tolerance"])
    wwtp_buffer = int(cfg["wwtp_buffer"])
    wwtp_country_col = cfg["country_output_column"]
    wwtp_filepath = os.path.abspath(cfg['paths']['annotated_all_filepath'])
    
    non_served_outpath = os.path.abspath(cfg["paths"]["non_served_outpath"].replace('.gpkg', '.csv'))
    non_served_above_threshold_outpath = os.path.abspath(cfg["paths"]["non_served_above_threshold_outpath"])
    urban_filepath = os.path.abspath(cfg["paths"]["urban_areas_filepath"])
    rows_written = create_unserved_pop(
        non_served_outpath,
        threshold,
        non_served_above_threshold_outpath,
        urban_filepath,
        wwtp_filepath, 
        max_workers=max_workers,
        is_parallel=is_parallel,
        zoom_level=zoom_level,
        urban_buffer=urban_buffer,
        wwtp_buffer = wwtp_buffer,
        wwtp_country_col=wwtp_country_col,
        tolerance=tolerance
    )
    logger.info("Completed non-served population extraction. rows_written=%s", rows_written)

if __name__ == "__main__":
    configure_logging()
    main()