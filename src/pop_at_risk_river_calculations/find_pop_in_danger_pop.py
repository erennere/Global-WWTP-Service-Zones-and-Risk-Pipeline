"""Aggregate population-at-risk outputs onto web-map tiles.

The script assigns impact polygons to XYZ tiles, intersects them with country
metadata and raster-derived population sums, and exports a tiled summary layer.
"""

import argparse
import os
from glob import glob
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import mercantile

try:
    from ..add_pop import intersect_all_files
    from ..service_areas import intersects_with_country_db
    from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
    from .. import provenance
    from ..utils import configure_logging, ensure_output_dir_for_file
except ImportError:
    from src.add_pop import intersect_all_files
    from src.service_areas import intersects_with_country_db
    from src.starter import add_standard_override_arguments, load_config, parse_config_overrides
    from src import provenance
    from src.utils import configure_logging, ensure_output_dir_for_file

# Configure logging
logger = logging.getLogger(__name__)


def _extract_radius_from_path(path):
    """Extract trailing numeric radius from a file path ending with '_<radius>.gpkg'."""
    match = re.search(r'_(\d+)\.gpkg$', os.path.basename(path))
    return int(match.group(1)) if match else None


def _zonal_sum_columns(df):
    """Return zonal-sum columns present in the frame."""
    return [col for col in df.columns if col.endswith('_zonal_sum')]


def _log_stage_frame_stats(stage, frame):
    """Log shape, tile counts, and zonal metric availability for a stage."""
    rows = len(frame)
    cols = len(frame.columns)
    tile_count = int(frame['tile'].nunique()) if 'tile' in frame.columns else 0
    zonal_cols = _zonal_sum_columns(frame)
    logger.info(
        "%s stats: rows=%s cols=%s unique_tiles=%s zonal_cols=%s",
        stage,
        rows,
        cols,
        tile_count,
        len(zonal_cols),
    )


def _log_and_validate_no_nan(stage, frame, strict=True):
    """Log NaN diagnostics for zonal columns and optionally fail fast."""
    zonal_cols = _zonal_sum_columns(frame)
    if not zonal_cols:
        logger.info("%s: no zonal sum columns present.", stage)
        return

    nan_counts = frame[zonal_cols].isna().sum()
    total_nan = int(nan_counts.sum())
    if total_nan == 0:
        logger.info("%s: no NaN detected across %s zonal columns.", stage, len(zonal_cols))
        return

    problematic = nan_counts[nan_counts > 0].sort_values(ascending=False)
    logger.error(
        "%s: detected %s NaN cells across zonal columns. Breakdown: %s",
        stage,
        total_nan,
        problematic.to_dict(),
    )

    first_col = problematic.index[0]
    sample_cols = ['tile', first_col] if 'tile' in frame.columns else [first_col]
    sample = frame.loc[frame[first_col].isna(), sample_cols].head(20)
    logger.error("%s: sample rows with NaN in %s:\n%s", stage, first_col, sample.to_string(index=False))

    if strict:
        raise ValueError(
            f"{stage}: NaN values detected in zonal sums; this indicates upstream processing errors."
        )


def _log_zero_value_diagnostics(stage, frame):
    """Log zero-value diagnostics for zonal columns so zero-producing paths are visible."""
    zonal_cols = _zonal_sum_columns(frame)
    if not zonal_cols:
        return

    zero_counts = {}
    for col in zonal_cols:
        vals = pd.to_numeric(frame[col], errors='coerce')
        zero_counts[col] = int((vals == 0).sum())

    total_zeros = int(sum(zero_counts.values()))
    logger.info(
        "%s: zero-value cells across zonal columns=%s. Breakdown: %s",
        stage,
        total_zeros,
        zero_counts,
    )

    if total_zeros > 0:
        dominant_col = sorted(zero_counts.items(), key=lambda item: item[1], reverse=True)[0][0]
        sample_cols = ['tile', dominant_col] if 'tile' in frame.columns else [dominant_col]
        sample = frame.loc[pd.to_numeric(frame[dominant_col], errors='coerce') == 0, sample_cols].head(20)
        logger.warning(
            "%s: sample rows with zero in %s:\n%s",
            stage,
            dominant_col,
            sample.to_string(index=False),
        )

def finding_tiles(polygon, zoom_level):
    """Find all XYZ tiles that intersect a polygon.
    
    Args:
        polygon: Shapely polygon object.
        zoom_level: Zoom level for tile calculation.
    
    Returns:
        List of strings in format 'x-y-z' for all intersecting tiles.
    """
    bbox = polygon.bounds
    tiles = [f'{int(tile.x)}-{int(tile.y)}-{int(tile.z)}' for tile in mercantile.tiles(*bbox, zooms=zoom_level)]
    logger.debug(f"Urban area intersects {len(tiles)} tiles at zoom level {zoom_level}")
    return tiles

def find_bbox(tile):
    """Find the bounding box of a tile.
    Args:
        tile: Mercantile tile object in 'x-y-z' string format.
    Returns:
        Bounding box of the tile.
    """
    return box(*mercantile.bounds(*map(int, tile.split('-'))))

def find_tiles_in_a_country(country_polygon, country, country_id_col, zoom_level):
    """Find all XYZ tiles that intersect a country's polygon.
    
    Args:
        country_polygon: Shapely polygon object representing the country.
        zoom_level: Zoom level for tile calculation.
    
    Returns:
        GeoDataFrame with tiles and their corresponding geometries.
    """
    tiles = finding_tiles(country_polygon, zoom_level)
    bboxes = map(find_bbox, tiles)
    logger.info(f"Country intersects {len(tiles)} tiles at zoom level {zoom_level}")
    gdf = gpd.GeoDataFrame({'tile': tiles, 'geometry': bboxes}, crs=4326)
    gdf = gdf.clip(country_polygon)
    gdf[country_id_col] = country
    return gdf

def find_tiles_in_countries(countries_gdf, zoom_level, country_id_col, max_workers=4):
    """Find all XYZ tiles that intersect multiple countries.
    
    Args:
        countries_gdf: GeoDataFrame with country geometries and ISO_2 codes.
        zoom_level: Zoom level for tile calculation.
        country_id_col: Column name in countries_gdf that contains the country identifier.
    
    Returns:
        GeoDataFrame with tiles and their corresponding geometries for all countries.
    """
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")

    all_tiles = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(find_tiles_in_a_country, row['geometry'], row[country_id_col], country_id_col, zoom_level) for _, row in countries_gdf.iterrows()]
        for future in as_completed(futures):
            all_tiles.append(future.result())
    if all_tiles:
        logger.info(f"Total tiles found across all countries: {len(all_tiles)}")
        return pd.concat(all_tiles, ignore_index=True)
    else:
        logger.warning("No tiles found for any country.")
        return pd.DataFrame(columns=['tile', 'geometry', country_id_col])
    
def assign_tile_to_df_worker(df, zoom_level):
    """Assign one dataframe chunk to intersecting tiles and clip each row to tile bounds."""
    df['tile'] = df['geometry'].apply(lambda geom: finding_tiles(geom, zoom_level))
    df = df.explode('tile', ignore_index=True)
    df['geometry'] = df.apply(lambda row: row['geometry'].intersection(find_bbox(row['tile'])) if pd.notna(row['tile']) else row['geometry'], axis=1)
    return df
    
def assign_tile_to_df(df, zoom_level, max_workers=4):
    """Assign all rows to tiles in parallel and concatenate the exploded chunks."""
    if df.empty:
        return df

    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")

    r = max(1, len(df) // max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(assign_tile_to_df_worker, df.iloc[i: min(i + r, len(df))].copy(), zoom_level) for i in range(0, len(df), r)]
        results = [future.result() for future in as_completed(futures)]
    return pd.concat(results, ignore_index=True)

def group_tile_population_sums(df):
    """Aggregate all zonal-sum columns to one row per tile."""
    zonal_sum_cols = [col for col in df.columns if col.endswith('_zonal_sum')]
    if 'tile' not in df.columns or not zonal_sum_cols:
        return df

    # Keep all-NaN groups as NaN instead of coercing them to 0.0.
    grouped = df.groupby('tile', as_index=False)[zonal_sum_cols].sum(min_count=1)
    return grouped


def rename_cols(df, radius):
    """Prefix non-geometry result columns with the source impact radius."""
    return df.rename({col: f'{radius}_{col}' for col in df.columns if col not in ['tile', 'geometry']}, axis=1) 

def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run find_pop_in_danger_pop.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Load impact polygons, tile them, attach population sums, and export parquet."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="find_pop_in_danger_pop", **overrides)
    provenance.write_run_manifest(cfg, "find_pop_in_danger_pop")
    zoom_level = int(cfg['zoom_level'])
    max_workers = int(cfg['annotations']['max_workers'])
    tif_dir = cfg['paths']['pop_tif_dir']
    country_boundary_col = cfg['country_boundary_column']
    country_output_col = cfg['country_output_column']

    input_pattern = cfg['paths']['impact_pop_polygons_outpath'].replace('.gpkg', '_*.gpkg')
    input_files = glob(input_pattern)
    input_files = sorted(input_files, key=lambda path: (_extract_radius_from_path(path) is None, _extract_radius_from_path(path) or 0, path))
    results = None

    logger.info("find_pop_in_danger_pop config: zoom_level=%s max_workers=%s tif_dir=%s", zoom_level, max_workers, tif_dir)
    logger.info("Impact polygon search pattern: %s", input_pattern)
    logger.info("Matched %s impact polygon files.", len(input_files))

    if not input_files:
        logger.warning("No impact polygon files matched pattern %s. Writing empty output.", input_pattern)

    for input_file in input_files:
        radius_int = _extract_radius_from_path(input_file)
        if radius_int is None:
            logger.warning("Skipping impact polygon file with unrecognized radius suffix: %s", input_file)
            continue
        radius = str(radius_int)
        logger.info(f"Processing impact polygons for radius {radius} from file {input_file}")

        impact_polygons = gpd.read_file(input_file)
        _log_stage_frame_stats(f"radius={radius} read_file", impact_polygons)
        impact_polygons = assign_tile_to_df(impact_polygons, zoom_level, max_workers)
        _log_stage_frame_stats(f"radius={radius} assign_tile_to_df", impact_polygons)
        impact_polygons = intersects_with_country_db(
            impact_polygons,
            cfg['paths']['overture'],
            polygon_country_col=country_boundary_col,
            output_country_col=country_output_col,
        )
        _log_stage_frame_stats(f"radius={radius} intersects_with_country_db", impact_polygons)
        ensure_output_dir_for_file('impact_polygons_tiled.gpkg')
        #impact_polygons.to_file('impact_polygons_tiled.gpkg', index=False, driver='GPKG')
        intersect_workers = max(1, int(max_workers / 8))
        logger.info("radius=%s intersect_all_files workers=%s", radius, intersect_workers)
        impact_polygons = intersect_all_files(
            impact_polygons,
            tif_dir,
            intersect_workers,
            all_years=False,
            country_col=country_output_col,
        )
        _log_stage_frame_stats(f"radius={radius} intersect_all_files", impact_polygons)
        _log_and_validate_no_nan(f"radius={radius} post_intersect", impact_polygons, strict=True)
        _log_zero_value_diagnostics(f"radius={radius} post_intersect", impact_polygons)
        tile_groups = group_tile_population_sums(impact_polygons)
        _log_stage_frame_stats(f"radius={radius} grouped_tile_sums", tile_groups)
        _log_and_validate_no_nan(f"radius={radius} grouped_tile_sums", tile_groups, strict=True)
        _log_zero_value_diagnostics(f"radius={radius} grouped_tile_sums", tile_groups)
        del impact_polygons

        if 'tile' not in tile_groups.columns:
            logger.warning("No 'tile' column found after grouping; writing empty output.")
            tile_groups_gdf = gpd.GeoDataFrame(columns=['tile', 'geometry'], geometry='geometry', crs=4326)
        else:
            geoms = tile_groups['tile'].apply(find_bbox)
            tile_groups['geometry'] = geoms
            tile_groups_gdf = gpd.GeoDataFrame(tile_groups, geometry='geometry', crs=4326)

        if results is None:
            tile_groups_gdf = rename_cols(tile_groups_gdf, radius)
            results = tile_groups_gdf
        else:
            tile_groups_gdf = rename_cols(tile_groups_gdf, radius)
            results = pd.merge(results, tile_groups_gdf, on='tile', how='outer', suffixes=('', '_new'))
            if 'geometry_new' in results.columns:
                results['geometry'] = results['geometry'].combine_first(results['geometry_new'])
                results = results.drop(columns=['geometry_new'])

        _log_stage_frame_stats(f"radius={radius} merged_results", results)

    if results is None:
        results = gpd.GeoDataFrame(columns=['tile', 'geometry'], geometry='geometry', crs=4326)
    else:
        _log_zero_value_diagnostics("final_merged_output", results)
        results = gpd.GeoDataFrame(results, geometry=results['geometry'], crs=4326)
        _log_stage_frame_stats("final_output", results)
    ensure_output_dir_for_file(cfg['paths']['pop_at_risk_output_filepath'])
    results.to_parquet(cfg['paths']['pop_at_risk_output_filepath'], engine='pyarrow', index=False)
    logger.info("Wrote pop-at-risk parquet: %s", cfg['paths']['pop_at_risk_output_filepath'])

if __name__ == '__main__':
    configure_logging()
    main()
    







    


