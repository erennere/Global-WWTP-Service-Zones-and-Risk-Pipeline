#!/usr/bin/env python
"""
Identify unconnected industrial areas not served by any WWTP.

This script:
1. Loads merged industrial land use data and WWTP locations.
2. Filters WTTPs to those with industrial usage.
3. Creates Voronoi service areas for those WTTPs.
4. Identifies industrial areas NOT overlapping any WWTP service area.
5. Saves unconnected areas to GeoPackage.

Usage:
    python -m src.industrial_analysis.find_unconnected_industrial_areas [level] [version] [buffer] [weight_method] [weight_func] [dynamic_buffering] [dynamic_buffer_k]
"""

import argparse
import sys
import os
import logging
from typing import Optional

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import geopandas as gpd
import pandas as pd

try:
    from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
    from ..utils import configure_logging, default_cpu_workers, select_industrial_categories
    from .. import pipelines as _pipelines_module
    from ..pipelines import (
        run_voronoi_approach,
        prepare_data,
        create_output_paths,
        _resolve_configured_callable,
    )
    from ..create_voronoi import intersect_with_polygon_sindex, orchestrate_overlaps, drop_duplicates
except ImportError:
    from src.starter import add_standard_override_arguments, load_config, parse_config_overrides
    from src.utils import configure_logging, default_cpu_workers, select_industrial_categories
    import src.pipelines as _pipelines_module
    from src.pipelines import (
        run_voronoi_approach,
        prepare_data,
        create_output_paths,
        _resolve_configured_callable,
    )
    from src.create_voronoi import intersect_with_polygon_sindex, orchestrate_overlaps, drop_duplicates

logger = logging.getLogger(__name__)


def load_industrial_areas(cfg: dict) -> Optional[gpd.GeoDataFrame]:
    """Load merged industrial areas from GeoPackage."""
    path = cfg['paths']['industrial_merged_filepath']
    
    if not os.path.exists(path):
        logger.warning(f"Industrial areas file not found: {path}")
        return None
    
    logger.info(f"Loading industrial areas from {path}...")
    gdf = gpd.read_file(path, driver='GPKG')
    logger.info(f"Loaded {len(gdf)} industrial features")
    
    return gdf


def load_wwtps(cfg: dict, approach_id: str) -> gpd.GeoDataFrame:
    """Load WWTP data and add basin information if missing."""
    path = cfg['paths']['annotated_all_filepath']
    logger.info(f"Loading WTTPs from {path}...")
    
    gdf = gpd.read_file(path, driver='GPKG')
    logger.info(f"Loaded {len(gdf)} WTTPs")
    
    basin_col = cfg['basin_column_name']
    if approach_id == '1' and (basin_col not in gdf.columns or gdf[basin_col].isna().all()):
        logger.info(f"Basin information '{basin_col}' missing; attempting to add from watershed intersection...")
        basin_gdf = gpd.read_file(
            cfg['paths']['watershed'],
            driver='GPKG'
        )

        if basin_col not in basin_gdf.columns:
            raise KeyError(f"Configured basin column '{basin_col}' not found in basin dataset.")
        
        if gdf.crs != basin_gdf.crs:
            basin_gdf = basin_gdf.to_crs(gdf.crs)
        
        # Use the same basin intersection method as industrial vectorization.
        gdf = intersect_with_polygon_sindex(
            gdf,
            basin_gdf[[basin_col, 'geometry']].copy(),
            basin_col,
            concurrency=cfg['sindex_concurrency'],
        )
        logger.info(f"Added basin info to {len(gdf[gdf[basin_col].notna()])} WTTPs")
    
    return gdf


def filter_industrial_wwtps(cfg: dict, wwtps_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter WTTPs to those with industrial usage."""
    industrial_categories = cfg['industrial_category_numbers']
    mix_use_categories = cfg['mix_use_categories']
    if not industrial_categories:
        logger.warning("No industrial categories configured; using all WTTPs")
        return wwtps_gdf
    
    logger.info(f"Filtering WTTPs by industrial categories: {industrial_categories}")
    return select_industrial_categories(
        wwtps_gdf, industrial_categories, mix_use_categories, keep=True, logger=logger
    )


def run_voronoi_for_wwtps(
    cfg: dict,
    approach_id: str,
    wwtps_gdf: gpd.GeoDataFrame,
    basin_gdf: gpd.GeoDataFrame,
    country_gdf: gpd.GeoDataFrame,
    paths_dict: dict,
    output_path: str,
    only_round: bool,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create Voronoi service areas for WTTPs using default approach.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    wwtps_gdf : geopandas.GeoDataFrame
        Filtered WWTP locations.
    country_gdf : geopandas.GeoDataFrame
        Country boundary polygons.
    
    Returns
    -------
    geopandas.GeoDataFrame or None
        Voronoi polygons, or None if orchestration failed.
    """
    logger.info(f"Running Voronoi diagram orchestration for filtered WTTPs (approach {approach_id} style)...")

    basin_col = cfg['basin_column_name']
    country_output_col = cfg['country_output_column']
    site_id_col = cfg['site_id_column']
    if basin_col not in wwtps_gdf.columns:
        raise KeyError(f"Expected basin column '{basin_col}' in WWTP dataframe before Voronoi run.")
    if basin_col not in basin_gdf.columns:
        raise KeyError(f"Expected basin column '{basin_col}' in basin dataframe.")

    scale_weights = cfg['weight_func'] in {'mult', 'add'}
    if approach_id == '1':
        # Match create_voronoi approach 1 setup.
        run_gdf = wwtps_gdf.copy()
        run_gdf['buffer_id'] = run_gdf[basin_col]
        clipping_gdf = basin_gdf.copy()
        clipping_gdf['buffer_id'] = clipping_gdf[basin_col]
        buffering = True
    elif approach_id == '0':
        # Match create_voronoi approach 0 setup.
        dissolved_buffers = orchestrate_overlaps(
            wwtps_gdf.copy(),
            cfg['max_workers'],
            paths_dict['buffers']['WWTP'],
            cfg['buffer'],
            country_col=country_output_col,
        )
        dissolved_buffers = drop_duplicates(drop_duplicates(dissolved_buffers, site_id_col), 'geometry')
        dissolved_buffers['buffer_id'] = np.arange(len(dissolved_buffers))

        run_gdf = wwtps_gdf.copy()
        run_gdf = intersect_with_polygon_sindex(
            run_gdf,
            dissolved_buffers,
            'buffer_id',
            concurrency=cfg['sindex_concurrency'],
        )
        run_gdf = drop_duplicates(drop_duplicates(run_gdf, site_id_col), 'geometry')
        clipping_gdf = dissolved_buffers
        buffering = False
    else:
        raise ValueError(f"Unsupported approach '{approach_id}'. Supported: 0, 1")

    result = run_voronoi_approach(approach_id, run_gdf, clipping_gdf, country_gdf, cfg, cfg['distance_fn'],
                                  output_path, buffer_id_col='buffer_id',
                                  scale_weights=scale_weights, only_round=only_round, buffering=buffering,
                                  method=cfg['weight_method'])

    region_df, _ = result
    if region_df is None:
        # In boolean mode run_voronoi_approach writes the regions itself and
        # returns (None, None) on success as well as on failure, so success has
        # to be established from the output file rather than the return value.
        if cfg['return_boolean'] and os.path.exists(output_path):
            region_df = gpd.read_file(output_path)
            logger.info("Read %d Voronoi polygons back from %s", len(region_df), output_path)
            return region_df
        logger.error("Voronoi orchestration failed")
        return None
    logger.info(f"Got {len(region_df)} Voronoi polygons directly from orchestration")
    return region_df


_WEB_MERCATOR = "EPSG:3857"


def _sjoin_chunk_unconnected(
    chunk: gpd.GeoDataFrame,
    voronoi_geom: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Return the rows of ``chunk`` that touch no service area at all.

    ``intersects``, not ``within``: an industrial area that straddles two
    adjacent service areas lies within neither, yet it is served, so
    ``within`` wrongly reported it as unconnected.
    """
    joined = gpd.sjoin(chunk, voronoi_geom, how='left', predicate='intersects')
    unmatched = joined[joined['index_right'].isna()].copy()
    return unmatched.drop(columns=['index_right'])


def find_unconnected_areas(
    industrial_gdf: gpd.GeoDataFrame,
    voronoi_gdf: gpd.GeoDataFrame,
    max_workers: int = 1,
) -> gpd.GeoDataFrame:
    """
    Identify industrial areas NOT overlapping any WWTP service area.
    
    Parameters
    ----------
    industrial_gdf : geopandas.GeoDataFrame
        Industrial land use areas.
    voronoi_gdf : geopandas.GeoDataFrame
        WWTP service area polygons.
    max_workers : int
        Number of threads for parallel spatial joins.
    
    Returns
    -------
    geopandas.GeoDataFrame
        Industrial areas with no WWTP service.
    """
    logger.info("Finding unconnected industrial areas...")

    original_crs = industrial_gdf.crs

    # Project both layers to Web Mercator for consistent planar geometry during join
    industrial_merc = industrial_gdf.to_crs(_WEB_MERCATOR)
    voronoi_merc = voronoi_gdf.to_crs(_WEB_MERCATOR)[['geometry']]

    if max_workers <= 1:
        unconnected_merc = _sjoin_chunk_unconnected(industrial_merc, voronoi_merc)
    else:
        chunk_size = math.ceil(len(industrial_merc) / max_workers)
        chunks = [
            industrial_merc.iloc[i : i + chunk_size]
            for i in range(0, len(industrial_merc), chunk_size)
        ]
        logger.info(f"Splitting {len(industrial_merc)} features into {len(chunks)} chunks across {max_workers} workers")
        parts: list[gpd.GeoDataFrame] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_sjoin_chunk_unconnected, chunk, voronoi_merc): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                parts[future_to_idx[future]] = future.result()
        unconnected_merc = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=False),
            crs=_WEB_MERCATOR,
        )

    # Reproject back to original CRS
    unconnected = unconnected_merc.to_crs(original_crs)

    logger.info(f"Found {len(unconnected)} unconnected industrial areas (from {len(industrial_gdf)} total)")

    return unconnected


def main():
    """Identify and save unconnected industrial areas."""

    parser = argparse.ArgumentParser(
        description='Find unconnected industrial areas using Approach 1 Voronoi with industrial-filtered WWTPs'
    )
    parser.add_argument('--approach', nargs='+', type=str, default=None,
                       help='Approach(es) to run: 0 (WWTP no watersheds), 1 (WWTP with watersheds). Default: 1')
    parser.add_argument('--only-round', action='store_true',
                       help='Use only round-area weights (same meaning as create_voronoi).')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    add_standard_override_arguments(parser)

    args = parser.parse_args()

    requested_approaches = [str(a).lower() for a in args.approach] if args.approach else ['1']
    valid_approaches = {'0', '1'}
    invalid = [a for a in requested_approaches if a not in valid_approaches]
    if invalid:
        parser.error(f"Invalid approach(es): {', '.join(invalid)}. Valid: 0, 1")
    if len(requested_approaches) > 1:
        parser.error("This script accepts one approach per run. Use --approach 0 or --approach 1")
    selected_approach = requested_approaches[0]

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="find_unconnected_industrial_areas", **overrides)
    only_round = args.only_round
    paths_dict = create_output_paths(cfg)
    
    output_path = cfg['paths']['industrial_unconnected_output']
    overwrite = cfg['overwrite_existing']
    
    # Check if output exists and overwrite is disabled
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"Output file exists and overwrite=false. Skipping.")
        return True
    
    try:
        # Load data
        industrial_gdf = load_industrial_areas(cfg)
        if industrial_gdf is None or industrial_gdf.empty:
            logger.error("No industrial areas available; cannot proceed")
            return False
        
        wwtps_gdf = load_wwtps(cfg, approach_id=selected_approach)
        if wwtps_gdf.empty:
            logger.error("No WTTPs available; cannot proceed")
            return False
        
        # Filter to industrial WTTPs
        industrial_wwtps = filter_industrial_wwtps(cfg, wwtps_gdf)
        if industrial_wwtps.empty:
            logger.warning("No industrial WTTPs after filtering")
            unconnected = industrial_gdf.copy()
        else:
            # Load preprocessed watershed + country data
            data = _resolve_configured_callable(
                cfg['prepare_data_fn'], prepare_data, 'prepare_data_fn', _pipelines_module,
            )(cfg)
            country_gdf = data['country_df']
            basin_gdf = data['basin_gdf']
            path_key = f"{selected_approach}_only_round" if only_round and selected_approach in {'0', '1'} else selected_approach
            voronoi_output_path = paths_dict['voronoi'][path_key]
            
            # Run Voronoi orchestration
            voronoi_gdf = run_voronoi_for_wwtps(
                cfg,
                selected_approach,
                industrial_wwtps,
                basin_gdf,
                country_gdf,
                paths_dict,
                voronoi_output_path,
                only_round,
            )
            if voronoi_gdf is None or voronoi_gdf.empty:
                logger.error("Failed to generate Voronoi service areas")
                return False
            
            # Find unconnected areas
            unconnected = find_unconnected_areas(
                industrial_gdf, voronoi_gdf,
                max_workers=cfg['unconnected_sjoin_workers'] or default_cpu_workers(),
            )
        
        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        logger.info(f"Writing {len(unconnected)} unconnected areas to {output_path}...")
        lower_output_path = output_path.lower()
        if lower_output_path.endswith('.parquet'):
            unconnected.to_parquet(output_path, index=False)
        elif lower_output_path.endswith('.gpkg'):
            unconnected.to_file(output_path, driver='GPKG', index=False)
        else:
            raise ValueError(f"Unsupported industrial_unconnected_output extension: {output_path}")
        logger.info(f"Successfully created {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    configure_logging()
    success = main()
    sys.exit(0 if success else 1)
