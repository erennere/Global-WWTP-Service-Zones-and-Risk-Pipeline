"""Service areas (stage create_voronoi): command-line entry point.

Builds the service area of every plant with approach 0 (dissolved buffers), 1 (basins) or 2
(cities). The code lives in ``src/service_areas/`` (size, weights, solver, contours, joins,
overlaps, orchestrate); this module reads the config and runs the requested approaches.

Run as ``python -m src.create_voronoi --approach 1``.
"""
import argparse
import os
import logging
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import from_wkt
from .geo_utils import buffer_geometry
from .utils import configure_logging
from .service_areas import (
    download_overture_maps,
    drop_duplicates,
    intersect_with_polygon_sindex,
    intersect_with_polygons_parallelized,
    intersects_with_country_db,
    orchestrate_overlaps,
)

_LOG_LEVEL_NAME = os.environ.get(
    "WWTP_SERVICE_PIPELINE_LOG_LEVEL",
    "INFO",
).upper()


_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)


logger = logging.getLogger(__name__)


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
        from . import provenance
    except ImportError:
        from starter import add_standard_override_arguments
        import provenance
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
    provenance.write_run_manifest(cfg, "create_voronoi")
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
