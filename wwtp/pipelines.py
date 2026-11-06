"""
Pipeline orchestration for Voronoi-based spatial analysis.

Functions are grouped by purpose:

**Internal helpers**
  - ``_resolve_configured_callable``    — resolve a config string or callable
    to an actual function in a given module

**Path builders**
  - ``create_output_paths``    — canonical output paths for all Voronoi approaches
  - ``create_pop_output_paths`` — population-enriched output path variants

**Data preparation**
  - ``prepare_data`` — load and enrich all spatial inputs (WWTP, basin, country
    layers); can be swapped via ``cfg['prepare_data_fn']``

**Voronoi execution**
  - ``run_voronoi_approach`` — run one Voronoi approach end-to-end using
    configurable area/buffer/data functions resolved through
    ``_resolve_configured_callable``

All config values are read directly from the ``cfg`` dict produced by
``starter.load_config``. The one exception is ``run_voronoi_approach``, whose
keyword arguments (``buffer_id_col``, ``scale_weights``, ``only_round``,
``buffering``, ``method``) carry Python-level defaults; every caller in the
pipeline passes them explicitly from ``cfg``, so those defaults only apply to
direct in-process calls.
"""

import hashlib
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import from_wkt, to_wkt, from_wkb
import shapely
import logging

logger = logging.getLogger(__name__)


SITE_UID_COLUMN = 'SITE_UID'


def _site_uid(source_id, wkt):
    """Positive 63-bit integer hash of a site's source ID and geometry."""
    key = f"{'' if pd.isna(source_id) else source_id}|{wkt}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big') & 0x7FFF_FFFF_FFFF_FFFF


def assign_stable_site_ids(gdf, site_id_col, old_site_id_col, wkt_col='WKT_WWTP'):
    """Give every site an ID that does not depend on row order or later filters.

    The source ID moves to ``old_site_id_col``. ``SITE_UID`` is a hash of the
    source ID and the geometry (WKT), so it is the same for the same plant in
    every run and every dataset version. ``site_id_col`` stays a dense integer
    for the joins that need one, numbered in ``SITE_UID`` order - not in file
    order - and is assigned once, before ``remove_industrial`` filters rows, so
    a plant keeps its ID whether industrial plants are removed or not.
    """
    gdf = gdf.copy()
    gdf[old_site_id_col] = gdf[site_id_col]
    gdf[SITE_UID_COLUMN] = [
        _site_uid(source_id, wkt) for source_id, wkt in zip(gdf[old_site_id_col], gdf[wkt_col])
    ]
    if gdf[SITE_UID_COLUMN].duplicated().any():
        raise ValueError("Duplicate SITE_UID: two sites share the same source ID and geometry")
    order = np.argsort(gdf[SITE_UID_COLUMN].to_numpy(), kind='stable')
    ids = np.empty(len(gdf), dtype=np.int64)
    ids[order] = np.arange(len(gdf))
    gdf[site_id_col] = ids
    return gdf


def _resolve_configured_callable(value, default_fn, cfg_key, module):
    """Resolve a config value to a callable, looking it up by name in *module* if needed.

    Parameters
    ----------
    value : callable | str | None
        Raw config value — already a callable, a function-name string, or
        ``None`` (meaning "use the default").
    default_fn : callable
        Returned when *value* is ``None`` or an empty string.
    cfg_key : str
        Config key being resolved, used in error messages only.
    module : module
        Python module searched by ``getattr`` when *value* is a string.
    """
    if value is None:
        return default_fn
    if callable(value):
        return value
    if isinstance(value, str):
        fn_name = value.strip()
        if not fn_name:
            return default_fn
        resolved = getattr(module, fn_name, None)
        if not callable(resolved):
            raise ValueError(
                f"cfg['{cfg_key}'] references '{fn_name}', but no callable with "
                f"that name exists in {module.__name__!r}."
            )
        return resolved
    raise TypeError(
        f"cfg['{cfg_key}'] must be a callable or string function name, got {type(value).__name__!r}."
    )


def create_output_paths(cfg):
    """Generate canonical output paths for all Voronoi approaches.

    Returns paths for the buffer products plus the approach family used by
    create_voronoi.py. Weighted/unweighted and multiplicative/additive variants
    are encoded in the output directory name via ``weight_type`` and
    ``weight_func`` (resolved from config); the only per-run variant captured
    here is the ``only_round`` flag:

    - `0` / `0_only_round`: WWTP buffer Voronoi (all points / only-round)
    - `1` / `1_only_round`: watershed-constrained Voronoi (all points / only-round)
    - `2`: city-based Voronoi
    """
    version = cfg['version']
    level = cfg['level']
    buffer = cfg['buffer_path_token']
    weight_func = cfg['weight_func']
    weight_func_suffix = cfg['weight_func_suffix']
    weight_type = cfg['weight_type']
    buffers_dir = cfg['paths']['buffers_dir']
    voronoi_dir = cfg['paths']['voronoi_dir']
    
    paths = {
        'buffers': {
            'WWTP': os.path.join(buffers_dir, f'dissolved_wwtp_buffers_v{version}_lvl{level}_bf{buffer}.gpkg'),
            'city': os.path.join(buffers_dir, f'dissolved_city_buffers_v{version}_lvl{level}_bf{buffer}.gpkg'),
            'WWTP_convex': os.path.join(buffers_dir, f'dissolved_wwtp_convex_hull_v{version}_lvl{level}_bf{buffer}.gpkg'),
            'city_convex': os.path.join(buffers_dir, f'dissolved_city_convex_hull_v{version}_lvl{level}_bf{buffer}.gpkg'),
        },
        'voronoi': {
            '0': os.path.join(voronoi_dir, f'appr_0_v{version}_lvl{level}_bf{buffer}_{weight_type}{weight_func_suffix}.gpkg'),
            '0_only_round': os.path.join(voronoi_dir, f'appr_0_only_round_v{version}_lvl{level}_bf{buffer}_{weight_type}{weight_func_suffix}.gpkg'),
            '1': os.path.join(voronoi_dir, f'appr_1_v{version}_lvl{level}_bf{buffer}_{weight_type}{weight_func_suffix}.gpkg'),
            '1_only_round': os.path.join(voronoi_dir, f'appr_1_only_round_v{version}_lvl{level}_bf{buffer}_{weight_type}{weight_func_suffix}.gpkg'),
            '2': os.path.join(voronoi_dir, f'appr_2_v{version}_lvl{level}_bf{buffer}_{weight_type}{weight_func_suffix}.gpkg'),
        }
    }
    return paths

def create_pop_output_paths(cfg):
    """Return output paths for population-enriched copies of Voronoi outputs."""
    voronois = create_output_paths(cfg)['voronoi']
    return {
        'voronoi' : {
            k: os.path.abspath(os.path.join(cfg['paths']['pop_output_dir'], f'pop_added_{os.path.basename(v)}')) for k, v in voronois.items()
            }
        }
    

def run_voronoi_approach(approach_id, gdf, clipping_gdf, country_df, cfg, distance_fn, output_path,
                        buffer_id_col='buffer_id', scale_weights=False, only_round=False, buffering=False,
                        method='linear', site_country_col=None, country_boundary_col=None):
    """Run a single Voronoi generation approach end-to-end.

    Parameters
    ----------
    approach_id : str
        Approach identifier such as ``'0'``, ``'1'``, or ``'2'``.
    gdf : geopandas.GeoDataFrame
        Input sites for the selected approach.
    clipping_gdf : geopandas.GeoDataFrame | None
        Optional clipping geometries used to constrain Voronoi regions (e.g.
        basin polygons for approach 1, dissolved WWTP buffers for approach 0).
    country_df : geopandas.GeoDataFrame
        Country boundaries used for final spatial clipping.
    cfg : dict
        Runtime configuration dictionary produced by ``starter.load_config``.
        All function names and kwargs are resolved from this dict; no defaults
        are applied here.
    distance_fn : callable
        Distance function passed to the weighted Voronoi solver
        (``cfg['distance_fn']`` from the caller).
    output_path : str
        Absolute path where the resulting GeoPackage is written.
    buffer_id_col : str, default='buffer_id'
        Column used to group features into independent Voronoi partitions.
    scale_weights : bool, default=False
        Whether to scale feature weights before region generation.
    only_round : bool, default=False
        When ``True``, only the round-area component is used for weighting.
        Forwarded as ``area_fn_kwargs['only_round']``.
    buffering : bool, default=False
        When ``True``, intersect the output with the local feature buffers
        (approach-0 style).
    method : str, default='linear'
        Weight-transformation method passed to the Voronoi orchestrator
        (one of ``'linear'``, ``'logarithmic'``, ``'square_root'``,
        ``'sigmoid'``).
    site_country_col : str | None, default=None
        Country-code column on ``gdf``. Falls back to
        ``cfg['country_output_column']`` when ``None``.
    country_boundary_col : str | None, default=None
        Country-code column on ``country_df``. Falls back to
        ``cfg['country_boundary_column']`` when ``None``.

    Returns
    -------
    tuple
        ``(region_df, point_df)`` as returned by the Voronoi orchestrator, or
        ``(None, None)`` if the run was skipped or orchestration failed.

    Notes
    -----
    Skip-if-exists logic is applied at the top of this function using
    ``cfg['overwrite_existing']``.  The area function, buffer function, and
    their respective kwargs are resolved via ``_resolve_configured_callable``
    using ``cfg['calculate_area_fn']``, ``cfg['calculate_buffer_fn']``,
    ``cfg['area_fn_kwargs']``, and ``cfg['calculate_buffer_kwargs']``.
    """
    if os.path.exists(output_path) and not cfg['overwrite_existing']:
        logger.info(f"Approach {approach_id}: Output already exists at {output_path} and overwrite is False. Skipping.")
        return None, None

    if int(cfg['max_workers']) < 1:
        raise ValueError("cfg['max_workers'] must be >= 1")
    
    from . import service_areas as service_areas_module
    from .service_areas import orchestrate_voronoi_weights, drop_duplicates, calculate_area, calculate_buffer
    from .utils import ensure_output_dir_for_file, select_industrial_categories

    site_country_col = site_country_col or cfg['country_output_column']
    country_boundary_col = country_boundary_col or cfg['country_boundary_column']
    site_id_col = cfg['site_id_column']
    
    calc_buffer_kwargs = cfg['calculate_buffer_kwargs']
    if calc_buffer_kwargs is not None and not isinstance(calc_buffer_kwargs, dict):
        raise TypeError("cfg['calculate_buffer_kwargs'] must be a dict when provided.")

    calculate_area_fn = _resolve_configured_callable(
        cfg['calculate_area_fn'],
        calculate_area,
        'calculate_area_fn',
        service_areas_module,
    )

    calculate_buffer_fn = _resolve_configured_callable(
        cfg['calculate_buffer_fn'],
        calculate_buffer,
        'calculate_buffer_fn',
        service_areas_module,
    )
    
    cfg_area_kwargs = cfg['area_fn_kwargs']
    if cfg_area_kwargs is None:
        cfg_area_kwargs = {}
    if not isinstance(cfg_area_kwargs, dict):
        raise TypeError("cfg['area_fn_kwargs'] must be a dict when provided.")
    area_kwargs = dict(cfg_area_kwargs)
    area_kwargs['only_round'] = only_round

    logger.info(f"Approach {approach_id}: Running Voronoi generation (scale_weights={scale_weights}, only_round={only_round})")
    
    orchestrate_result = orchestrate_voronoi_weights(
        gdf, buffer_id_col, country_df, cfg['max_workers'],
        scale_weights=scale_weights,
        clipping=clipping_gdf,
        n_points=cfg['n_points'],
        distance_fn=distance_fn,
        scipy_true=cfg['scipy_true'],
        cv2_true=cfg['cv2_true'],
        centroid_points=True,
        buffering=buffering,
        threshold=cfg['threshold'],
        sigma=cfg['sigma'],
        percent_threshold=cfg['percent_threshold'],
        area_fn=calculate_area_fn,
        area_fn_kwargs=area_kwargs,
        method=method,
        output_path=output_path if cfg['return_boolean'] else None,
        overwrite=cfg['temp_voronoi_overwrite'],
        flush_size=cfg['flush_size'],
        calculate_buffer_fn=calculate_buffer_fn,
        buffer_fn_kwargs=calc_buffer_kwargs,
        site_country_col=site_country_col,
        country_boundary_col=country_boundary_col,
        site_id_col=site_id_col,
    )

    if isinstance(orchestrate_result, bool):
        if not orchestrate_result:
            logger.error(f"Approach {approach_id}: Voronoi orchestration failed")
            return None, None
        logger.info(f"Approach {approach_id}: Saved regions to {output_path}")
        return None, None

    region_df, point_df = orchestrate_result
    ensure_output_dir_for_file(output_path)
    region_df.to_file(output_path, driver='GPKG', index=False)
    logger.info(f"Approach {approach_id}: Saved {len(region_df)} regions to {output_path}")
    return region_df, point_df


def prepare_data(cfg):
    """Load and enrich all spatial inputs required by the Voronoi pipeline.

    This is the default data-preparation function.  It can be replaced by
    setting ``cfg['prepare_data_fn']`` to the name of another function in
    this module (or a callable) — see ``_resolve_configured_callable``.

    Parameters
    ----------
    cfg : dict
        Runtime configuration dictionary produced by ``starter.load_config``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'gdf_bbox'``   : WWTP site GeoDataFrame enriched with country codes
          and basin identifiers, with buffered geometry and re-indexed site IDs.
        - ``'basin_gdf'`` : Basin polygon GeoDataFrame enriched with country
          codes.
        - ``'country_df'``    : Country boundary GeoDataFrame loaded from the
          Overture parquet cache.
    """
    from .service_areas import (
        drop_duplicates, intersects_with_country_db,
        download_overture_maps, intersect_with_polygon_sindex,
        orchestrate_overlaps,
    )
    from .geo_utils import buffer_geometry
    from .utils import ensure_output_dir_for_file, select_industrial_categories
    
    logger.info("Preparing input data...")
    paths = cfg['paths']
    country_output_col = cfg['country_output_column']
    country_boundary_col = cfg['country_boundary_column']
    site_id_col = cfg['site_id_column']
    
    # Load WWTP bounding boxes
    if cfg['csv_files']:
        gdf_bbox = pd.read_csv(paths['bboxes'])
        hydrowaste_df = pd.read_csv(paths['hydrowaste'])
        gdf_bbox = pd.merge(gdf_bbox, hydrowaste_df.drop(['LON_WWTP', 'LAT_WWTP', 'geometry', 'POP_SERVED'], axis=1), on=[site_id_col])
        gdf_bbox = gpd.GeoDataFrame(
            gdf_bbox,
            geometry=gdf_bbox['geometry'].map(from_wkt),
            crs='epsg:4326',
        )
    else:
        gdf_bbox = gpd.read_file(paths['annotated_all_filepath'])
        if 'final_geometry' in gdf_bbox.columns:
            gdf_bbox['geometry_wkt'] = gdf_bbox['geometry'].apply(to_wkt)
            gdf_bbox['geometry'] = gdf_bbox['final_geometry']
            gdf_bbox = gdf_bbox.drop(columns=['final_geometry'])
    
    gdf_bbox = drop_duplicates(drop_duplicates(gdf_bbox, site_id_col), 'geometry')
    gdf_bbox['geometry'] = pd.Series(
        [buffer_geometry(geom) for geom in gdf_bbox['geometry']],
        index=gdf_bbox.index,
    )
    gdf_bbox['WKT_WWTP'] = gdf_bbox['geometry'].apply(lambda geom: to_wkt(geom))
    gdf_bbox = assign_stable_site_ids(gdf_bbox, site_id_col, cfg['old_site_id_column'])

    if cfg['remove_industrial']:
        # Discard direction: industrial and mixed-use sites both leave the layer.
        # The industrial pipeline keeps exactly this set - see
        # utils.select_industrial_categories.
        gdf_bbox = select_industrial_categories(
            gdf_bbox,
            cfg['industrial_category_numbers'],
            cfg['mix_use_categories'],
            keep=False,
            logger=logger,
        )

    # Add country codes. Rejoining is the default: an inherited country column
    # from an upstream layer can disagree with the boundary layer in use.
    if cfg['force_country_rejoin'] or country_output_col not in gdf_bbox.columns:
        if country_output_col in gdf_bbox.columns:
            gdf_bbox = gdf_bbox.drop(columns=[country_output_col])
        if not os.path.exists(paths['overture']):
            download_overture_maps(paths['overture_s3_url'], paths['overture'])
        gdf_bbox = intersects_with_country_db(
            gdf_bbox,
            paths['overture'],
            polygon_country_col=country_boundary_col,
            output_country_col=country_output_col,
        )
    gdf_bbox.loc[gdf_bbox[country_output_col].isna(), country_output_col] = 'XX'
    
    # Load basins
    basin_col = cfg['basin_column_name']
    basin_gdf = gpd.read_file(paths['watershed'], crs='epsg:4326')
    basin_gdf = basin_gdf.drop_duplicates(subset=[basin_col, 'geometry']).reset_index(drop=True)
    basin_gdf['geometry'] = pd.Series(
        [buffer_geometry(geom) for geom in basin_gdf['geometry']],
        index=basin_gdf.index,
    )
    basin_gdf['basin_area'] = basin_gdf[['geometry']].to_crs(6933).geometry.area
    if cfg['force_country_rejoin'] or country_output_col not in basin_gdf.columns:
        if country_output_col in basin_gdf.columns:
            basin_gdf = basin_gdf.drop(columns=[country_output_col])
        if not os.path.exists(paths['overture']):
            download_overture_maps(paths['overture_s3_url'], paths['overture'])
        basin_gdf = intersects_with_country_db(
            basin_gdf,
            paths['overture'],
            polygon_country_col=country_boundary_col,
            output_country_col=country_output_col,
        )
    watershed_gpkg_filepath = os.path.abspath(paths['watershed'].replace('.geojson', '.gpkg'))
    if not os.path.exists(watershed_gpkg_filepath):
        ensure_output_dir_for_file(watershed_gpkg_filepath)
        basin_gdf.to_file(watershed_gpkg_filepath, driver='GPKG', index=False)

    # Add basin information to WWTP
    if basin_col not in gdf_bbox.columns:
        gdf_bbox = intersect_with_polygon_sindex(gdf_bbox, basin_gdf, basin_col, concurrency=cfg['sindex_concurrency'])
        gdf_bbox = pd.merge(gdf_bbox, basin_gdf[[basin_col, 'basin_area']], on=basin_col, how='left')
        gdf_bbox = drop_duplicates(drop_duplicates(gdf_bbox, site_id_col), 'geometry')
        filename = os.path.join(os.path.dirname(paths['bboxes']), f"expanded_{os.path.basename(paths['bboxes'])}")
        if not os.path.exists(f"{filename}"):    
            ensure_output_dir_for_file(filename)
            gdf_bbox.to_csv(f"{filename}", index=False)
        expanded_gpkg = filename.replace('.csv', '.gpkg')
        if not os.path.exists(expanded_gpkg):
            ensure_output_dir_for_file(expanded_gpkg)
            gdf_bbox_export = gpd.GeoDataFrame(gdf_bbox, geometry='geometry', crs='epsg:4326')
            gdf_bbox_export.to_file(expanded_gpkg, index=False, driver='GPKG')
        
    # Load country boundaries
    country_df = pd.read_parquet(paths['overture'])
    country_df['geometry'] = country_df['geometry'].map(lambda geom: from_wkb(geom) if pd.notna(geom) else None)
    country_df = gpd.GeoDataFrame(country_df, geometry='geometry', crs=4326)
    
    logger.info(f"Loaded {len(gdf_bbox)} WWTP sites, {len(basin_gdf)} basins, {len(country_df)} countries")
    return {'gdf_bbox': gdf_bbox, 'basin_gdf': basin_gdf, 'country_df': country_df}


