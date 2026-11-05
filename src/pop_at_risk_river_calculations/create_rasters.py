"""Create signed population rasters and extract unserved-population islands by basin.

This script combines:
1. Raster sign assignment (+ inside WWTP Voronoi polygons, - outside), and
2. Island extraction/statistics for non-served populated areas by watershed basin.

It is intended for country-wise batch processing of WorldPop rasters.
"""

import os
import random
import logging
import gc
import argparse
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio import windows
from rasterio.features import shapes, geometry_mask, rasterize
from exactextract import exact_extract
from shapely.geometry import shape, box
from shapely import to_wkt
from shapely.ops import unary_union

try:
    from ..add_pop import find_newest_country_tif_files
    from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
    from .. import provenance
    from ..service_areas import download_overture_maps, intersects_with_country_db
    from ..pipelines import create_pop_output_paths
    from ..utils import configure_logging, ensure_output_dir_for_file
    from ..geo_utils import repair_geometry
    from .find_pop_in_danger_pop import find_bbox, finding_tiles
except ImportError:
    from src.add_pop import find_newest_country_tif_files
    from src.starter import add_standard_override_arguments, load_config, parse_config_overrides
    from src import provenance
    from src.service_areas import download_overture_maps, intersects_with_country_db
    from src.pipelines import create_pop_output_paths
    from src.utils import configure_logging, ensure_output_dir_for_file
    from src.geo_utils import repair_geometry
    from src.pop_at_risk_river_calculations.find_pop_in_danger_pop import find_bbox, finding_tiles

logger = logging.getLogger(__name__)


def _sanitize_polygon_geom(geom):
    """Return a valid Polygon/MultiPolygon geometry or None."""
    geom = repair_geometry(geom)
    if geom is None:
        return None

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom

    if geom.geom_type == "GeometryCollection":
        polys = [
            g for g in geom.geoms
            if (not g.is_empty) and g.geom_type in ("Polygon", "MultiPolygon")
        ]
        if not polys:
            return None
        merged = unary_union(polys)
        if merged.is_empty:
            return None
        if merged.geom_type in ("Polygon", "MultiPolygon"):
            return merged

    return None

def geotiff_exists_and_valid(path):
    """Return True when a GeoTIFF exists and raster metadata can be read."""
    if not os.path.exists(path):
        return False
    try:
        with rasterio.open(path) as src:
            _ = src.count        # forces metadata read
        return True
    except Exception:
        return False
    
def extract_worldpop_universal(raster_path, hybas_gdf, exclude_gdf, min_pixels=9, zoom_level=8, basin_col='HYBAS_ID',
                               all_touched=False):
    """
    Extracts population islands from WorldPop rasters with strict RAM management.
    Designed for 64GB / 4 processes (16GB per worker).

    Parameters
    ----------
    raster_path : str
        Input population raster path.
    hybas_gdf : geopandas.GeoDataFrame
        Watershed polygons that provide basin IDs and metadata.
    exclude_gdf : geopandas.GeoDataFrame
        Polygons to exclude (served areas).
    min_pixels : int, default=9
        Minimum island size in pixels, unless touching window boundaries.

    Returns
    -------
    geopandas.GeoDataFrame | None
        Extracted islands with basin metadata and zonal statistics.

    Notes
    -----
    This chunked implementation is the maintained extraction path. Earlier
    experimental versions were removed to avoid ambiguity about which logic is
    currently in use.
    """
    geom_registry = {}
    country_code = os.path.basename(raster_path)
    MERGE_THRESHOLD = 200  # Merge shards into the registry every 200 items

    try:
        with rasterio.open(raster_path) as src:
            crs = src.crs
            res = src.res[0]
            transform = src.transform

            logger.info("[%s] Aligning CRS and building spatial indices", country_code)
            hybas_gdf = hybas_gdf.to_crs(crs)
            exclude_gdf = exclude_gdf.to_crs(crs)

            # Pre-trigger spatial index creation
            _ = hybas_gdf.sindex
            _ = exclude_gdf.sindex

            # Iterate windows
            for i, (index, window) in enumerate(src.block_windows(1)):
                if i % 250 == 0:
                    logger.info("[%s] Processing window %s", country_code, i)

                w_bounds = windows.bounds(window, transform)
                w_transform = windows.transform(window, transform)
                w_poly_boundary = box(*w_bounds).boundary

                # Spatial intersection check
                possible_idx = list(hybas_gdf.sindex.intersection(w_bounds))
                if not possible_idx:
                    continue

                # 1. READ AND BINARIZE IN-PLACE (Save RAM)
                binary_data = src.read(1, window=window)
                np.nan_to_num(binary_data, copy=False)
                binary_data[binary_data <= 0] = 0
                binary_data[binary_data > 0] = 1
                binary_data = binary_data.astype(np.uint8, copy=False)

                # 2. EXCLUSION MASKING
                excl_idx = list(exclude_gdf.sindex.intersection(w_bounds))
                if excl_idx:
                    rel_excl = exclude_gdf.iloc[excl_idx]
                    excl_mask = geometry_mask(
                        rel_excl.geometry,
                        (window.height, window.width),
                        w_transform,
                        invert=True,
                        all_touched=all_touched,
                    )
                    binary_data[excl_mask] = 0
                    del excl_mask, rel_excl

                # 3. BASIN PROCESSING
                for idx in possible_idx:
                    row = hybas_gdf.iloc[idx]
                    h_id = row.get(basin_col)

                    # Create basin mask
                    h_mask = geometry_mask(
                        [row.geometry],
                        (window.height, window.width),
                        w_transform,
                        invert=False
                    )

                    # Combine masks: population pixels inside basin
                    h_mask = (binary_data == 1) & (~h_mask)
                    if not h_mask.any():
                        continue

                    # Vectorize shards
                    shape_gen = shapes(
                        h_mask.astype(np.uint8, copy=False),
                        mask=h_mask,
                        transform=w_transform,
                        connectivity=8
                    )

                    window_shards = []
                    for geom, _ in shape_gen:
                        poly = shape(geom).buffer(0)
                        shard_pixels = round(poly.area / (res * res))
                        
                        # Filter noise unless it touches window boundary (potential split island)
                        if shard_pixels < min_pixels and not poly.intersects(w_poly_boundary):
                            continue
                        window_shards.append(poly)

                    if not window_shards:
                        continue

                    # Update Registry with Deferred Unioning
                    if h_id not in geom_registry:
                        geom_registry[h_id] = {
                            "geom_list": window_shards,
                            "meta": {k: row.get(k) for k in ["NEXT_DOWN", "NEXT_SINK", "MAIN_BAS"]}
                        }
                    else:
                        geom_registry[h_id]["geom_list"].extend(window_shards)
                        
                        # Merge if list is too long to prevent row-count bloat
                        if len(geom_registry[h_id]["geom_list"]) >= MERGE_THRESHOLD:
                            merged = unary_union(geom_registry[h_id]["geom_list"])
                            geom_registry[h_id]["geom_list"] = [merged]

                del binary_data
                gc.collect()

        # 4. FINAL MERGE AND EXPLODE
        logger.info("[%s] Loop complete. Exploding MultiPolygons", country_code)
        final_rows = []

        for h_id, content in geom_registry.items():
            # Final merge of all shards for this Basin ID
            merged = unary_union(content["geom_list"])
            geoms = getattr(merged, "geoms", None)
            islands = list(geoms) if geoms is not None else [merged]

            for island in islands:
                if not island.is_empty:
                    tiles = finding_tiles(island, zoom_level=zoom_level)  # Example zoom level for tile assignment
                    for tile in tiles:
                        clipped = _sanitize_polygon_geom(island.intersection(find_bbox(tile)))
                        if clipped is None:
                            continue
                        final_rows.append({
                        "geometry": clipped,
                        basin_col: h_id,
                        "tile": tile,
                        **content["meta"]
                    })
                                        
        del geom_registry
        gc.collect()

        if not final_rows:
            return None

        final_gdf = gpd.GeoDataFrame(final_rows, crs=crs)

        del final_rows
        logger.info("[%s] Total islands to check: %s", country_code, len(final_gdf))

        # 5. CHUNKED ZONAL STATS (The OOM-Killer Prevention)
        sums = np.zeros(len(final_gdf), dtype=np.int64)
        counts = np.zeros(len(final_gdf), dtype=np.int64)
        chunk_size = 100000

        for start_idx in range(0, len(final_gdf), chunk_size):
            end_idx = min(start_idx + chunk_size, len(final_gdf))
            logger.info("[%s] Calculating stats for chunk %s-%s", country_code, start_idx, end_idx)

            # exact_extract is faster and more memory-efficient than rasterstats
            chunk = final_gdf.iloc[start_idx:end_idx][["geometry"]].copy()
            chunk["geometry"] = pd.Series(
                [_sanitize_polygon_geom(geom) for geom in chunk["geometry"]],
                index=chunk.index,
            )
            invalid = chunk["geometry"].isna()
            if invalid.any():
                logger.warning(
                    "[%s] Dropping %s invalid geometries before exact_extract in chunk %s-%s",
                    country_code,
                    int(invalid.sum()),
                    start_idx,
                    end_idx,
                )
                chunk = chunk[~invalid]

            if chunk.empty:
                continue

            stats_df = exact_extract(
                rast=raster_path,
                vec=chunk,
                ops=["sum", "count"],
                output="pandas"
            )

            if stats_df is None or len(stats_df) == 0:
                continue
            stats_df = pd.DataFrame(stats_df)

            row_index = chunk.index.to_numpy()
            sum_values = np.asarray(
                np.round(stats_df["sum"].fillna(0)).astype(np.int64)
            )
            count_values = np.asarray(
                stats_df["count"].fillna(0).astype(np.int64)
            )
            sums[row_index] = sum_values
            counts[row_index] = count_values

            del stats_df, chunk
            gc.collect()

        final_gdf["pop_sum"] = sums
        final_gdf["pixel_count"] = counts
        del sums, counts

        # 6. FINAL CLEANUP
        logger.info("[%s] Filtering and resetting index", country_code)
        final_gdf = final_gdf[
            (final_gdf["pop_sum"] > 0) & 
            (final_gdf["pixel_count"] >= min_pixels)
        ].copy().reset_index(drop=True)

        final_gdf["pop_sum"] = final_gdf["pop_sum"].astype(np.int64)
        final_gdf["pixel_count"] = final_gdf["pixel_count"].astype(np.int64)

        logger.info("[%s] SUCCESS. Final islands: %s", country_code, len(final_gdf))
        return final_gdf

    except Exception as e:
        logger.exception("[%s] CRITICAL FAILURE: %s", country_code, str(e))
        return None
    
def polygon_raster_sign_from_gdf(raster_path, polygons_gdf, output_path, all_touched=False):
    """Write a signed raster that is positive inside served polygons and negative outside.

    Parameters
    ----------
    raster_path : str
        Input population raster path.
    polygons_gdf : geopandas.GeoDataFrame
        Served-area polygons used to determine the sign mask.
    output_path : str
        Output raster path.

    Returns
    -------
    tuple[str, int | None, int | None]
        Output path, positive sum, and negative sum.
    """
    try:
        # 1. Build a spatial index for the GDF (CRITICAL for speed)
        sindex = polygons_gdf.sindex

        with rasterio.open(raster_path) as src:
            logger.info("Raster block shapes for %s: %s", raster_path, src.block_shapes)

            profile = src.profile.copy()
            transform = src.transform
            nodata_val = src.nodata

            profile.update(
                dtype="float32",
                count=1,
                nodata=None,
                compress="lzw",
                tiled=True,
                blockxsize=256, # Standardizing block sizes for better I/O
                blockysize=256
            )

            # WorldPop counts are fractional (most rural pixels hold < 1
            # person); keep them as floats and sum in float64.
            sum_positive = 0.0
            sum_negative = 0.0

            ensure_output_dir_for_file(output_path)
            with rasterio.open(output_path, "w", **profile) as dst:
                for _, window in src.block_windows(1):
                    # Get geographic bounds of the current window
                    win_bounds = windows.bounds(window, transform)
                    
                    # 2. Filter polygons: Only those inside/touching this window
                    possible_idx = list(sindex.intersection(win_bounds))
                    
                    # Read the data block
                    data = src.read(1, window=window)
                    
                    # Convert nodata/NaN to 0 (no integer cast: that truncated
                    # every pixel below 1 person to 0)
                    data = data.astype(np.float64)
                    if nodata_val is not None:
                        data = np.where(np.isclose(data, nodata_val), 0.0, data)
                    data = np.nan_to_num(data, nan=0.0)
                    
                    # Force strictly positive to avoid sign errors
                    data = np.abs(data)

                    # 3. Conditional Rasterization
                    if possible_idx:
                        # Extract the actual geometries for this window
                        window_shapes = [(geom, 1) for geom in polygons_gdf.iloc[possible_idx].geometry]
                        
                        mask = rasterize(
                            window_shapes,
                            out_shape=(window.height, window.width),
                            transform=windows.transform(window, transform),
                            fill=0,
                            dtype=np.uint8,
                            all_touched=all_touched
                        )
                        
                        # Within polygons = positive, Outside = negative
                        signed = np.where(mask == 1, data, -data)
                    else:
                        # No polygons in this window? Everything is negative
                        signed = -data

                    # 4. Global statistics
                    sum_positive += float(signed[signed > 0].sum())
                    sum_negative += float(signed[signed < 0].sum())

                    dst.write(signed.astype(np.float32), 1, window=window)
        return output_path, sum_positive, sum_negative

    except Exception as err:
        logger.exception("Error processing raster %s: %s", raster_path, err)
        return output_path, None, None

def orchestrate_country_intersection(raster_path, polygons_gdf, watershed_gdf, output_path, min_pixels=9, zoom_level=8, basin_col='HYBAS_ID',
                                     all_touched=False):
    """Process one country raster and return signed-raster stats plus extracted islands.

    Parameters
    ----------
    raster_path : str
        Input country raster path.
    polygons_gdf : geopandas.GeoDataFrame
        Served-area polygons for the country.
    watershed_gdf : geopandas.GeoDataFrame
        Watershed polygons used to aggregate non-served islands.
    output_path : str
        Output path for the signed raster.
    min_pixels : int, default=9
        Minimum island size retained after extraction.
    zoom_level : int, default=8
        Tile zoom level used for non-served island tagging.

    Returns
    -------
    tuple
        Tuple ``(filepath, sum_pos, sum_neg, gdf)`` containing signed-raster
        statistics and the extracted island GeoDataFrame.
    """
    # One 'served' rule for both outputs: a pixel is served when its centre
    # (or, with all_touched, any part) lies in a service area.
    filepath, sum_pos, sum_neg = polygon_raster_sign_from_gdf(raster_path, polygons_gdf, output_path, all_touched=all_touched)
    #gdf = extract_worldpop_optimized_v2(raster_path, polygons_gdf)
    gdf = extract_worldpop_universal(raster_path, watershed_gdf, polygons_gdf, min_pixels=min_pixels, zoom_level=zoom_level, basin_col=basin_col,
                                     all_touched=all_touched)
    return filepath, sum_pos, sum_neg, gdf

def _append_csv(df, filepath):
    """Append ``df`` to ``filepath``, writing the header only on first creation."""
    ensure_output_dir_for_file(filepath)
    exists = os.path.exists(filepath)
    df.to_csv(filepath, index=False, mode='a' if exists else 'w', header=not exists)


def orchestrate_intersections(tif_dict, gdf, watershed_gdf, output_dir, csv_output_filepath, non_served_outpath, max_workers=4,
                              min_pixels=9, zoom_level=8, country_col='ISO_2', basin_col='HYBAS_ID', all_touched=False):
    """
    Process multiple raster files in parallel, intersecting with polygons
    and creating signed rasters (+ inside polygons, - outside).

    Parameters
    ----------
    tif_dict : dict
        Dictionary mapping country codes to raster filepaths.
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygons with a country-code column (see ``country_col``).
    watershed_gdf : geopandas.GeoDataFrame
        Watershed polygons used for island extraction.
    output_dir : str
        Directory where output rasters will be saved.
    csv_output_filepath : str
        Output CSV path for per-country population statistics.
    non_served_outpath : str
        Output base path for the extracted non-served polygons.
    max_workers : int
        Maximum number of parallel processes.
    min_pixels : int, default=9
        Minimum island size retained after extraction.
    zoom_level : int, default=8
        Tile zoom level used for non-served island tagging.

    Returns
    -------
    dict[str, bool]
        Mapping from country code to success flag.
    """
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")

    logger.info("Starting orchestration for %s countries", len(tif_dict))
    results = {}

    # Create a mapping: Future -> country
    future_to_country = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        countries = list(tif_dict.keys())  # convert dict keys to a list
        random.shuffle(countries)          # shuffle in-place

        if os.path.exists(csv_output_filepath):
            stats = pd.read_csv(csv_output_filepath).country.unique()
            countries = [c for c in countries if c not in stats]
            logger.info("Skipping %s previously processed countries", len(tif_dict) - len(countries))
        
        for country in countries:
            tif_filepath = tif_dict[country]
            country_dir = os.path.join(output_dir, country)
            os.makedirs(country_dir, exist_ok=True)

            output_filepath = os.path.join(
                country_dir, f'WWTP_{os.path.basename(tif_filepath)}'
            )

            future = executor.submit(
                orchestrate_country_intersection,
                tif_filepath,
                gdf[gdf[country_col] == country],
                watershed_gdf[watershed_gdf[country_col] == country],
                output_filepath,
                min_pixels=min_pixels,
                zoom_level=zoom_level,
                basin_col=basin_col,
                all_touched=all_touched,
            )
            future_to_country[future] = country
        # Collect results with a progress bar

        for future in tqdm(as_completed(future_to_country),
                           total=len(future_to_country),
                           desc="Processing countries"):
            country = future_to_country[future]
            try:
                # Named island_gdf, not gdf: this loop runs inside a function whose
                # own `gdf` parameter is still needed by later iterations.
                _, sum_pos, sum_neg, island_gdf = future.result()  # Raises exception if failed

                if island_gdf is not None and not island_gdf.empty:
                    island_gdf['country'] = country
                    if island_gdf.crs is None:
                        logger.warning("[%s] Extracted islands missing CRS; assuming EPSG:4326", country)
                        island_gdf = island_gdf.set_crs(4326, allow_override=True)
                    elif island_gdf.crs.to_epsg() != 4326:
                        island_gdf = island_gdf.to_crs(4326)
                    island_gdf['geometry'] = island_gdf.geometry.apply(to_wkt)
                elif island_gdf is None:
                    logger.warning("[%s] No island dataframe returned", country)

                if sum_pos is None:
                    logger.warning("[%s] Skipping stats write due to missing signed sums", country)
                    continue
                
                stats = {
                    'country' : [country],
                    'population_served' : [sum_pos],
                    'population_unserved': [abs(sum_neg)],
                    'population_total': [sum_pos + abs(sum_neg)],
                    'population_served_index': [sum_pos/(sum_pos + abs(sum_neg) + 0.1)]
                }
                stats = pd.DataFrame(stats)
                # No inner except here: a failed stats/polygon write must reach the
                # outer handler so the country is recorded as failed rather than
                # reported as "processed successfully" with no output on disk.
                _append_csv(stats, csv_output_filepath)

                if island_gdf is not None and not island_gdf.empty:
                    _append_csv(island_gdf, non_served_outpath.replace('.gpkg', '.csv'))

                logger.warning("[OK] %s: processed successfully", country)
                results[country] = True
            except Exception as e:
                logger.exception("[FAIL] %s: failed with error: %s", country, e)
                results[country] = False
    return results

def parse_args():
    """Parse sharding args and standardized named config overrides."""
    parser = argparse.ArgumentParser(
        description="Create signed rasters and unserved island stats for a country shard."
    )
    parser.add_argument("--job-index", type=int, default=0,
                        help="Index of this shard, in [0, total-jobs-1] (default: 0).")
    parser.add_argument("--total-jobs", type=int, default=1,
                        help="Total number of shards the country list is split into (default: 1).")
    add_standard_override_arguments(parser)

    args = parser.parse_args()
    return args

def shard_tif_dict(tif_dict, job_index, total_jobs, seed):
    """Split the country-to-raster mapping into a deterministic worker shard."""
    if total_jobs < 1:
        raise ValueError(f"total_jobs must be >= 1, got {total_jobs}")
    if job_index < 0 or job_index >= total_jobs:
        raise ValueError(f"job_index must be in [0, {total_jobs - 1}], got {job_index}")

    countries = sorted(tif_dict.keys())
    random.Random(seed).shuffle(countries)
    shard_countries = countries[job_index::total_jobs]
    return {country: tif_dict[country] for country in shard_countries}

def main():
    """Load configuration, prepare inputs, and run country batch processing.

    Returns
    -------
    None
        The function writes signed rasters, summary CSV rows, and non-served
        polygon outputs for the configured shard.
    """
    args = parse_args()
    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name="create_rasters", **overrides)
    provenance.write_run_manifest(cfg, "create_rasters")
    max_workers = cfg['annotations']['max_workers']
    seed = int(cfg['annotations']['random_seed'])
    min_pixels = int(cfg['min_pixels'])
    tif_dir = cfg['paths']['pop_tif_dir']

    zoom_level = int(cfg['zoom_level'])
    output_tif_dir = cfg['paths']['WWTP_tif_dir']
    non_served_outpath = os.path.abspath(cfg['paths']['non_served_outpath'].replace('.gpkg', '.csv'))
    csv_output_filepath = os.path.abspath(cfg['paths']['csv_output_filepath'].replace('.gpkg', '.csv'))

    approach = str(cfg['figures']['approach'])
    voronoi_map = create_pop_output_paths(cfg)['voronoi']
    if approach not in voronoi_map:
        raise KeyError(
            f"Configured figures.approach '{approach}' is not available; "
            f"expected one of {sorted(voronoi_map)}"
        )
    voronoi_3a_filepath = os.path.abspath(voronoi_map[approach])
    if not os.path.exists(output_tif_dir):
        os.makedirs(output_tif_dir, exist_ok=True)
    logger.info("Loading Voronoi polygons from %s", voronoi_3a_filepath)

    gdf = gpd.read_file(voronoi_3a_filepath)
    country_output_col = cfg['country_output_column']
    tif_dict = find_newest_country_tif_files(gdf[country_output_col].unique(), tif_dir)
    logger.info("Resolved %s newest country TIFF files", len(tif_dict))

    tif_dict = shard_tif_dict(tif_dict, args.job_index, args.total_jobs, seed)
    logger.info(
        "Running shard %s/%s with %s countries",
        args.job_index,
        args.total_jobs,
        len(tif_dict)
    )

    watershed_gdf = gpd.read_file(cfg['paths']['watershed'])
    if watershed_gdf.crs is None:
        raise ValueError("Watershed dataset must include CRS metadata")
    watershed_gdf = watershed_gdf.to_crs('EPSG:4326').drop_duplicates(
        subset=[cfg['basin_column_name'], 'geometry'],
        keep='first',
    ).reset_index(drop=True)
    country_boundary_col = cfg['country_boundary_column']
    if country_output_col not in watershed_gdf.columns: 
        logger.warning("Watershed %s missing; running overture enrichment", country_output_col)
        if not os.path.exists(cfg['paths']['overture']):
            download_overture_maps(cfg['paths']['overture_s3_url'], cfg['paths']['overture'])
        watershed_gdf = intersects_with_country_db(
            watershed_gdf,
            cfg['paths']['overture'],
            polygon_country_col=country_boundary_col,
            output_country_col=country_output_col,
        )
        # Persist to a dedicated path, never back onto cfg['paths']['watershed']:
        # that was a self-loop that mutated the asset combine_watersheds
        # produces and would race under concurrent/partitioned execution.
        ensure_output_dir_for_file(cfg['paths']['watershed_with_countries'])
        watershed_gdf.to_file(cfg['paths']['watershed_with_countries'], driver='GPKG', index=False)
    
    logger.info("Starting country intersection workflow with max_workers=%s", max_workers)
    orchestrate_intersections(tif_dict, gdf, watershed_gdf, output_tif_dir, csv_output_filepath, non_served_outpath, max_workers, min_pixels=min_pixels, zoom_level=zoom_level, country_col=country_output_col, basin_col=cfg['basin_column_name'],
                              all_touched=bool(cfg['served_all_touched']))
if __name__ == '__main__':
    configure_logging()
    main()
    


