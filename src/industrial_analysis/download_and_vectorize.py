#!/usr/bin/env python
"""
Download, vectorize, and merge industrial land-use raster data.

This script downloads industrial land classification rasters from Zenodo,
vectorizes them (converts rasters to polygons), merges all geometries, clips
to watershed and country boundaries, and saves the result as a GeoPackage.

Usage:
    python -m src.industrial_analysis.download_and_vectorize [level] [version] [buffer] [weight_method] [weight_func] [dynamic_buffering] [size_measure]
"""

import argparse
import sys
import os
import logging
import tempfile
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
import pandas as pd

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from .. import provenance
from ..service_areas import (
    dissolve_overlapping_geometries_fast,
    download_overture_maps,
    intersects_with_country_db,
    intersect_with_polygon_sindex,
)
from ..geo_utils import estimate_utm_epsg, repair_geometry
from ..utils import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    configure_logging,
    default_cpu_workers,
    requests_session_with_retries,
)

logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: str, chunk_size: int) -> None:
    """Download a file from URL with progress tracking."""
    logger.info(f"Downloading from {url}...")
    session = requests_session_with_retries()
    response = session.get(url, stream=True, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    if int(percent) % 10 == 0:
                        logger.debug(f"Downloaded {percent:.1f}%")
    
    logger.info(f"Downloaded to {dest_path}")

def vectorize_raster_file(raster_path: str, crs: str = "EPSG:4326", min_cells: int = 100) -> gpd.GeoDataFrame:
    """
    Vectorize a single raster file to polygons.
    
    Parameters
    ----------
    raster_path : str
        Path to the raster file.
    crs : str
        Coordinate reference system for output.
    min_cells : int
        Minimum number of connected pixels a polygon must cover to be retained.
        At 10 m resolution, 100 cells ≈ 1 ha.
    
    Returns
    -------
    geopandas.GeoDataFrame
        Vectorized polygons with values.
    """
    logger.info(f"Vectorizing {Path(raster_path).name}...")
    
    polygons = []
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        data = src.read(1)
        transform = src.transform
        
        # Minimum area filter: one pixel = |dx| * |dy| in raster native units
        pixel_area = abs(transform.a) * abs(transform.e)
        min_area = min_cells * pixel_area

        # Extract shapes from raster (convert pixels to polygons)
        for geom, value in shapes(data, transform=transform):
            if value == 1:  # Keep only class-1 industrial cells
                geom_shape = shape(geom)
                if geom_shape.area < min_area:
                    continue
                polygons.append({
                    'geometry': geom_shape,
                    'value': value
                })
    
    if not polygons:
        logger.warning(f"No valid polygons found in {raster_path}")
        return gpd.GeoDataFrame({'geometry': [], 'value': []}, crs=raster_crs or crs)
    
    gdf = gpd.GeoDataFrame(polygons, crs=raster_crs or crs)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    
    logger.info(f"Vectorized {len(gdf)} polygons from {Path(raster_path).name}")
    return gdf


def vectorize_rasters_parallel(
    raster_dir: str,
    max_workers: int = 8,
    crs: str = "EPSG:4326",
    min_cells: int = 100,
) -> List[gpd.GeoDataFrame]:
    """
    Vectorize multiple raster files in parallel.
    
    Parameters
    ----------
    raster_dir : str
        Directory containing raster files.
    max_workers : int
        Number of parallel workers.
    crs : str
        Coordinate reference system.
    
    Returns
    -------
    list of geopandas.GeoDataFrame
        Vectorized geodataframes.
    """
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")

    raster_files = list(Path(raster_dir).glob("*.tif")) + list(Path(raster_dir).glob("*.tiff"))
    
    if not raster_files:
        logger.warning(f"No raster files found in {raster_dir}")
        return []
    
    logger.info(f"Vectorizing {len(raster_files)} raster files with {max_workers} workers...")
    
    gdfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(vectorize_raster_file, str(raster_file), crs, min_cells)
            for raster_file in raster_files
        ]
        
        for idx, future in enumerate(futures, 1):
            try:
                gdf = future.result()
                gdfs.append(gdf)
                logger.info(f"Completed {idx}/{len(futures)} rasters")
            except Exception as e:
                logger.error(f"Error vectorizing raster: {e}")
    
    return gdfs


def _repair_geometry(geom):
    """Repair invalid geometries before overlay operations."""
    return repair_geometry(geom)


def _dissolve_by_overlap_groups(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve original geometries using fast overlap groups from service_areas."""
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(gdf, "crs", None))

    working = gdf.reset_index(drop=True).copy()
    working["some_id"] = working.index.astype(int)

    overlap_groups, _ = dissolve_overlapping_geometries_fast(
        working[["some_id", "geometry"]].copy(),
        radius=0,
        convex=True,
    )

    group_map = {}
    for group_index, group in enumerate(overlap_groups):
        for some_id in group:
            group_map[int(some_id)] = group_index

    working["group_id"] = working["some_id"].map(group_map)
    missing_mask = working["group_id"].isna()
    if missing_mask.any():
        start = len(overlap_groups)
        working.loc[missing_mask, "group_id"] = list(range(start, start + int(missing_mask.sum())))
    working["group_id"] = working["group_id"].astype(int)

    dissolved = working[["group_id", "geometry"]].dissolve(by="group_id").reset_index(drop=True)
    dissolved["geometry"] = dissolved.geometry.map(_repair_geometry)
    dissolved = dissolved[dissolved.geometry.notna() & ~dissolved.geometry.is_empty].copy()
    return gpd.GeoDataFrame(dissolved, geometry="geometry", crs=gdf.crs)


def _morton_code(x: float, y: float, max_depth: int = 16) -> int:
    """Compute Morton code (Z-order curve) for a 2D point to enable spatial clustering."""
    # Normalize to [0, 2^max_depth)
    xi = max(0, min(int(x * (1 << max_depth)), (1 << max_depth) - 1))
    yi = max(0, min(int(y * (1 << max_depth)), (1 << max_depth) - 1))
    
    # Interleave bits
    morton = 0
    for i in range(max_depth):
        morton |= ((xi >> i) & 1) << (2 * i)
        morton |= ((yi >> i) & 1) << (2 * i + 1)
    return morton


def _iter_batches_by_proximity(geoms: List, batch_size: int):
    """Yield geometry batches grouped by Morton index proximity.

    Uses a spatial index key (Morton code) so nearby geometries are unioned together.
    """
    if not geoms:
        return

    logger.info("Computing centroid keys for %d geometries...", len(geoms))

    centroids = []
    for geom in geoms:
        if geom is None or geom.is_empty:
            centroids.append((0.0, 0.0))
            continue
        c = geom.centroid
        centroids.append((c.x, c.y))

    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0

    indexed_codes = [
        (
            i,
            _morton_code(
                (centroids[i][0] - min_x) / range_x,
                (centroids[i][1] - min_y) / range_y,
            ),
        )
        for i in range(len(geoms))
    ]
    logger.info("Sorting %d geometries by Morton code...", len(indexed_codes))
    indexed_codes.sort(key=lambda pair: pair[1])

    total_batches = (len(indexed_codes) + batch_size - 1) // batch_size
    logger.info("Prepared %d proximity batches", total_batches)

    for batch_start in range(0, len(indexed_codes), batch_size):
        batch_indices = [idx for idx, _ in indexed_codes[batch_start: batch_start + batch_size]]
        yield [geoms[idx] for idx in batch_indices]


def _incremental_union_batch(geoms_batch: List):
    """Perform incremental union on a batch of geometries (for parallel execution)."""
    if not geoms_batch:
        return None
    if len(geoms_batch) == 1:
        return geoms_batch[0]
    
    result = geoms_batch[0]
    for geom in geoms_batch[1:]:
        result = result.union(geom)
    return result


def _chunked_unary_union(geoms: List, chunk_size: int = 32):
    """Union geometries via chunked tree-reduction to avoid O(n)-style serial blowups."""
    if not geoms:
        return None

    current = [g for g in geoms if g is not None and not g.is_empty]
    if not current:
        return None

    round_num = 0
    while len(current) > 1:
        round_num += 1
        logger.info("Final merge round %d: reducing %d geometries", round_num, len(current))
        next_round = []
        for i in range(0, len(current), chunk_size):
            chunk = current[i:i + chunk_size]
            merged = unary_union(chunk)
            repaired = _repair_geometry(merged)
            if repaired is not None and not repaired.is_empty:
                next_round.append(repaired)
        current = next_round
        if not current:
            return None
    return current[0]


def _process_and_extract_geoms(gdf: gpd.GeoDataFrame, simplify_tolerance: float):
    """Yield cleaned geometries from one GeoDataFrame with minimal copying."""
    if gdf is None or gdf.empty:
        return

    geometries = gdf.geometry.map(_repair_geometry)
    valid_mask = geometries.notna() & ~geometries.is_empty
    geometries = geometries[valid_mask]
    if geometries.empty:
        return

    if simplify_tolerance is not None:
        geometries = geometries.simplify(tolerance=simplify_tolerance, preserve_topology=True)
        geometries = geometries.map(_repair_geometry)
        valid_mask = geometries.notna() & ~geometries.is_empty
        geometries = geometries[valid_mask]
        if geometries.empty:
            return

    for geom in geometries.values:
        yield geom


def merge_geodataframes(
    gdfs: List[gpd.GeoDataFrame],
    simplify_tolerance: float = 0.01,
    max_workers: int = 8,
    batch_size: int = 5000,
) -> gpd.GeoDataFrame:
    """Merge multiple GeoDataFrames incrementally with parallel batch processing.
    
    Parallelizes input GeoDataFrame processing, repairs/simplifies geometries,
    batches them, and uses ProcessPoolExecutor to compute incremental unions in parallel.
    
    Parameters
    ----------
    gdfs : List[gpd.GeoDataFrame]
        List of geodataframes to merge.
    simplify_tolerance : float
        Tolerance (in degrees) for geometry simplification. Default 0.01 (~1.1km at equator).
        Set to None to skip simplification.
    max_workers : int
        Number of worker processes for parallel batch union and input processing.
    batch_size : int
        Number of geometries per batch for parallel processing. Larger batches = fewer tasks.
    
    Returns
    -------
    gpd.GeoDataFrame
        Merged and exploded geodataframe with individual polygons.
    """
    if not gdfs:
        raise ValueError("No geodataframes to merge")

    if max_workers is None:
        worker_count = default_cpu_workers()
    else:
        try:
            worker_count = int(max_workers)
        except (TypeError, ValueError) as err:
            raise ValueError("max_workers must be a positive integer or None") from err
    if worker_count < 1:
        raise ValueError("max_workers must be a positive integer or None")

    try:
        batch_size_int = int(batch_size)
    except (TypeError, ValueError) as err:
        raise ValueError("batch_size must be >= 1") from err
    if batch_size_int < 1:
        raise ValueError("batch_size must be >= 1")
    
    logger.info(f"Merging {len(gdfs)} geodataframes (parallel mode)...")
    target_crs = gdfs[0].crs or "EPSG:4326"
    
    # Parallel processing of input GeoDataFrames: repair, simplify, extract geometries
    logger.info(f"Processing {len(gdfs)} input GeoDataFrames with {worker_count} workers...")
    all_geoms = []

    if worker_count <= 1:
        for idx, gdf in enumerate(gdfs, 1):
            logger.info(f"Processing GeoDataFrame {idx}/{len(gdfs)}")
            count_before = len(all_geoms)
            all_geoms.extend(_process_and_extract_geoms(gdf, simplify_tolerance))
            logger.info(f"Completed GeoDataFrame {idx}/{len(gdfs)}: {len(all_geoms) - count_before} geometries")
    else:
        pool_size = min(worker_count, len(gdfs))
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            for idx, geoms_iter in enumerate(
                executor.map(_process_and_extract_geoms, gdfs, [simplify_tolerance] * len(gdfs)),
                1,
            ):
                try:
                    count_before = len(all_geoms)
                    all_geoms.extend(geoms_iter)
                    logger.info(f"Completed GeoDataFrame {idx}/{len(gdfs)}: {len(all_geoms) - count_before} geometries")
                except Exception as e:
                    logger.error(f"Error processing GeoDataFrame {idx}: {e}")
    
    if not all_geoms:
        raise ValueError("No valid industrial polygons found across all inputs")
    
    logger.info(f"Accumulated {len(all_geoms)} total geometries")

    n_batches = (len(all_geoms) + batch_size_int - 1) // batch_size_int
    logger.info(
        "Processing %d geometries in %d proximity batches with %d worker(s)...",
        len(all_geoms), n_batches, worker_count,
    )

    merged_geom = None
    batches_iter = _iter_batches_by_proximity(all_geoms, batch_size_int)
    batch_union_iter = ()
    if worker_count <= 1 or n_batches == 1:
        batch_union_iter = (_incremental_union_batch(batch) for batch in batches_iter)
    else:
        pool_size = min(worker_count, n_batches)
        with ProcessPoolExecutor(max_workers=pool_size) as executor:
            future_to_batch_idx = {}
            for batch_idx, batch in enumerate(batches_iter, 1):
                future = executor.submit(_incremental_union_batch, batch)
                future_to_batch_idx[future] = batch_idx
            merged_batch_geoms = []
            merged_count = 0
            failed_batches = []
            for future in as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                try:
                    batch_geom = future.result()
                except Exception as err:
                    logger.exception("Proximity batch %d/%d failed: %s", batch_idx, n_batches, err)
                    failed_batches.append((batch_idx, err))
                    continue
                repaired = _repair_geometry(batch_geom) if batch_geom is not None else None
                if repaired is None or repaired.is_empty:
                    if batch_idx % 5 == 0 or batch_idx == n_batches:
                        logger.info(
                            "Processed proximity batch %d/%d (current merged count: %d)",
                            batch_idx,
                            n_batches,
                            merged_count,
                        )
                    continue
                merged_batch_geoms.append(repaired)
                merged_count += 1
                if batch_idx % 5 == 0 or batch_idx == n_batches:
                    logger.info(
                        "Processed proximity batch %d/%d (current merged count: %d)",
                        batch_idx,
                        n_batches,
                        merged_count,
                    )
            if failed_batches:
                first_idx, first_err = failed_batches[0]
                raise RuntimeError(
                    f"{len(failed_batches)} of {n_batches} proximity batch(es) failed; "
                    f"first failure was batch {first_idx}: {first_err}"
                )
            merged_geom = _chunked_unary_union(merged_batch_geoms)
            if merged_geom is None:
                raise ValueError("No valid geometry after batch processing")
            logger.info("Merged %d non-empty batch result(s)", merged_count)

    if worker_count <= 1 or n_batches == 1:
        merged_batch_geoms = []
        merged_count = 0
        for batch_geom in batch_union_iter:
            repaired = _repair_geometry(batch_geom) if batch_geom is not None else None
            if repaired is None or repaired.is_empty:
                continue
            merged_batch_geoms.append(repaired)
            merged_count += 1
        merged_geom = _chunked_unary_union(merged_batch_geoms)
        if merged_geom is None:
            raise ValueError("No valid geometry after batch processing")
        logger.info("Merged %d non-empty batch result(s)", merged_count)
    
    # Repair final merged geometry
    merged_geom = _repair_geometry(merged_geom)
    if merged_geom is None or merged_geom.is_empty:
        raise ValueError("No valid geometry after merge")
    
    # Create GeoDataFrame and explode
    temp_gdf = gpd.GeoDataFrame(
        {'geometry': [merged_geom], 'category': ['industrial_land']},
        crs=target_crs
    )
    
    if target_crs is not None and temp_gdf.crs != target_crs:
        temp_gdf = temp_gdf.to_crs(target_crs)
    
    logger.info("Exploding multipart geometries into individual features...")
    result = temp_gdf.explode(index_parts=False, ignore_index=True)
    
    logger.info(f"Merged and exploded GeoDataFrame with {len(result)} feature(s)")
    return result


def add_boundary_info(
    industrial_gdf: gpd.GeoDataFrame,
    watershed_gdf: gpd.GeoDataFrame,
    overture_path: str,
    overture_s3_url: str,
    basin_col: str,
    sindex_concurrency: bool,
    country_boundary_col: str = 'country',
    country_output_col: str = 'ISO_2',
) -> gpd.GeoDataFrame:
    """Add country and basin attributes without clipping industrial geometry."""
    if industrial_gdf is None or industrial_gdf.empty:
        empty = industrial_gdf.copy() if industrial_gdf is not None else gpd.GeoDataFrame(geometry=[], crs=getattr(watershed_gdf, "crs", None))
        if country_output_col not in empty.columns:
            empty[country_output_col] = pd.Series(dtype="object")
        if basin_col not in empty.columns:
            empty[basin_col] = pd.Series(dtype="object")
        return gpd.GeoDataFrame(empty, geometry="geometry", crs=getattr(empty, "crs", None))

    if basin_col not in watershed_gdf.columns:
        raise KeyError(f"Watershed column '{basin_col}' not found in watershed dataset.")

    if not os.path.exists(overture_path):
        logger.info("Overture boundary parquet not found. Downloading...")
        download_overture_maps(overture_s3_url, overture_path)

    # Work on a copy and preserve original geometry so helper side-effects
    # do not alter output shapes.
    enriched = industrial_gdf.reset_index(drop=True).copy()
    enriched["__row_id"] = enriched.index.astype(int)
    original_geometry = enriched[["__row_id", "geometry"]].rename(columns={"geometry": "_original_geometry"})

    logger.info("Adding %s via service_areas.intersects_with_country_db...", country_output_col)
    enriched = intersects_with_country_db(
        enriched,
        overture_path,
        polygon_country_col=country_boundary_col,
        output_country_col=country_output_col,
    )

    logger.info(f"Adding basin info via service_areas.intersect_with_polygon_sindex ({basin_col})...")
    if enriched.crs is not None and enriched.crs != watershed_gdf.crs:
        watershed_gdf = watershed_gdf.to_crs(enriched.crs)
    enriched = intersect_with_polygon_sindex(
        enriched,
        watershed_gdf,
        basin_col,
        concurrency=sindex_concurrency,
    )

    # Restore original shapes and row order.
    enriched = enriched.drop(columns=["geometry"]).merge(original_geometry, on="__row_id", how="left")
    enriched = enriched.sort_values("__row_id").reset_index(drop=True)
    enriched = enriched.rename(columns={"_original_geometry": "geometry"})
    enriched = enriched.drop(columns=["__row_id"], errors="ignore")
    return gpd.GeoDataFrame(enriched, geometry="geometry", crs=industrial_gdf.crs)


def _find_raster_dirs(base_dir: str) -> List[str]:
    """Walk *base_dir* and return every subdirectory that contains .tif/.tiff files."""
    raster_dirs = []
    for root, _dirs, files in os.walk(base_dir):
        if any(f.lower().endswith((".tif", ".tiff")) for f in files):
            raster_dirs.append(root)
    return raster_dirs


def _vectorize_and_merge(
    raster_dirs: List[str],
    max_workers: int,
    min_cells: int,
    simplify_tolerance: float = 0.01,
    batch_size: int = 5000,
) -> gpd.GeoDataFrame:
    """Vectorize all rasters in *raster_dirs* and dissolve into a single GeoDataFrame."""
    if not raster_dirs:
        raise FileNotFoundError("No raster directories found")
    gdfs = []
    for raster_dir in raster_dirs:
        logger.info(f"Vectorizing rasters in directory: {raster_dir}")
        gdfs.extend(
            vectorize_rasters_parallel(raster_dir, max_workers=max_workers, crs="EPSG:4326", min_cells=min_cells)
        )
    if not gdfs:
        raise ValueError("Failed to vectorize any raster files")
    return merge_geodataframes(
        gdfs,
        simplify_tolerance=simplify_tolerance,
        max_workers=max_workers,
        batch_size=batch_size,
    )


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(
        description="Download, vectorize, and merge industrial land rasters."
    )
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main():
    """Download, vectorize, and merge industrial land data."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="download_and_vectorize", **overrides)
    provenance.write_run_manifest(cfg, "download_and_vectorize")

    vectorized_path = cfg['paths']['industrial_merged_filepath']
    overwrite = cfg['overwrite_existing']
    min_cells = cfg['industrial_min_cells']
    if int(min_cells) < 1:
        raise ValueError("industrial_min_cells must be >= 1")
    persist_rasters = cfg['industrial_persist_rasters']
    simplify_tolerance = cfg['industrial_simplify_tolerance']
    industrial_batch_size = cfg['industrial_batch_size']
    if int(industrial_batch_size) < 1:
        raise ValueError("industrial_batch_size must be >= 1")
    download_chunk_size = int(cfg['industrial_download_chunk_size'])

    # Intermediate file: merged vectorized polygons (pre-enrichment).
    # Named after min_cells so different thresholds don't clobber each other.
    industrial_analysis_dir = os.path.dirname(vectorized_path)
    merged_gdf = None

    try:
        # ------------------------------------------------------------------ #
        # Step 1 – obtain merged_gdf (from cache or fresh vectorization)      #
        # ------------------------------------------------------------------ #
        if os.path.exists(vectorized_path) and not overwrite:
            logger.info(f"Loading cached vectorized polygons from {vectorized_path}")
            merged_gdf = gpd.read_parquet(vectorized_path)
        else:
            if persist_rasters:
                # Rasters are kept on disk so future runs can skip download.
                raster_base_dir = cfg['paths']['industrial_raster_persistent_dir']
                os.makedirs(raster_base_dir, exist_ok=True)
                existing = (
                    list(Path(raster_base_dir).rglob("*.tif"))
                    + list(Path(raster_base_dir).rglob("*.tiff"))
                )
                if existing and not overwrite:
                    logger.info(f"Reusing {len(existing)} existing raster(s) in {raster_base_dir}")
                else:
                    zip_path = os.path.join(raster_base_dir, "industrial_land.zip")
                    download_file(cfg['industrial_zenodo_url'], zip_path, chunk_size=download_chunk_size)
                    logger.info(f"Extracting to {raster_base_dir}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(raster_base_dir)
                raster_dirs = _find_raster_dirs(raster_base_dir)
            else:
                # Use a temporary directory; rasters are discarded after vectorization.
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_path = os.path.join(temp_dir, "industrial_land.zip")
                    extract_dir = os.path.join(temp_dir, "extracted")
                    os.makedirs(extract_dir, exist_ok=True)
                    download_file(cfg['industrial_zenodo_url'], zip_path, chunk_size=download_chunk_size)
                    logger.info(f"Extracting to {extract_dir}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    raster_dirs = _find_raster_dirs(extract_dir)
                    merged_gdf = _vectorize_and_merge(
                        raster_dirs,
                        cfg['max_workers'],
                        min_cells,
                        simplify_tolerance,
                        industrial_batch_size,
                    )

            if persist_rasters:
                # _vectorize_and_merge is called outside the with-block for the
                # persistent-dir branch so the rasters remain accessible.
                merged_gdf = _vectorize_and_merge(
                    raster_dirs,
                    cfg['max_workers'],
                    min_cells,
                    simplify_tolerance,
                    industrial_batch_size,
                )

            # Save intermediate result so boundary enrichment can be re-run
            # independently without repeating download + vectorization.
            os.makedirs(industrial_analysis_dir, exist_ok=True)
            logger.info(f"Saving vectorized polygons to {vectorized_path}...")
            if merged_gdf is None:
                raise ValueError("Vectorization did not produce any industrial polygons")
            vectorized_tmp_path = f"{vectorized_path}.tmp"
            if os.path.exists(vectorized_tmp_path):
                os.remove(vectorized_tmp_path)
            merged_gdf.to_parquet(vectorized_tmp_path, index=False)
            os.replace(vectorized_tmp_path, vectorized_path)
            logger.info(f"Saved {len(merged_gdf)} feature(s) to {vectorized_path}")

        # ------------------------------------------------------------------ #
        # Step 2 – enrich with country / basin boundaries                     #
        # ------------------------------------------------------------------ #
        logger.info("Loading watershed data...")
        watershed_gdf = gpd.read_file(cfg['paths']['watershed'], driver='GPKG')

        enriched_gdf = add_boundary_info(
            merged_gdf,
            watershed_gdf,
            cfg['paths']['overture'],
            cfg['paths']['overture_s3_url'],
            cfg['basin_column_name'],
            cfg['sindex_concurrency'],
            cfg['country_boundary_column'],
            cfg['country_output_column'],
        )

        logger.info(f"Writing to {vectorized_path}...")
        enriched_tmp_path = f"{vectorized_path}.tmp"
        if os.path.exists(enriched_tmp_path):
            os.remove(enriched_tmp_path)
        enriched_gdf.to_parquet(enriched_tmp_path, index=False)
        os.replace(enriched_tmp_path, vectorized_path)
        logger.info(f"Successfully created {vectorized_path}")
        return True

    except Exception as e:
        logger.error(f"Error during vectorization: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    configure_logging()
    success = main()
    sys.exit(0 if success else 1)
