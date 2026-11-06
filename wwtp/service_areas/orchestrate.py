"""Running the solver over all basin/buffer groups in parallel, with checkpoints.

Part of the service-area stage (``python -m wwtp.create_voronoi``).
"""

import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
from ..geo_utils import estimate_utm_crs
from ..utils import ensure_output_dir_for_file
from .size import calculate_buffer
from .weights import create_weights, default_distance_multiplicative
from .joins import normalize_column_to_rounded_str
from .solver import finalize_gdf, voronoi_worker

logger = logging.getLogger(__name__)


def orchestrate_voronoi_weights(
    df,
    col,
    country_df,
    workers=12,
    scale_weights=False,
    clipping=None,
    n_points=100,
    distance_fn=default_distance_multiplicative,
    scipy_true=False,
    cv2_true=False,
    centroid_points=False,
    buffering=False,
    threshold=500,
    sigma=3,
    percent_threshold=10,
    area_fn=None,
    area_fn_kwargs=None,
    method="linear",
    output_path=None,
    overwrite=False,
    flush_size=None,
    calculate_buffer_fn=calculate_buffer,
    buffer_fn_kwargs=None,
    site_country_col="ISO_2",
    country_boundary_col="country",
    site_id_col="WASTE_ID",
):
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
    size_weights = method == "size"

    # Group both df and clipping by the same column
    if output_path:
        logger.info(
            f"Starting orchestrate_voronoi_weights for {len(df)} sites with {workers} workers (target={output_path})"
        )
    else:
        logger.info(f"Starting orchestrate_voronoi_weights for {len(df)} sites with {workers} workers")
    if flush_size is None:
        flush_size = max(1, workers)
    temp_output_path = (
        os.path.join(os.path.dirname(output_path), f"temp_{os.path.basename(output_path)}") if output_path else None
    )

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
                logger.warning(
                    "Could not read temp checkpoint %s (%s). Continuing without resume.", temp_output_path, err
                )
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
        "generated": 0,
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
                task_stats["generated"] += 1
                task = (
                    sub_df,
                    col,
                    country_clip,
                    scale_weights,
                    sub_clip,
                    n_points,
                    distance_fn,
                    scipy_true,
                    cv2_true,
                    centroid_points,
                    buffering,
                    threshold,
                    calculate_buffer_fn,
                    buffer_fn_kwargs,
                    site_id_col,
                    size_weights,
                )
                batch.append(task)
            else:
                for country in country_iso_2:
                    country_sub_df = sub_df[sub_df[site_country_col] == country].copy().reset_index(drop=True)
                    if country_sub_df.empty:
                        continue
                    country_sub_clip = (
                        country_clip[country_clip[country_boundary_col] == country].copy().reset_index(drop=True)
                    )
                    task_stats["generated"] += 1
                    task = (
                        country_sub_df,
                        col,
                        country_sub_clip,
                        scale_weights,
                        sub_clip,
                        n_points,
                        distance_fn,
                        scipy_true,
                        cv2_true,
                        centroid_points,
                        buffering,
                        threshold,
                        calculate_buffer_fn,
                        buffer_fn_kwargs,
                        site_id_col,
                        size_weights,
                    )
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
                region_chunk.to_file(temp_output_path, mode="a", driver="GPKG", index=False)
            else:
                region_chunk.to_file(temp_output_path, driver="GPKG", index=False)

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

    if task_stats["generated"] == 0:
        logger.warning("No Voronoi tasks were generated")
    else:
        logger.debug(
            "Processed %s Voronoi task results (generated=%s)",
            processed_results,
            task_stats["generated"],
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
                logger.info(
                    "Orchestrate Voronoi complete: processed=%s (generated=%s) -> %s",
                    processed_results,
                    task_stats["generated"],
                    output_path,
                )
                return True
            except Exception as err:
                logger.error("Failed to finalize temp checkpoint %s -> %s: %s", temp_output_path, output_path, err)
                return False
        logger.warning("No temp checkpoint produced for %s", output_path)
        return False

    # Legacy tuple-return mode when no output path is provided.
    region_df_final = finalize_gdf(region_df_all, df.columns)
    point_df_final = finalize_gdf(point_df_all, df.columns)
    logger.info(f"Orchestrate Voronoi complete: {len(region_df_final)} regions, {len(point_df_final)} points")
    return region_df_final, point_df_final
