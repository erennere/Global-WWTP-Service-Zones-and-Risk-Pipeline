"""The weighted Voronoi solver: site clustering, grid assignment, regions, circle cut.

Part of the service-area stage (``python -m wwtp.create_voronoi``).
"""

import logging
from typing import Any, cast
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point, Polygon, LineString, MultiPolygon, MultiLineString, box, vectorized
from shapely.ops import unary_union
import shapely.affinity
import shapely
from ..geo_utils import buffer_geometry, cluster_point_indices, estimate_utm_crs
from .size import calculate_buffer, plant_size_people
from .weights import default_distance_multiplicative, initialize_voronoi_weights
from .contours import extract_contours_cv2, extract_contours_rasterio, extract_contours_scipy
from .joins import create_centroid_points, drop_duplicates
from .overlaps import resolve_polygon_overlaps

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
        ``weights`` and ``POP_SERVED`` the cluster maximum (never the sum: the
        records are one plant reported by several sources)
        and ``n_merged_records`` the number of records merged.

    Notes:
        Uses centroid-to-centroid distance for clustering
        Union-Find ensures transitive closure (A~B and B~C → A~C)
    """

    df = df.copy()
    df["num_missing"] = df.isnull().sum(axis=1)

    cluster_sets = cluster_point_indices(df["geometry"], threshold)
    data = []

    for cluster_set in cluster_sets:
        sub_df = df.iloc[list(cluster_set)]
        if len(sub_df) == 1:
            data.append(sub_df.assign(n_merged_records=1))
        else:
            weights = sub_df["weights"].max()
            # Get the row(s) with the fewest NaNs
            min_missing = sub_df["num_missing"].min()
            best_rows = sub_df[sub_df["num_missing"] == min_missing]
            best_row = best_rows.iloc[0].copy()  # Choose the first one if there's a tie
            best_row["weights"] = weights
            if "POP_SERVED" in sub_df.columns:
                # Records this close are the same plant reported by several
                # sources; adding their populations double-counts it.
                best_row["POP_SERVED"] = pd.to_numeric(sub_df["POP_SERVED"], errors="coerce").max()
            best_row["n_merged_records"] = len(sub_df)
            data.append(pd.DataFrame([best_row]))
    df = pd.concat(data, ignore_index=True)
    df = df.drop(columns=["num_missing"])
    return df


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
    for geom in df["geometry"].apply(create_centroid_points):
        if isinstance(geom, Point):
            points.append((geom.x, geom.y))
        elif isinstance(geom, (LineString, MultiLineString, Polygon, MultiPolygon)):
            points.append((geom.centroid.x, geom.centroid.y))
        else:
            points.append((None, None))
    logger.debug(f"Computed {len(points)} site centroids")

    return points


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
    df = gpd.GeoDataFrame(df, geometry="geometry", crs="epsg:4326")
    if len(df) > 0:
        df["geometry"] = df["geometry"].map(buffer_geometry)
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


def weighted_voronoi(
    df,
    col,
    country_clip,
    scale_weights=False,
    clipping=None,
    n_points=100,
    distance_fn=default_distance_multiplicative,
    scipy_true=False,
    cv2_true=False,
    centroid_points=False,
    buffering=False,
    threshold=500,
    calculate_buffer_fn=calculate_buffer,
    buffer_fn_kwargs=None,
    site_id_col="WASTE_ID",
    size_weights=False,
):
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
        df = df.set_crs("epsg:4326")
        utm = estimate_utm_crs(df)
        df = df.to_crs(utm)
        logger.debug(f"Projected to UTM CRS: {utm}")
    crs = df.crs

    # === PHASE 2: SITE PREPROCESSING ===
    # Cluster nearby points to avoid creating duplicate Voronoi regions
    # Points within 'threshold' distance are merged together with aggregate properties
    df = gpd.GeoDataFrame(cluster_points(df, threshold), geometry="geometry", crs=crs)
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
        raise ValueError(f"calculate_buffer_fn returned {len(buffer_lengths)} buffer lengths for {len(df)} rows")
    df["buffer_length"] = buffer_lengths
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
            clipping = clipping.set_crs("epsg:4326")
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
            country_clip = country_clip.set_crs("epsg:4326")
        if country_clip.crs != crs:
            country_clip = country_clip.to_crs(crs)

    if len(df) == 1:
        # === PHASE 7: SPECIAL CASE (SINGLE SITE) ===
        # When only one site exists, create Voronoi region from clipping boundary
        # This is NOT a true Voronoi diagram, but the site's service area
        region_polygons = df.copy()
        point_df = df.copy().reset_index(drop=True)
        region_geom = df.iloc[0]["geometry"].buffer(df.iloc[0]["buffer_length"])
        geom = df.iloc[0]["geometry"]

        if isinstance(region_geom, (Point, Polygon, MultiPolygon, LineString, MultiLineString)):
            region_geom = buffer_geometry(region_geom)  # type: ignore[index]

        # Optionally intersect region with buffer around site point
        if buffering:
            point_buffer = geom.centroid.buffer(df.iloc[0]["buffer_length"])
            region_geom = region_geom.intersection(point_buffer).buffer(0)  # type: ignore[union-attr, index]
        region_polygons.at[region_polygons.index[0], "geometry"] = cast(Any, region_geom)  # type: ignore[index]

        # Clip region to actual clipping geometry to ensure it does not exceed bounds
        region_polygons = gpd.clip(region_polygons, actual_clipping_object)  # type: ignore[arg-type]
        region_polygons["geometry"] = region_polygons["geometry"].map(buffer_geometry)

        # Apply country boundary clipping if provided
        if country_clip is not None and not country_clip.empty:
            region_polygons = gpd.clip(region_polygons, country_clip)
        region_polygons["geometry"] = region_polygons["geometry"].map(buffer_geometry)

        # Same 'area' column the multi-site branch emits, computed in the metric
        # CRS before reprojecting - without it downstream consumers see the
        # column only for multi-site groups.
        region_polygons["area"] = region_polygons.geometry.area

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
    # A grid point stands for the square cell around it. Keeping only points
    # inside the clip dropped every cell whose centre fell outside, so each
    # region lost a band of half a grid step along the clip edge (2 % of a
    # 10 km box at 100 m). Keep every cell that touches the clip; the final
    # clip below trims the regions back to the exact boundary.
    half_cell = 0.5 * float(n_points) * np.sqrt(2.0)
    mask = geometry_contains_points(actual_clipping_object.buffer(half_cell), grid_points)
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
            region_polygons.append({site_id_col: row[site_id_col], "geometry": None})
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
                point_buffer = Point(point).buffer(row["buffer_length"])
                polygons = polygons.intersection(point_buffer).buffer(0)
            region_polygons.append({site_id_col: row[site_id_col], "geometry": polygons})
        else:
            # No contours found for this site
            region_polygons.append({site_id_col: row[site_id_col], "geometry": None})

    del assignment_grid_2d, assignments, mask

    # === PHASE 11: GEODATAFRAME CONVERSION & DEDUPLICATION ===
    # Convert region list to DataFrame for further processing
    region_polygons = pd.DataFrame(region_polygons)
    region_polygons = pd.merge(region_polygons, df.drop(["geometry"], axis=1), on=[site_id_col])
    region_polygons = gpd.GeoDataFrame(region_polygons, geometry="geometry", crs=crs)
    region_polygons["geometry"] = region_polygons["geometry"].map(buffer_geometry)
    region_polygons = drop_duplicates(region_polygons, site_id_col)

    # === PHASE 12: OVERLAP RESOLUTION ===
    # Remove overlapping areas between adjacent Voronoi regions
    # Each region intersection is assigned to larger polygon via area comparison
    non_intersecting_polygons = resolve_polygon_overlaps(region_polygons)
    region_polygons["geometry"] = non_intersecting_polygons
    region_polygons["geometry"] = region_polygons["geometry"].map(buffer_geometry)

    # === PHASE 13: FINAL BOUNDARY CLIPPING ===
    # Clip regions to computed bounding box
    region_polygons = gpd.clip(region_polygons, actual_clipping_object)  # type: ignore[arg-type]
    region_polygons["geometry"] = region_polygons["geometry"].map(buffer_geometry)

    # Clip regions to country boundary if provided (second clipping operation)
    if country_clip is not None and not country_clip.empty:
        region_polygons = gpd.clip(region_polygons, country_clip)
        region_polygons["geometry"] = region_polygons["geometry"].map(buffer_geometry)

    # Area is measured on the final (clipped) polygons, in the metric CRS,
    # so it matches the geometry that is actually written out.
    region_polygons = region_polygons[region_polygons.geometry.notna() & ~region_polygons.geometry.is_empty].copy()
    region_polygons["area"] = region_polygons.geometry.area

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
        (
            sub_df,
            col,
            country_clip,
            scale_weights,
            clipping,
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
        ) = args
        logger.debug(f"voronoi_worker: Unpacked arguments for {len(sub_df)} sites")
    except Exception as err:
        logger.error(f"voronoi_worker: Error unpacking arguments: {err}")
        raise
    return weighted_voronoi(
        sub_df,
        col,
        country_clip,
        scale_weights,
        clipping,
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
