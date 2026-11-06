"""Voronoi site weights and weighted distance functions.

Part of the service-area stage (``python -m wwtp.create_voronoi``).
"""
import logging
import numpy as np
import pandas as pd
from ..geo_utils import nearest_neighbor_distances

logger = logging.getLogger(__name__)


def normalize_plane(a, b):
    """Normalize 2-D coordinates by one isotropic scale (the larger extent).

    Both axes are shifted by their minimum and divided by the SAME scalar,
    the larger of the x and y extents, so Euclidean distances keep their
    shape: the result is ``(points - min) / max(width, height)``. Scaling the
    axes separately (the previous behaviour) stretched the shorter axis and
    moved Voronoi boundaries whenever the grid was not square.

    Parameters
    ----------
    a : numpy.ndarray
        Array of shape ``(n, 2)`` with ``(x, y)`` coordinates.
    b : tuple or array-like
        Single ``(x, y)`` coordinate pair.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(a_normalized, b_normalized)``; the longer axis spans ``[0, 1]``.
        A degenerate extent of 0 is treated as 1.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    all_points = np.vstack([a, b.reshape(1, 2)])
    min_vals = np.min(all_points, axis=0)
    extent = float(np.max(np.max(all_points, axis=0) - min_vals))
    scale = extent if extent > 0 else 1.0
    return (a - min_vals) / scale, (b - min_vals) / scale


def auto_weight_scale(points):
    """
    Compute automatic weight scaling factor based on median nearest neighbor distance.
    
    Calculates the median distance between each point and its nearest neighbor.
    This is used to normalize weights in distance-based weighting functions.
    
    Args:
        points (list or np.ndarray): Array of shape (n, 2) with point coordinates
        
    Returns:
        float: Median of minimum inter-point distances
        
    Notes:
        Used for weight scaling in additive distance functions.
    """
    points = [(x, y) for x, y in points if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    # nanmedian, not nanmean: the name, the docstring and every caller's use of
    # the value all say median, and the O(n^2) pdist/squareform matrix this
    # replaced was only ever reduced to each row's minimum.
    return float(np.nanmedian(nearest_neighbor_distances(points, neighbors=1)))


def default_distance_additive(a, b, weight, factor):
    """
    Additive weighted distance function for Voronoi weighting.
    
    Returns the power distance in the normalized plane, ``|a - b|^2 - w^2``
    (Laguerre / power-diagram weighting): a site with weight ``w`` claims a
    disc of radius ``w`` (a fraction of the grid's longer extent) outright and
    competes beyond it. The value can be negative inside that disc; it is only
    ever compared, never used as a length.
    
    Args:
        a (np.ndarray): Grid points array shape (n, 2)
        b (tuple): Single point coordinate (x, y)
        weight (float): Weight parameter (affects distance directly)
        factor (float): Optional scale normalization factor. When > 0,
            effective additive weight is computed as ``weight / factor``.
        
    Returns:
        np.ndarray: Power distances for all points in a (smaller = closer)

    Notes:
        The previous ``sqrt(max(d^2 - w^2, 0.01))`` replaced every point inside
        a site's own disc by 0.1, so those points ranked *behind* points just
        outside the disc. The signed power distance is monotonic in true
        distance for every site.
    """
    a, b = normalize_plane(a, b)
    effective_weight = weight / factor if factor and factor > 0 else weight
    return np.sum((a - b) ** 2, axis=-1) - effective_weight ** 2


def default_distance_multiplicative(a, b, weight, factor):
    """
    Multiplicative weighted distance function for Voronoi weighting.
    
    Computes normalized Euclidean distance scaled inversely by weight.
    weight > 1 contracts the Voronoi region, weight < 1 expands it.
    
    Args:
        a (np.ndarray): Grid points array shape (n, 2)
        b (tuple): Single point coordinate (x, y)
        weight (float): Weight parameter (affects scaling - 1/weight)
        factor (float): Unused in multiplicative metric
        
    Returns:
        np.ndarray: Weighted distances for all points in a
        
    Notes:
        Division by weight means weight=0 causes division error (not handled).
    """
    a, b = normalize_plane(a, b)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))/weight


def initialize_voronoi_weights(df, distance_fn, scale_weights, points):
    """
    Initialize weight parameters for Voronoi computation based on distance function type.
    
    Handles weight setup differently depending on whether additive or multiplicative
    distance metrics are used. Optionally applies weight scaling based on nearest neighbors.
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input sites with a ``weights`` column.
    distance_fn : callable
        Distance function used by the Voronoi solver.
    scale_weights : bool
        Whether to scale weights using nearest-neighbor spacing.
    points : list | numpy.ndarray
        Site coordinates used for optional scaling.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Weight array and the applied scaling factor.

    Notes
    -----
    Additive and multiplicative distance functions use different default weight
    initializations, so the returned array depends on the selected metric.
    """
    weights = df['weights'].values.astype(float)
    factor = 0
    
    logger.debug(f"Initializing weights for {len(df)} sites (distance_fn={distance_fn.__name__}, scale_weights={scale_weights})")
    
    if distance_fn == default_distance_additive:
        if scale_weights:
            logger.debug("Additive distance: Computing weight scaling factor")
            #factor = auto_weight_scale(points)
            #if factor is None or pd.isna(factor) or factor <= 0:
            #    factor = 1.0
            factor = 1.0
            weights = weights * factor
            logger.debug(f"Additive distance: Applied scaling factor {factor:.4f}")
        else:
            logger.debug("Additive distance: Using zero weights (standard Euclidean)")
            weights = np.zeros(len(df))
    elif distance_fn == default_distance_multiplicative:
        if not scale_weights:
            logger.debug("Multiplicative distance: Using equal weights (standard Voronoi)")
            weights = np.ones(len(df))
        else:
            logger.debug("Multiplicative distance: Using provided weights)")
    logger.debug(f"Weight initialization complete: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
    return weights, factor


def create_weights(sub_df, sigma=3, percent_threshold=10, method='linear'):
    """Calculate normalized site weights from a detection-adjusted area proxy.

    Parameters
    ----------
    sub_df : pandas.DataFrame | geopandas.GeoDataFrame
        Input records with ``total_area`` plus detection count columns
        ``num_detection_rect`` and ``num_detection_circle``. Must already
        include a ``base_values`` column (produced by a custom function  'area_fn').
    sigma : float, default=3
        Standard-deviation multiplier used for upper clipping.
    percent_threshold : float, default=10
        Divisor used to derive the lower clipping threshold from the median.
    method : {'linear', 'logarithmic', 'square_root', 'sigmoid', 'size'}, default='linear'
        Transformation applied before normalization. ``size`` returns equal
        placeholder weights: ``weighted_voronoi`` sets sqrt(people) after
        clustering (``size_weights``).

    Returns
    -------
    pandas.DataFrame
        Copy of ``sub_df`` with added ``capacity_proxy`` and normalized
        ``weights`` columns.
    """
    df = sub_df.copy()
    if method == 'size':
        df['weights'] = 1.0 / len(df)
        return df

    # 1. Base signal must be precomputed in area_fn.
    if 'base_values' not in df.columns:
        raise KeyError("Missing 'base_values'. Run area_fn before create_weights.")
    base_values = pd.to_numeric(df['base_values'], errors='coerce')
    
    # If everything is still NaN (empty or all zeros), fallback to equal distribution
    if base_values.isnull().all() or base_values.sum() == 0:
        df['weights'] = 1.0 / len(df)
        return df

    # 2. Apply the chosen transformation method
    if method == 'logarithmic':
        df['weights'] = np.log1p(base_values)
        
    elif method == 'square_root':
        df['weights'] = np.sqrt(base_values)
        
    elif method == 'sigmoid':
        # Normalize to Z-scores so the sigmoid center (0) aligns with the data mean
        base_std = base_values.std()
        if pd.isna(base_std) or base_std == 0:
            df['weights'] = pd.Series(1.0, index=df.index, dtype=float)
        else:
            z = (base_values - base_values.mean()) / base_std
            df['weights'] = 1 / (1 + np.exp(-z))
    else: # Default to 'linear'
        df['weights'] = base_values
    raw_total = df['weights'].sum()
    if pd.isna(raw_total) or raw_total == 0:
        df['weights'] = 1.0 / len(df)
        return df
    # 3. Initial Normalization to sum=1
    df['weights'] = df['weights'] / df['weights'].sum()
    # 4. Outlier Clipping
    sub_std = df['weights'].std()
    sub_median = df['weights'].median()
    if not pd.isna(sub_std) and sub_std > 0:
        # Upper bound clipping
        upper_limit = sub_median + (sub_std * sigma)
        # Lower bound clipping
        lower_limit = sub_median / percent_threshold
        
        df['weights'] = df['weights'].clip(lower=lower_limit, upper=upper_limit)
    # 5. Final Re-normalization 
    # Necessary because clipping changes the total sum
    total_w = df['weights'].sum()
    if total_w > 0:
        df['weights'] = df['weights'] / total_w
    return df
