"""Shared geometry/data helpers used across multiple pipeline modules.

Functions here were previously duplicated (with subtly different behavior)
across create_voronoi.py, data_merge/*.py, and pop_at_risk_river_calculations/*.py.
They are centralized here so a fix or behavior change only needs to happen once.
"""

import logging
import re
from collections import defaultdict

import geopandas as gpd
import numpy as np
import duckdb
from pyproj import CRS
from scipy.spatial import cKDTree  # type: ignore[attr-defined]
from shapely import Point, LineString, MultiLineString, Polygon, MultiPolygon, to_wkt, make_valid

logger = logging.getLogger(__name__)


class UnionFind:
    """Efficient union-find data structure with path compression."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union two sets by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank - attach smaller tree under larger
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def cluster_point_indices(geoms, threshold):
    """
    Group point geometries into spatial clusters using K-D tree and Union-Find.

    Finds all points within 'threshold' distance of each other using
    K-D tree spatial indexing and Union-Find for efficient clustering.

    Args:
        geoms (array-like): Iterable of shapely Point geometries
        threshold (float): Distance threshold for clustering (in geometry units)

    Returns:
        list[set]: List of point index clusters, each cluster is a set of indices

    Notes:
        Uses Union-Find with path compression for O(n log n) complexity
        vs O(n^2) worst case with BFS. More efficient for large point sets.

        Expected input is an iterable of Point geometries. Callers are
        responsible for pre-filtering non-point geometries before clustering,
        and for converting other representations (e.g. WKT strings) beforehand.
    """
    coords = np.array([(pt.x, pt.y) for pt in geoms])
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, threshold)

    uf = UnionFind(len(coords))
    for i, neighbor_list in enumerate(neighbors):
        for j in neighbor_list:
            if i != j:
                uf.union(i, j)

    clusters_dict = defaultdict(set)
    for i in range(len(coords)):
        root = uf.find(i)
        clusters_dict[root].add(i)

    clusters = list(clusters_dict.values())
    logger.debug(f"Clustered {len(coords)} points into {len(clusters)} clusters (threshold={threshold}m)")
    return clusters


def estimate_utm_epsg(lon, lat):
    """
    Estimate UTM EPSG code from WGS84 longitude and latitude.

    Determines the appropriate UTM zone and hemisphere from coordinates,
    then returns the corresponding EPSG code for that zone.

    Args:
        lon (float): Longitude in degrees (-180 to 180)
        lat (float): Latitude in degrees (-90 to 90)

    Returns:
        int: EPSG code for the UTM zone (e.g., 32633 for UTM zone 33N)

    Notes:
        UTM zones are 6 degrees wide
        Hemisphere determined by latitude sign (S = southern, N = northern)
        EPSG codes: 32601-32660 for northern, 32701-32760 for southern
    """
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        logger.error(f"Invalid coordinates: lon={lon}, lat={lat}")
        raise ValueError("Invalid longitude or latitude")

    zone = int((lon + 180) // 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    epsg = 32600 + zone if hemisphere == 'north' else 32700 + zone
    logger.debug(f"Estimated UTM EPSG {epsg} for zone {zone} ({hemisphere}) from lon={lon:.2f}, lat={lat:.2f}")
    try:
        CRS.from_epsg(epsg)
    except Exception as err:
        logger.warning(f"Invalid EPSG {epsg} when estimating UTM: {err}. Falling back to Web Mercator (3857)")
        return 3857
    return epsg


def estimate_utm_epsg_for_geom(geom):
    """Estimate a UTM EPSG code for a geometry.

    Uses the geometry's own coordinates when it's a Point, otherwise its
    centroid - the "point-or-centroid" dispatch used whenever a mix of
    Point and non-Point geometries needs a per-row UTM zone.
    """
    if isinstance(geom, Point):
        return estimate_utm_epsg(geom.x, geom.y)
    return estimate_utm_epsg(geom.centroid.x, geom.centroid.y)


def estimate_utm_crs(gdf):
    """
    Estimate appropriate UTM CRS from GeoDataFrame geometries.

    Extracts a valid centroid from geometries and estimates UTM zone.
    Validates the resulting CRS and falls back to Web Mercator if needed.

    Args:
        gdf (pd.GeoDataFrame): GeoDataFrame with geometry column

    Returns:
        pyproj.CRS: UTM CRS for the region, or Web Mercator EPSG:3857

    Logs:
        WARNING: When valid geometries are insufficient or coordinates invalid

    Notes:
        Validates both lon and lat are finite using 'and' logic.
    """
    valid_geoms = gdf.geometry[
        gdf.geometry.is_valid &
        ~gdf.geometry.is_empty &
        gdf.geometry.notna()
    ]

    if valid_geoms.empty:
        logger.info("No valid geometries available to estimate UTM CRS. Falling back to Web Mercator (3857).")
        return CRS.from_epsg(3857)

    centroid = valid_geoms.unary_union.centroid
    try:
        lon, lat = centroid.x, centroid.y
        logger.debug(f"Extracted centroid from {len(valid_geoms)} valid geometries: lon={lon:.4f}, lat={lat:.4f}")
    except Exception:
        lon, lat = None, None
        logger.debug("Unable to read centroid coordinates from valid geometries; searching for a fallback geometry.")

    if lon is None or lat is None or not (np.isfinite(lon) and np.isfinite(lat)):
        logger.debug("Initial centroid has non-finite coordinates, searching for valid point geometry...")
        check = True
        for geom in valid_geoms:
            if isinstance(geom, Point):
                lon, lat = geom.x, geom.y
                check = False
                logger.debug(f"Found valid Point geometry: lon={lon:.4f}, lat={lat:.4f}")
                break
            elif isinstance(geom, (Polygon, LineString, MultiPolygon, MultiLineString)):
                try:
                    fallback_centroid = geom.centroid
                    lon, lat = fallback_centroid.x, fallback_centroid.y
                except Exception:
                    continue
                check = False
                logger.debug(f"Found valid {geom.geom_type} with centroid: lon={lon:.4f}, lat={lat:.4f}")
                break
        if check:
            logger.info("Centroid has non-finite coordinates (inf or NaN). Falling back to Web Mercator (3857).")
            return CRS.from_epsg(3857)

    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    logger.debug(f"Estimated UTM EPSG {epsg} for zone {zone} from centroid")
    try:
        epgs = CRS.from_epsg(epsg)
        return epgs
    except Exception as err:
        logger.info(f'Failed to create UTM CRS EPSG:{epsg}: {err}. Falling back to Web Mercator (3857)')
        return CRS.from_epsg(3857)


def buffer_geometry(geom):
    """Apply a zero-buffer topology fix to polygon geometries.

    Point and line geometries are returned unchanged.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Geometry to process.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Original geometry with topology artefacts repaired for polygonal
        types; other geometry types are returned as-is.
    """
    if isinstance(geom, Point):
        return geom
    elif isinstance(geom, (LineString, MultiLineString)):
        return geom
    elif isinstance(geom, (Polygon, MultiPolygon)):
        try:
            return geom.buffer(0)
        except Exception as e:
            logger.debug(f"Error buffering geometry: {e}")
            return geom
    else:
        logger.debug(f"Unknown geometry type in buffer_geometry: {type(geom)}")
        return geom


def safe_to_wkt(geom):
    """Convert a geometry to WKT, or None if it isn't a recognized geometry type."""
    return to_wkt(geom) if isinstance(geom, (Point, LineString, Polygon, MultiLineString, MultiPolygon)) else None


def batch_estimate_utm_epsg(gdf):
    """Estimate UTM EPSG and latitude arrays from geometry centroids.

    Vectorized counterpart of `estimate_utm_epsg` for use over a full
    GeoDataFrame at once. Out-of-range coordinates fall back to EPSG:3857
    instead of raising, since this is used over batches where a single bad
    row shouldn't abort the whole computation.
    """
    centroids = gdf.geometry.centroid
    lons, lats = centroids.x, centroids.y
    zones = (np.floor((lons + 180) / 6) + 1).astype(int)
    epsg_codes = np.where(lats >= 0, 32600 + zones, 32700 + zones)

    invalid_mask = (lats > 84) | (lats < -80) | (lons < -180) | (lons > 180)
    if invalid_mask.any():
        epsg_codes[invalid_mask] = 3857
    return epsg_codes, lats


def parse_diameters_to_round_area(diameters):
    """Parse a '[d1 d2 ...]'-style diameter string into a total circle area.

    Extracts every numeric token from `diameters` and sums pi*(d/2)^2 over
    each, treating each token as the diameter of a separate circular pond.
    """
    diameters_2 = [float(i) for i in re.findall(r"[-+]?\d*\.\d+|\d+", str(diameters))]
    return np.sum([(d / 2) ** 2 * np.pi for d in diameters_2])


def nearest_within_threshold(sindex, geom, threshold=None):
    """Return the positional index of the nearest indexed feature, or None.

    `threshold` is the maximum search distance in the index's CRS units;
    `None` means an unbounded nearest-neighbor search. Returns None if
    `geom` is missing/empty, if no feature is found within `threshold`, or
    if the spatial index lookup itself raises.
    """
    if geom is None or geom.is_empty:
        return None
    try:
        nearest = sindex.nearest(geom, max_distance=threshold)
        if len(nearest[1]) == 0:
            return None
        return nearest[1][0]
    except Exception:
        return None


def ensure_duckdb_spatial(connection=None):
    """Install and load the DuckDB SPATIAL extension (idempotent).

    Both INSTALL and LOAD are no-ops if already satisfied, so this is safe
    to call before every spatial query rather than tracking state manually.
    Pass `connection` when using a dedicated (e.g. file-backed) connection
    instead of the module-level default `duckdb` connection.
    """
    query = "INSTALL SPATIAL; LOAD SPATIAL;"
    if connection is not None:
        connection.execute(query)
    else:
        duckdb.sql(query)


def load_eu_reference_layer(filepath, factor, capacity_column="uwwCapacity",
                            pop_column="POP_SERVED_EU"):
    """Load the UWWTD reference layer, project it, and derive the served-population column.

    All three EU-comparison entry points (``eu_comparison``,
    ``compare_pop_sweep_hw_eu``, ``sweep_ver_ranking``) used to inline this and had
    drifted: two applied ``factor`` and one dropped it, and each picked its own
    projection by hand (a hardcoded 3857 in one, a hardcoded UTM zone in the
    others). The CRS is now estimated from the layer itself, so the metre
    distances every caller thresholds on are actually metres.

    ``factor`` is required rather than defaulted: a silent ``1`` is what let the
    three call sites disagree in the first place.
    """
    ref_gdf = gpd.read_file(filepath)
    if capacity_column not in ref_gdf.columns:
        raise KeyError(
            f"Reference column '{capacity_column}' not found in EU reference layer "
            f"{filepath}; columns present: {sorted(ref_gdf.columns)}"
        )
    ref_gdf = ref_gdf.to_crs(estimate_utm_crs(ref_gdf))
    ref_gdf[pop_column] = float(factor) * ref_gdf[capacity_column]
    return ref_gdf


def nearest_neighbor_distances(coords, neighbors=2):
    """Mean distance from each point to its ``neighbors`` closest other points.

    ``cKDTree.query`` returns the point itself as the first (zero-distance)
    result, so ``k`` is ``neighbors + 1`` and column 0 is dropped. When fewer
    points are available than requested, every present neighbour is used.

    Returns an all-NaN array for fewer than two points - a single point has no
    neighbour to measure against.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return np.full(n, np.nan, dtype=float)

    k = min(int(neighbors) + 1, n)
    distances, _ = cKDTree(coords).query(coords, k=k)
    distances = np.atleast_2d(distances)
    return np.nanmean(distances[:, 1:k], axis=1).astype(float)


def repair_geometry(geom):
    """Return a valid, non-empty geometry, or ``None`` if it cannot be repaired.

    Consolidates four drifted copies. The strictness of the most complete one
    (``download_and_vectorize``) is kept: ``make_valid`` can itself return an
    invalid result, so the outcome is re-checked and passed through ``buffer(0)``
    once more, and anything still invalid after that is rejected rather than
    returned. The interactive-map copies skipped both steps and could hand a
    still-invalid geometry to the tiler.

    The ``except Exception`` around ``make_valid`` comes from the map variant -
    GEOS raises on some degenerate inputs and ``buffer(0)`` is the fallback.
    """
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom

    try:
        repaired = make_valid(geom)
    except Exception:
        repaired = geom.buffer(0)

    if repaired is None or repaired.is_empty:
        return None
    if not repaired.is_valid:
        repaired = repaired.buffer(0)
    if repaired is None or repaired.is_empty or not repaired.is_valid:
        return None
    return repaired


# --------------------------------------------------------------------------- #
# Metric distances and buffers on geographic data                             #
# --------------------------------------------------------------------------- #
# EPSG:3857 and EPSG:4326 are not metric: a "500 m" threshold in Web Mercator is
# 321 m on the ground at 50 deg N and 250 m at 60 deg N, and the same number in
# EPSG:4326 means 500 degrees. These helpers measure on the sphere (distances)
# or in each feature's own UTM zone (buffers), so thresholds in metres mean
# metres everywhere.

EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius (IUGG)


def _lonlat_of(gdf):
    """Return an (n, 2) lon/lat array (NaN for missing/empty geometries)."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame needs a CRS to measure distances in metres")
    geoms = gdf.geometry.to_crs(4326)
    out = np.full((len(geoms), 2), np.nan, dtype=float)
    valid = geoms.notna().to_numpy() & ~geoms.is_empty.to_numpy()
    if valid.any():
        sub = geoms[valid]
        is_point = (sub.geom_type == "Point").to_numpy()
        pts = sub.where(is_point, sub.representative_point())
        out[valid, 0] = pts.x.to_numpy()
        out[valid, 1] = pts.y.to_numpy()
    return out


def _unit_vectors(lonlat):
    lon = np.radians(lonlat[:, 0])
    lat = np.radians(lonlat[:, 1])
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _chord_to_metres(chord):
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.clip(np.asarray(chord, dtype=float) / 2.0, 0.0, 1.0))


def nearest_distances_m(source_gdf, target_gdf, k=1):
    """Great-circle distance in metres from each target feature to its ``k`` nearest sources.

    Points are used as they are, other geometries by their representative
    point. Returns ``(distances, indices)``: arrays of shape ``(n_target,)``
    for ``k == 1`` and ``(n_target, k)`` otherwise; positional indices into
    ``source_gdf``. Missing results are ``inf`` / ``-1``.
    """
    target_ll = _lonlat_of(target_gdf)
    source_ll = _lonlat_of(source_gdf)
    shape = (len(target_ll),) if k == 1 else (len(target_ll), k)
    distances = np.full(shape, np.inf, dtype=float)
    indices = np.full(shape, -1, dtype=int)

    src_ok = ~np.isnan(source_ll).any(axis=1)
    tgt_ok = ~np.isnan(target_ll).any(axis=1)
    if not src_ok.any() or not tgt_ok.any():
        return distances, indices

    src_pos = np.flatnonzero(src_ok)
    kk = min(int(k), len(src_pos))
    chord, idx = cKDTree(_unit_vectors(source_ll[src_ok])).query(_unit_vectors(target_ll[tgt_ok]), k=kk)
    chord = np.asarray(chord, dtype=float).reshape(-1, kk)
    idx = np.asarray(idx).reshape(-1, kk)
    found = np.isfinite(chord)
    metres = np.where(found, _chord_to_metres(np.where(found, chord, 0.0)), np.inf)
    mapped = np.where(found, src_pos[np.clip(idx, 0, len(src_pos) - 1)], -1)

    if k == 1:
        distances[tgt_ok] = metres[:, 0]
        indices[tgt_ok] = mapped[:, 0]
    else:
        distances[tgt_ok, :kk] = metres
        indices[tgt_ok, :kk] = mapped
    return distances, indices


def buffer_meters(gdf, distance_m):
    """Buffer every geometry by ``distance_m`` metres in its own UTM zone.

    Returns a GeoSeries in the input CRS. Features are grouped by the UTM zone
    of their centroid, buffered there and projected back, so the radius is
    within 0.1 % of ``distance_m`` wherever the feature lies (Web Mercator, by
    contrast, shrinks a buffer by cos(latitude)).
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame needs a CRS to buffer in metres")
    geoms = gdf.geometry
    out = geoms.copy()
    valid = geoms.notna().to_numpy() & ~geoms.is_empty.to_numpy()
    if not valid.any():
        return out
    lonlat = _lonlat_of(gdf)
    zones = (np.floor((lonlat[:, 0] + 180.0) / 6.0) + 1).clip(1, 60)
    epsg = np.where(lonlat[:, 1] >= 0, 32600, 32700) + np.nan_to_num(zones, nan=1).astype(int)
    # Polar caps fall outside UTM: use the polar stereographic projections there.
    epsg = np.where(lonlat[:, 1] > 84, 3413, epsg)
    epsg = np.where(lonlat[:, 1] < -80, 3031, epsg)
    for code in np.unique(epsg[valid]):
        mask = valid & (epsg == code)
        part = geoms[mask].to_crs(int(code)).buffer(float(distance_m)).to_crs(gdf.crs)
        out[mask] = part.to_numpy()
    return out

