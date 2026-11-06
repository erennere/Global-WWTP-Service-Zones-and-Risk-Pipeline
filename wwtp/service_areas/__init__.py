"""Service areas: the weighted Voronoi split of land between plants (stage ``create_voronoi``).

Modules: size, weights, contours, joins, overlaps, solver, orchestrate. The names below are the stage's public API.
"""

from .size import (
    LAGOON_COLUMN,
    SIZE_MEASURES,
    _truthy,
    calculate_area,
    reported_size_people,
    plant_size_people,
    service_radius_m,
    calculate_buffer,
)
from .weights import (
    normalize_plane,
    auto_weight_scale,
    default_distance_additive,
    default_distance_multiplicative,
    initialize_voronoi_weights,
    create_weights,
)
from .contours import extract_contours_scipy, extract_contours_cv2, extract_contours_rasterio
from .joins import (
    is_valid_geom,
    drop_duplicates,
    create_centroid_points,
    normalize_column_to_rounded_str,
    download_overture_maps,
    process_centroid,
    intersect_with_polygon_sindex,
    intersect_with_polygons_db,
    intersect_with_polygons_parallelized,
    intersects_with_country_db,
)
from .overlaps import dissolve_overlapping_geometries_fast, orchestrate_overlaps, resolve_polygon_overlaps
from .solver import (
    geometry_contains_points,
    create_ranges,
    cluster_points,
    extract_site_coordinates,
    finalize_gdf,
    assign_sites_streaming,
    weighted_voronoi,
    voronoi_worker,
)
from .orchestrate import orchestrate_voronoi_weights

__all__ = [
    "LAGOON_COLUMN",
    "SIZE_MEASURES",
    "_truthy",
    "calculate_area",
    "reported_size_people",
    "plant_size_people",
    "service_radius_m",
    "calculate_buffer",
    "normalize_plane",
    "auto_weight_scale",
    "default_distance_additive",
    "default_distance_multiplicative",
    "initialize_voronoi_weights",
    "create_weights",
    "extract_contours_scipy",
    "extract_contours_cv2",
    "extract_contours_rasterio",
    "is_valid_geom",
    "drop_duplicates",
    "create_centroid_points",
    "normalize_column_to_rounded_str",
    "download_overture_maps",
    "process_centroid",
    "intersect_with_polygon_sindex",
    "intersect_with_polygons_db",
    "intersect_with_polygons_parallelized",
    "intersects_with_country_db",
    "dissolve_overlapping_geometries_fast",
    "orchestrate_overlaps",
    "resolve_polygon_overlaps",
    "geometry_contains_points",
    "create_ranges",
    "cluster_points",
    "extract_site_coordinates",
    "finalize_gdf",
    "assign_sites_streaming",
    "weighted_voronoi",
    "voronoi_worker",
    "orchestrate_voronoi_weights",
]
