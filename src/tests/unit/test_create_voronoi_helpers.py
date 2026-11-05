import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from src import create_voronoi


pytestmark = pytest.mark.unit


def test_calculate_area_combines_round_and_rect_components(sample_sites_gdf):
    result = create_voronoi.calculate_area(sample_sites_gdf.copy())

    expected_round_area = np.pi * (1.0**2 + 2.0**2)
    expected_total_area = 15.0 + expected_round_area

    assert result.loc[0, "round_area"] == pytest.approx(expected_round_area)
    assert result.loc[0, "total_area"] == pytest.approx(expected_total_area)
    assert result.loc[0, "num_detections"] == 4
    assert result.loc[0, "capacity_proxy"] == pytest.approx(expected_total_area * 2.0)

    fallback_mean = result["capacity_proxy"].mean()
    assert result.loc[1, "base_values"] == pytest.approx(fallback_mean)
    assert result.loc[2, "base_values"] == pytest.approx(fallback_mean)


def test_calculate_area_defaults_without_rectangles():
    df = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = create_voronoi.calculate_area(df.copy())

    assert result["total_area"].tolist() == [1, 1]
    assert result["num_detections"].tolist() == [0, 0]
    assert result["capacity_proxy"].tolist() == [0.0, 0.0]
    assert result["base_values"].tolist() == [0.0, 0.0]


def test_default_distance_functions_match_normalized_plane_math():
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    b = (0.0, 0.0)

    additive = create_voronoi.default_distance_additive(a, b, weight=0.5, factor=1.0)
    multiplicative = create_voronoi.default_distance_multiplicative(a, b, weight=0.5, factor=1.0)

    # Isotropic normalisation: both axes divided by the larger extent (4).
    # Additive returns the signed power distance |a-b|^2 - w^2.
    assert additive == pytest.approx(np.array([-0.25, 1.5625 - 0.25]))
    assert multiplicative == pytest.approx(np.array([0.0, 1.25 / 0.5]))


def test_cluster_point_indices_merges_transitive_neighbors():
    clusters = create_voronoi.cluster_point_indices(
        [Point(0, 0), Point(1, 0), Point(2, 0), Point(10, 0)],
        threshold=1.1,
    )

    assert {frozenset(cluster) for cluster in clusters} == {frozenset({0, 1, 2}), frozenset({3})}


def test_calculate_buffer_fixed_branch_clips_requested_buffer():
    df = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = create_voronoi.calculate_buffer(
        df,
        weights=np.array([0.5, 0.5]),
        buffer=9000,
        dynamic_buffering=False,
        min_buffer=2000,
        max_buffer=5000,
        area_ceiling=False,
        k_min=0.32,
        k_max=0.92,
        detection_confidence_threshold=3,
    )

    assert result.tolist() == [5000.0, 5000.0]


def test_calculate_buffer_single_site_dynamic_uses_buffer_fallback(monkeypatch):
    df = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0)],
            "num_detection_circle": [0],
            "num_detection_rect": [0],
            "basin_area": [1000.0],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(
        create_voronoi,
        "nearest_neighbor_distances_and_median",
        lambda input_df: (np.array([np.nan]), np.nan),
    )

    result = create_voronoi.calculate_buffer(
        df,
        weights=np.array([1.0]),
        buffer=10000,
        dynamic_buffering=True,
        min_buffer=2000,
        max_buffer=20000,
        area_ceiling=False,
        k_min=0.32,
        k_max=0.92,
        detection_confidence_threshold=3,
        k_value=0.5,
    )

    expected_k_density = np.log1p(10000) / np.log1p(60000)
    expected_k = 0.32 + expected_k_density * (0.92 - 0.32)
    expected_buffer = 10000 * expected_k

    assert result == pytest.approx(np.array([expected_buffer]))


def test_calculate_buffer_uses_mean_2_nnd_when_nnd_length_mismatches(monkeypatch):
    df = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0), Point(1, 1)],
            "mean_2_nnd": [4000.0, 9000.0],
            "num_detection_circle": [0, 0],
            "num_detection_rect": [0, 0],
            "basin_area": [1000.0, 1000.0],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(
        create_voronoi,
        "nearest_neighbor_distances_and_median",
        lambda input_df: (np.array([1.0]), 1.0),
    )

    result = create_voronoi.calculate_buffer(
        df,
        weights=np.array([0.5, 0.5]),
        buffer=7000,
        dynamic_buffering=True,
        min_buffer=1000,
        max_buffer=20000,
        area_ceiling=False,
        k_min=0.32,
        k_max=0.92,
        detection_confidence_threshold=3,
        k_value=0.5,
    )

    expected = []
    for nnd in [4000.0, 9000.0]:
        k_density = np.log1p(nnd) / np.log1p(60000)
        k = 0.32 + k_density * (0.92 - 0.32)
        expected.append(nnd * k)

    assert result == pytest.approx(np.array(expected))


def test_filter_requested_approaches_skips_city_when_disabled(monkeypatch):
    monkeypatch.setattr(create_voronoi.os.path, "exists", lambda path: False)

    runnable, skipped_existing, skipped_disabled = create_voronoi._filter_requested_approaches(
        ["0", "2"],
        {"city_voronoi": False, "overwrite_existing": False},
        {
            "voronoi": {
                "0": "appr_0.gpkg",
                "1": "appr_1.gpkg",
                "2": "appr_2.gpkg",
                "0_only_round": "appr_0_only_round.gpkg",
                "1_only_round": "appr_1_only_round.gpkg",
            }
        },
        only_round=False,
    )

    assert runnable == ["0"]
    assert skipped_existing == []
    assert skipped_disabled == ["2"]


def test_filter_requested_approaches_keeps_city_when_enabled_and_skips_existing(monkeypatch):
    monkeypatch.setattr(
        create_voronoi.os.path,
        "exists",
        lambda path: path == "appr_1_only_round.gpkg",
    )

    runnable, skipped_existing, skipped_disabled = create_voronoi._filter_requested_approaches(
        ["1", "2"],
        {"city_voronoi": True, "overwrite_existing": False},
        {
            "voronoi": {
                "0": "appr_0.gpkg",
                "1": "appr_1.gpkg",
                "2": "appr_2.gpkg",
                "0_only_round": "appr_0_only_round.gpkg",
                "1_only_round": "appr_1_only_round.gpkg",
            }
        },
        only_round=True,
    )

    assert runnable == ["2"]
    assert skipped_existing == ["1"]
    assert skipped_disabled == []


def test_weighted_voronoi_returns_none_for_empty_input(tiny_country_gdf):
    empty = gpd.GeoDataFrame({"WASTE_ID": [], "HYBAS_ID": [], "weights": []}, geometry=[], crs="EPSG:4326")

    result = create_voronoi.weighted_voronoi(
        empty,
        "HYBAS_ID",
        tiny_country_gdf,
        n_points=20,
        calculate_buffer_fn=lambda df, weights, **kwargs: np.array([]),
        site_id_col="WASTE_ID",
    )

    assert result is None


def test_weighted_voronoi_raises_when_site_id_column_is_missing(tiny_country_gdf):
    df = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101],
            "weights": [1.0],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(KeyError, match="Site identifier column 'WASTE_ID'"):
        create_voronoi.weighted_voronoi(
            df,
            "HYBAS_ID",
            tiny_country_gdf,
            n_points=20,
            calculate_buffer_fn=lambda sub_df, weights, **kwargs: np.array([1000.0]),
            site_id_col="WASTE_ID",
        )


def test_estimate_utm_crs_always_returns_a_crs():
    """The dead `if utm is None` guards were removed because this cannot happen.

    estimate_utm_crs falls back to Web Mercator rather than returning None, so
    callers have no unresolvable-CRS branch to handle.
    """
    from src.geo_utils import estimate_utm_crs

    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    assert estimate_utm_crs(empty).to_epsg() == 3857

    populated = gpd.GeoDataFrame(
        {"geometry": [Point(10, 50)]}, geometry="geometry", crs="EPSG:4326"
    )
    assert estimate_utm_crs(populated).to_epsg() == 32632


def test_weighted_voronoi_rejects_calculate_buffer_length_mismatch(tiny_country_gdf):
    df = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1, 2],
            "HYBAS_ID": [101, 101],
            "weights": [0.5, 0.5],
            "geometry": [Point(0, 0), Point(1, 0)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="buffer lengths"):
        create_voronoi.weighted_voronoi(
            df,
            "HYBAS_ID",
            tiny_country_gdf,
            threshold=0.1,
            n_points=20,
            calculate_buffer_fn=lambda sub_df, weights, **kwargs: np.array([1000.0]),
            site_id_col="WASTE_ID",
        )


def test_weighted_voronoi_single_site_clips_to_country_boundary():
    site = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1],
            "HYBAS_ID": [101],
            "weights": [1.0],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    country_clip = gpd.GeoDataFrame(
        {"country": ["DE"], "geometry": [box(-100, -100, 100, 100)]},
        geometry="geometry",
        crs="EPSG:3857",
    )

    region_df, point_df = create_voronoi.weighted_voronoi(
        site,
        "HYBAS_ID",
        country_clip,
        clipping=None,
        n_points=30,
        buffering=False,
        calculate_buffer_fn=lambda sub_df, weights, **kwargs: np.array([500.0]),
        site_id_col="WASTE_ID",
    )

    assert len(region_df) == 1
    assert len(point_df) == 1
    assert region_df.crs.to_epsg() == 4326
    assert point_df.crs.to_epsg() == 4326

    country_clip_wgs84 = country_clip.to_crs(4326)
    assert region_df.geometry.iloc[0].difference(country_clip_wgs84.geometry.iloc[0]).area == pytest.approx(0.0, abs=1e-9)


def test_geometry_contains_points_returns_expected_mask():
    geom = box(0, 0, 1, 1)
    points = np.array([[0.5, 0.5], [2.0, 2.0]])

    mask = create_voronoi.geometry_contains_points(geom, points)

    assert mask.tolist() == [True, False]


def test_normalize_plane_and_create_ranges_cover_edge_paths():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = (5.0, 6.0)
    a_norm, b_norm = create_voronoi.normalize_plane(a, b)
    assert a_norm.min() >= 0 and a_norm.max() <= 1
    assert b_norm.min() >= 0 and b_norm.max() <= 1

    same = create_voronoi.create_ranges(2.0, 2.0, step=10.0)
    short = create_voronoi.create_ranges(0.0, 1.0, step=10.0, min_step=0.2)
    exact = create_voronoi.create_ranges(0.0, 250.0, step=100.0)
    assert same.tolist() == [2.0, 2.0]
    # The spacing is always exactly `step`; the last node reaches past max.
    assert short.tolist() == [0.0, 10.0]
    assert exact.tolist() == [0.0, 100.0, 200.0, 300.0]


def test_is_valid_geom_and_create_centroid_points_handles_invalid_inputs():
    assert create_voronoi.is_valid_geom(None) is False
    assert create_voronoi.is_valid_geom(LineString([(0, 0), (np.inf, 1)])) is False
    assert create_voronoi.is_valid_geom(Point(0, 0)) is True

    assert create_voronoi.create_centroid_points(pd.NA) is None
    poly_centroid = create_voronoi.create_centroid_points(Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]))
    assert isinstance(poly_centroid, Point)


def test_normalize_column_to_rounded_str_preserves_nan_shape():
    series = pd.Series([1.2, 2.7, np.nan])

    out = create_voronoi.normalize_column_to_rounded_str(series)

    assert out.iloc[0] == "1"
    assert out.iloc[1] == "3"
    assert pd.isna(out.iloc[2])


def test_process_centroid_and_extract_site_coordinates_basic_cases():
    polygons = gpd.GeoDataFrame(
        {"zone": ["A"], "geometry": [box(0, 0, 2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    sidx = polygons.sindex

    match = create_voronoi.process_centroid((Point(1, 1), sidx, polygons, "zone"))
    miss = create_voronoi.process_centroid((Point(5, 5), sidx, polygons, "zone"))

    assert match == "A"
    assert miss is None

    sites = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(0, 0),
                LineString([(0, 0), (2, 0)]),
                MultiPolygon([Polygon([(0, 0), (1, 0), (0, 1), (0, 0)])]),
            ]
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    coords = create_voronoi.extract_site_coordinates(sites, centroid_points=True)
    assert len(coords) == 3
    assert coords[0] == (0.0, 0.0)


def test_dissolve_fast_and_resolve_overlaps_reduce_intersections():
    subdf = gpd.GeoDataFrame(
        {
            "some_id": [1, 2],
            "geometry": [box(0, 0, 1, 1), box(0.5, 0.5, 1.5, 1.5)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    groups, dissolved = create_voronoi.dissolve_overlapping_geometries_fast(subdf.copy(), radius=1000.0, convex=True)
    assert len(groups) >= 1
    assert dissolved is not None

    region = gpd.GeoDataFrame(
        {
            "geometry": [box(0, 0, 2, 2), box(1, 1, 3, 3)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    cleaned = create_voronoi.resolve_polygon_overlaps(region)
    assert cleaned[0].intersection(cleaned[1]).area == pytest.approx(0.0)


def test_extract_contours_finalize_and_assign_streaming_cover_core_logic():
    mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    polys_scipy = create_voronoi.extract_contours_scipy(mask, n_points=1, grid_minx=0, grid_miny=0)
    polys_cv2 = create_voronoi.extract_contours_cv2(mask, n_points=1, grid_minx=0, grid_miny=0)
    polys_rio = create_voronoi.extract_contours_rasterio(mask, n_points=1, grid_minx=0, grid_miny=0)
    assert len(polys_scipy) >= 1
    assert len(polys_cv2) >= 1
    assert len(polys_rio) >= 1

    part1 = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)], "x": [1]}, geometry="geometry", crs="EPSG:4326")
    part2 = gpd.GeoDataFrame({"geometry": [box(1, 1, 2, 2)], "x": [2]}, geometry="geometry", crs="EPSG:4326")
    final = create_voronoi.finalize_gdf([part1, part2], cols=part1.columns)
    assert len(final) == 2

    valid_points = np.array([[0.0, 0.0], [10.0, 0.0]])
    sites = np.array([[0.0, 0.0], [10.0, 0.0]])
    weights = np.array([1.0, 1.0])
    assignments = create_voronoi.assign_sites_streaming(
        valid_points,
        sites,
        weights,
        create_voronoi.default_distance_multiplicative,
        factor=1.0,
    )
    assert assignments.tolist() == [0, 1]