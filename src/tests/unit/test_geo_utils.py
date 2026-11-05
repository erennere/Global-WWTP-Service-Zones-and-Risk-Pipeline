from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from src import geo_utils
import argparse
import contextlib
import runpy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box
from src import create_voronoi as cv

pytestmark = pytest.mark.unit


def test_union_find_merges_transitive_groups():
    uf = geo_utils.UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)

    assert uf.find(0) == uf.find(2)
    assert uf.find(3) == uf.find(4)
    assert uf.find(0) != uf.find(3)


def test_cluster_point_indices_merges_transitive_neighbors():
    clusters = geo_utils.cluster_point_indices(
        [Point(0, 0), Point(1, 0), Point(2, 0), Point(10, 0)],
        threshold=1.1,
    )

    assert {frozenset(c) for c in clusters} == {frozenset({0, 1, 2}), frozenset({3})}


def test_estimate_utm_epsg_northern_and_southern_hemisphere():
    assert geo_utils.estimate_utm_epsg(10.0, 50.0) == 32632
    assert geo_utils.estimate_utm_epsg(10.0, -50.0) == 32732


def test_estimate_utm_epsg_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError):
        geo_utils.estimate_utm_epsg(200, 10)


def test_estimate_utm_epsg_for_geom_dispatches_point_vs_centroid():
    point = Point(10.0, 50.0)
    polygon = Polygon([(10, 50), (12, 50), (12, 52), (10, 52)])

    assert geo_utils.estimate_utm_epsg_for_geom(point) == geo_utils.estimate_utm_epsg(10.0, 50.0)
    assert geo_utils.estimate_utm_epsg_for_geom(polygon) == geo_utils.estimate_utm_epsg(
        polygon.centroid.x, polygon.centroid.y
    )


def test_estimate_utm_crs_falls_back_to_web_mercator_for_empty_input():
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    assert geo_utils.estimate_utm_crs(empty).to_epsg() == 3857


def test_estimate_utm_crs_returns_utm_zone_for_valid_geometry():
    gdf = gpd.GeoDataFrame({"geometry": [Point(10.0, 50.0)]}, geometry="geometry", crs="EPSG:4326")

    assert geo_utils.estimate_utm_crs(gdf).to_epsg() == 32632


def test_batch_estimate_utm_epsg_matches_scalar_and_flags_invalid():
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(10.0, 50.0), Point(10.0, -50.0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    epsg_codes, lats = geo_utils.batch_estimate_utm_epsg(gdf)

    assert epsg_codes.tolist() == [32632, 32732]
    assert lats.tolist() == pytest.approx([50.0, -50.0])


def test_parse_diameters_to_round_area_sums_multiple_circles():
    import math

    result = geo_utils.parse_diameters_to_round_area("[2 4]")

    assert result == pytest.approx((1.0**2 + 2.0**2) * math.pi)


def test_parse_diameters_to_round_area_handles_no_numbers():
    assert geo_utils.parse_diameters_to_round_area("[]") == 0


def test_nearest_within_threshold_returns_match_within_distance():
    target = gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(10, 10)]}, geometry="geometry")

    idx = geo_utils.nearest_within_threshold(target.sindex, Point(0.1, 0.1), threshold=5)

    assert idx == 0


def test_nearest_within_threshold_returns_none_when_out_of_range():
    target = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, geometry="geometry")

    idx = geo_utils.nearest_within_threshold(target.sindex, Point(100, 100), threshold=1)

    assert idx is None


def test_nearest_within_threshold_returns_none_for_empty_geometry():
    target = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, geometry="geometry")

    assert geo_utils.nearest_within_threshold(target.sindex, None, threshold=1) is None


def test_nearest_within_threshold_swallows_index_errors():
    class _BrokenIndex:
        def nearest(self, geom, max_distance=None):
            raise RuntimeError("index unavailable")

    assert geo_utils.nearest_within_threshold(_BrokenIndex(), Point(0, 0), threshold=1) is None


def test_ensure_duckdb_spatial_uses_module_level_connection_by_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(geo_utils.duckdb, "sql", lambda query: captured.setdefault("query", query))

    geo_utils.ensure_duckdb_spatial()

    assert captured["query"] == "INSTALL SPATIAL; LOAD SPATIAL;"


def test_ensure_duckdb_spatial_uses_provided_connection():
    calls = []

    class _Conn:
        def execute(self, query):
            calls.append(query)

    geo_utils.ensure_duckdb_spatial(_Conn())

    assert calls == ["INSTALL SPATIAL; LOAD SPATIAL;"]


def test_buffer_geometry_leaves_points_and_lines_unchanged():
    point = Point(0, 0)
    line = LineString([(0, 0), (1, 1)])

    assert geo_utils.buffer_geometry(point) is point
    assert geo_utils.buffer_geometry(line) is line


def test_buffer_geometry_repairs_polygon_topology():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])

    result = geo_utils.buffer_geometry(bowtie)

    assert result.is_valid


def test_safe_to_wkt_converts_recognized_geometry():
    point = Point(1, 2)

    assert geo_utils.safe_to_wkt(point) == point.wkt


def test_safe_to_wkt_returns_none_for_unrecognized_type():
    assert geo_utils.safe_to_wkt("not-a-geometry") is None
    assert geo_utils.safe_to_wkt(None) is None


# --- moved from the coverage-campaign files ---

def test_estimate_utm_crs_linestring_and_epsg_failure(monkeypatch):
    gdf = gpd.GeoDataFrame({"geometry": [LineString([(0, 0), (1, 1)])]}, geometry="geometry", crs="EPSG:4326")

    class _FakeUnion:
        @property
        def centroid(self):
            return Point(float("nan"), float("nan"))

    original_from_epsg = geo_utils.CRS.from_epsg

    def _patched_from_epsg(epsg):
        if epsg != 3857:
            raise RuntimeError("bad epsg")
        return original_from_epsg(epsg)

    monkeypatch.setattr(gpd.GeoSeries, "unary_union", property(lambda self: _FakeUnion()))
    monkeypatch.setattr(geo_utils.CRS, "from_epsg", _patched_from_epsg)
    assert geo_utils.estimate_utm_crs(gdf).to_epsg() == 3857
