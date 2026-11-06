"""Tests for annotation_extract_osm (moved from the coverage-campaign files)."""
import json
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from wwtp.annotation_scripts import NEW_01_GENERATEGRIDS as gen_grids
from wwtp.annotation_scripts import NEW_02_EXTRACTOSMDATAFULL_GEOJSON as osm_extract
from wwtp.figures_scripts import piechart_figure, piechart_interactive
from wwtp.figures_scripts import pop_at_risk_figures
import argparse
import os
import runpy
import sys
import types
import duckdb
import numpy as np
from shapely.geometry import LineString, Point, box
from wwtp import download_pop as dp
from wwtp.annotation_scripts import NEW_01_GENERATEGRIDS as n1
from wwtp.annotation_scripts import NEW_02_EXTRACTOSMDATAFULL_GEOJSON as n2
from wwtp.annotation_scripts import NEW_03_WASTEWATERJOIN_GEOJSON as ww3
from wwtp.annotation_scripts import download_bing_annotate as dba
from wwtp.figures_scripts import piechart_figure as pf
from wwtp import starter
from wwtp.pop_at_risk_river_calculations import create_rasters as cr
from wwtp.pop_at_risk_river_calculations import find_intersection_river as fir
from wwtp.pop_at_risk_river_calculations import impact_polygons_pop as ipp
from wwtp.industrial_analysis import download_and_vectorize as dv

pytestmark = pytest.mark.unit


def test_clean_columns_normalizes_names_and_values():
    gdf = gpd.GeoDataFrame(
        {
            "bad col!": [1],
            "another-col": [None],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    out = osm_extract.clean_columns(gdf)

    assert "bad_col_" in out.columns
    assert "another_col" in out.columns
    assert out["bad_col_"].iloc[0] == "1"


def test_query_overpass_handles_success_rate_limit_and_bad_json(monkeypatch):
    class _Response:
        def __init__(self, status, payload=None, bad_json=False):
            self.status_code = status
            self._payload = payload or {}
            self._bad_json = bad_json

        def json(self):
            if self._bad_json:
                raise json.JSONDecodeError("bad", "x", 0)
            return self._payload

    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(429)
        if calls["n"] == 2:
            return _Response(200, payload={"elements": []})
        return _Response(200, bad_json=True)

    class _Session:
        def get(self, url, params=None, timeout=None):
            return fake_get(url, params=params, timeout=timeout)

    monkeypatch.setattr(osm_extract, "requests_session_with_retries", lambda *a, **k: _Session())
    monkeypatch.setattr(osm_extract.time, "sleep", lambda _: None)

    ok = osm_extract.query_overpass("0,0,1,1", {"waterway": "river"}, "http://dummy")
    bad = osm_extract.query_overpass("0,0,1,1", {"waterway": "river"}, "http://dummy")

    assert ok == {"elements": []}
    assert bad is None


def test_elements_to_gdf_splits_line_and_polygon_content():
    data = {
        "elements": [
            {"type": "node", "id": 1, "lon": 0.0, "lat": 0.0},
            {"type": "node", "id": 2, "lon": 1.0, "lat": 0.0},
            {"type": "node", "id": 3, "lon": 1.0, "lat": 1.0},
            {"type": "node", "id": 4, "lon": 0.0, "lat": 1.0},
            {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"waterway": "river"}},
            {"type": "way", "id": 11, "nodes": [1, 2, 3, 4, 1], "tags": {"landuse": "industrial"}},
        ]
    }

    lines, polys = osm_extract.elements_to_gdf(data)

    assert len(lines) == 1
    assert len(polys) == 1
    assert lines.geometry.iloc[0].geom_type == "LineString"
    assert polys.geometry.iloc[0].geom_type == "Polygon"


def test_find_bbox_handles_invalid_and_empty_geometries():
    assert osm_extract.find_bbox(None) is None

    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    bbox = osm_extract.find_bbox(invalid)
    assert isinstance(bbox, str)


def test_create_tasks_batches_and_rotates_urls(monkeypatch):
    monkeypatch.setattr(osm_extract, "urls", ["u1", "u2", "u3"])
    gdf = gpd.GeoDataFrame({"idx": [1, 2, 3, 4], "bbox": ["a", "b", "c", "d"]}, geometry=[Point(0, 0)] * 4)

    batches = list(osm_extract.create_tasks(gdf, batch_size=2))

    assert len(batches) == 2
    assert batches[0][0] == ("a", 1, "u1")
    assert batches[0][1] == ("b", 2, "u2")


def test_row_operation_skips_none_bbox_and_writes_outputs(monkeypatch, tmp_path):
    captured = {"writes": 0}

    monkeypatch.setattr(osm_extract, "query_overpass", lambda *args, **kwargs: {"elements": []})
    monkeypatch.setattr(
        osm_extract,
        "elements_to_gdf",
        lambda data: (
            gpd.GeoDataFrame({"x": [1], "geometry": [LineString([(0, 0), (1, 1)])]}, geometry="geometry", crs="EPSG:4326"),
            gpd.GeoDataFrame({"y": [1], "geometry": [Polygon([(0, 0), (1, 0), (0, 1), (0, 0)])]}, geometry="geometry", crs="EPSG:4326"),
        ),
    )
    monkeypatch.setattr(osm_extract, "clean_columns", lambda gdf: gdf)
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: None)

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, *args, **kwargs):
        captured["writes"] += 1

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        osm_extract.row_operation(None, 7, "url", str(tmp_path))
        osm_extract.row_operation("0,0,1,1", 8, "url", str(tmp_path))
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["writes"] == 2


def test_new02_elements_timer_and_find_bbox_branches(monkeypatch):
    # elements_to_gdf: empty/no-elements branch
    l0, p0 = n2.elements_to_gdf(None)
    assert l0.empty and p0.empty

    # elements_to_gdf: relation-with-outer polygon branch
    data = {
        "elements": [
            {"type": "node", "id": 1, "lon": 0.0, "lat": 0.0},
            {"type": "node", "id": 2, "lon": 1.0, "lat": 0.0},
            {"type": "node", "id": 3, "lon": 1.0, "lat": 1.0},
            {"type": "node", "id": 4, "lon": 0.0, "lat": 1.0},
            {"type": "way", "id": 10, "nodes": [1, 2, 3, 4, 1], "tags": {}},
            {
                "type": "relation",
                "id": 20,
                "members": [{"type": "way", "role": "outer", "ref": 10}],
                "tags": {"landuse": "industrial"},
            },
        ]
    }
    l1, p1 = n2.elements_to_gdf(data)
    assert l1.empty
    assert len(p1) >= 1

    # find_bbox: invalid that stays invalid after buffer => None branch
    class _BadGeom:
        is_empty = False
        is_valid = False

        def buffer(self, _):
            return self

    assert n2.find_bbox(_BadGeom()) is None

    # timer decorator branch
    wrapped = n2.timer("demo")(lambda x: x + 1)
    assert wrapped(2) == 3


def test_new02_main_smoke_paths(monkeypatch, tmp_path):
    cfg = {
        "overwrite_existing": False,
        "annotations": {
            "retries": 1,
            "max_workers": 1,
            "overpass_urls": [
                "https://overpass.kumi.systems/api/interpreter",
            ],
            "overpass_pause_seconds": 0.1,
        },
        "paths": {
            "corrected_all_filepath": str(tmp_path / "all.gpkg"),
            "annotations_grid_dir": str(tmp_path / "grid"),
            "annotations_by_osm_dir": str(tmp_path / "osm"),
        },
    }

    grid = gpd.GeoDataFrame(
        {"idx": [1, 2], "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(n2.os, "chdir", lambda path: None)
    monkeypatch.setattr(n2, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(n2, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(n2.gpd, "read_file", lambda path: grid.copy())
    monkeypatch.setattr(n2.os, "makedirs", lambda *args, **kwargs: None)

    class _F:
        def __init__(self, value=None):
            self._value = value

        def result(self):
            return self._value

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)
            return _F(None)

    monkeypatch.setattr(n2, "ThreadPoolExecutor", _Exec)
    monkeypatch.setattr(n2, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(n2, "row_operation", lambda *args, **kwargs: None)

    # First pass: one idx already processed, one remains for execution.
    monkeypatch.setattr(n2.os, "listdir", lambda p: ["idx_1_lines.geojson"])
    n2.main()

    # Second pass: all done -> early return branch.
    monkeypatch.setattr(n2.os, "listdir", lambda p: ["idx_1_lines.geojson", "idx_2_polygons.geojson"])
    n2.main()
