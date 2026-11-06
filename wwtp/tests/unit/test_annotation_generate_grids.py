"""Tests for annotation_generate_grids (moved from the coverage-campaign files)."""
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


def test_point_to_square_handles_none_and_builds_bounds():
    assert gen_grids.point_to_square(None, 5) is None
    square = gen_grids.point_to_square(Point(10, 20), 2)
    assert square.bounds == pytest.approx((8, 18, 12, 22))


def test_generategrids_main_filters_points_and_writes(monkeypatch, tmp_path):
    src = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0), Point(1, 1), LineString([(0, 0), (1, 1)])],
            "id": [1, 2, 3],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    captured = {}
    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, driver=None, **kwargs):
        captured["rows"] = len(self)
        captured["path"] = path
        captured["driver"] = driver
        captured["geom_type"] = self.geometry.iloc[0].geom_type

    monkeypatch.setattr(gen_grids.gpd, "read_file", lambda path: src.copy())
    monkeypatch.setattr(gen_grids, "ensure_output_dir_for_file", lambda path: captured.setdefault("ensured", path))
    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        gen_grids.main(100.0, 50.0, "in.gpkg", str(tmp_path / "out.gpkg"))
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["rows"] == 2
    assert captured["driver"] == "GPKG"
    assert captured["geom_type"] == "Polygon"


def test_new01_point_to_square_none_and_valid():
    assert n1.point_to_square(None, 5) is None

    poly = n1.point_to_square(Point(1, 2), 3)
    assert poly is not None
    assert poly.bounds == (-2.0, -1.0, 4.0, 5.0)


def test_new01_main_script_smoke(monkeypatch, tmp_path):
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "annotations": {"cell_size": 10, "factor": 2},
        "paths": {
            "corrected_all_filepath": str(tmp_path / "all_points.gpkg"),
            "annotations_grid_dir": str(grid_dir),
        },
    }

    monkeypatch.setattr(starter, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(starter, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(n1.os, "chdir", lambda path: None)

    src = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    monkeypatch.setattr(gpd, "read_file", lambda path: src.copy())

    captured = {}

    def _to_file(self, path, driver=None, **kwargs):
        captured["path"] = str(path)
        captured["driver"] = driver
        captured["rows"] = len(self)

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", _to_file)

    runpy.run_module("wwtp.annotation_scripts.NEW_01_GENERATEGRIDS", run_name="__main__")

    assert captured["driver"] == "GPKG"
    assert captured["rows"] == 2
    assert captured["path"].endswith("grids_all_points.gpkg")
