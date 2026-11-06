"""Tests for find_intersection_river (moved from the coverage-campaign files)."""
import argparse
import os
import runpy
import sys
import types
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
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


def test_find_intersection_river_branch_helpers_and_empty_paths(monkeypatch):
    graph = {1: 2, 2: 0, 3: 4, 4: 0}
    assert fir.find_intersection_id(1, 3, graph) is None
    assert fir.find_common_intersection([1, 3], graph) is None

    empty_poly = gpd.GeoDataFrame({"river_list": [], "geometry": []}, geometry="geometry", crs="EPSG:4326")
    rivers = gpd.GeoDataFrame({"HYRIV_ID": [], "NEXT_DOWN": [], "geometry": []}, geometry="geometry", crs="EPSG:4326")
    out = fir.assign_river_juncture(empty_poly.copy(), rivers)
    assert "NXT_DIS" in out.columns


def test_find_intersection_orchestration_and_main_smoke(monkeypatch, tmp_path):
    class _Future:
        def __init__(self, value=None, err=None):
            self._value = value
            self._err = err

        def result(self):
            if self._err is not None:
                raise self._err
            return self._value

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            try:
                return _Future(value=fn(*args, **kwargs))
            except Exception as e:
                return _Future(err=e)

    polygons = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [10],
            "NEXT_DOWN": [0],
            "MAIN_RIV": [100],
            "HYBAS_ID": [1],
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(fir, "estimate_utm_epsg_for_geom", lambda geom: 3857)
    monkeypatch.setattr(fir, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(fir, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(fir, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(fir, "optimize_river_lookup", lambda *args, **kwargs: gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326"))

    out = fir.orchestrate_settlement_river_intersections(polygons.copy(), rivers.copy(), x_distance=1000, max_workers=1)
    assert "river_list" in out.columns
    assert out["river_list"].iloc[0] == []

    # Main smoke path
    cfg = {
        "x_distance": 5000,
        "basin_column_name": "HYBAS_ID",
        "paths": {
            "non_served_above_threshold_outpath": str(tmp_path / "poly.gpkg"),
            "rivershed_output_path": str(tmp_path / "rivers.gpkg"),
            "non_served_nxt_river_outpath": str(tmp_path / "out.gpkg"),
        }
    }
    monkeypatch.setattr(os, "chdir", lambda path: None)
    monkeypatch.setattr(fir, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(fir, "load_config", lambda **kwargs: cfg)

    def _read_file(path, columns=None, **kwargs):
        if str(path).endswith("poly.gpkg"):
            return gpd.GeoDataFrame(
                {"HYBAS_ID": [1], "geometry": [box(0, 0, 1, 1)]},
                geometry="geometry",
                crs="EPSG:4326",
            )
        return gpd.GeoDataFrame(
            {
                "HYRIV_ID": [10],
                "NEXT_DOWN": [0],
                "MAIN_RIV": [100],
                "HYBAS_ID": [1],
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

    monkeypatch.setattr(fir.gpd, "read_file", _read_file)
    monkeypatch.setattr(fir, "orchestrate_settlement_river_intersections", lambda p, r, x_distance, max_workers=1: p.assign(river_list=[[10]]))
    monkeypatch.setattr(fir, "assign_main_riv", lambda p, r: p.assign(MAIN_RIV=[100]))
    monkeypatch.setattr(fir, "orchestrate_river_assignment", lambda p, r, max_workers=1: p.assign(NXT_DIS=[10]))
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, path, driver="GPKG", index=False: None)
    monkeypatch.setattr(sys, "argv", ["prog"])

    fir.main()


def test_find_intersection_main_raises_when_missing_crs(monkeypatch, tmp_path):
    cfg = {
        "x_distance": 5000,
        "basin_column_name": "HYBAS_ID",
        "paths": {
            "non_served_above_threshold_outpath": str(tmp_path / "poly.gpkg"),
            "rivershed_output_path": str(tmp_path / "rivers.gpkg"),
            "non_served_nxt_river_outpath": str(tmp_path / "out.gpkg"),
        }
    }
    monkeypatch.setattr(os, "chdir", lambda path: None)
    monkeypatch.setattr(fir, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(fir, "load_config", lambda **kwargs: cfg)

    no_crs_poly = gpd.GeoDataFrame({"HYBAS_ID": [1], "geometry": [box(0, 0, 1, 1)]}, geometry="geometry")
    ok_rivers = gpd.GeoDataFrame(
        {"HYRIV_ID": [1], "NEXT_DOWN": [0], "MAIN_RIV": [1], "HYBAS_ID": [1], "geometry": [LineString([(0, 0), (1, 1)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(fir.gpd, "read_file", lambda path, columns=None, **kwargs: no_crs_poly.copy() if str(path).endswith("poly.gpkg") else ok_rivers.copy())
    monkeypatch.setattr(sys, "argv", ["prog", "--max-workers", "4"])

    with pytest.raises(ValueError):
        fir.main()


def test_find_intersection_main_rejects_non_positive_workers(monkeypatch, tmp_path):
    cfg = {
        "x_distance": 5000,
        "basin_column_name": "HYBAS_ID",
        "paths": {
            "non_served_above_threshold_outpath": str(tmp_path / "poly.gpkg"),
            "rivershed_output_path": str(tmp_path / "rivers.gpkg"),
            "non_served_nxt_river_outpath": str(tmp_path / "out.gpkg"),
        }
    }
    monkeypatch.setattr(os, "chdir", lambda path: None)
    monkeypatch.setattr(fir, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(fir, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(sys, "argv", ["prog", "--max-workers", "0"])
    monkeypatch.setattr(
        fir.gpd,
        "read_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("read_file should not be called for invalid workers")),
    )

    with pytest.raises(ValueError, match="max_workers"):
        fir.main()
