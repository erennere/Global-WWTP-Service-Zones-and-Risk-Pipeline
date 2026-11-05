"""Tests for annotation_export_geotiff (moved from the coverage-campaign files)."""
import io
import runpy
from unittest.mock import mock_open, patch
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box
from src.pop_at_risk_river_calculations import create_rasters

pytestmark = pytest.mark.unit


def test_new04_exportgeotiff_raises_when_id_field_missing(monkeypatch):
    grid = gpd.GeoDataFrame(
        {"wrong_id": [1], "geometry": [box(0, 0, 10, 10)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    class _Layer:
        def plot(self, ax=None, color=None, edgecolor=None):
            return ax

    calls = {"n": 0}

    def fake_read_file(path):
        calls["n"] += 1
        return grid.copy() if calls["n"] == 1 else _Layer()

    monkeypatch.setattr("geopandas.read_file", fake_read_file)
    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)

    with pytest.raises(Exception, match="field 'idx'"):
        runpy.run_module("src.annotation_scripts.NEW_04_EXPORTGEOTIFF", run_name="__main__")
