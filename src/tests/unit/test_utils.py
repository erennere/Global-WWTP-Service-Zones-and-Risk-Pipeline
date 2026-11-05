from __future__ import annotations

import pytest

from src import utils
import os
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
from src import download_pop as dp
from src.figures_scripts import composite_area_population_plots as capp
from src.figures_scripts import pop_at_risk_figures as prf
import io
import runpy
from unittest.mock import mock_open, patch
from shapely.geometry import Point, Polygon, box
from src.pop_at_risk_river_calculations import create_rasters

pytestmark = pytest.mark.unit


def test_ensure_output_dir_for_file_creates_parent_directory(tmp_path):
    target = tmp_path / "nested" / "deeper" / "output.gpkg"

    utils.ensure_output_dir_for_file(str(target))

    assert target.parent.is_dir()


def test_ensure_output_dir_for_file_handles_bare_filename():
    # Should not raise when the path has no directory component.
    utils.ensure_output_dir_for_file("output.gpkg")


def test_get_iso_codes_maps_known_country_both_directions():
    alpha_3_to_2, alpha_2_to_3, alpha_3_to_names, alpha_2_to_names = utils.get_iso_codes()

    assert alpha_3_to_2["DEU"] == "DE"
    assert alpha_2_to_3["DE"] == "DEU"
    assert alpha_3_to_names["DEU"] == "Germany"
    assert alpha_2_to_names["DE"] == "Germany"


def test_quote_sql_identifier_wraps_plain_name():
    assert utils.quote_sql_identifier("ISO_2") == '"ISO_2"'


def test_quote_sql_identifier_escapes_embedded_quotes():
    assert utils.quote_sql_identifier('weird"name') == '"weird""name"'


# --- moved from the coverage-campaign files ---

def test_composite_script_entrypoint_runs_via_fallback_imports(monkeypatch, tmp_path):
    import runpy
    import sys

    import src.pipelines as pipelines_mod
    import src.starter as starter_mod

    import matplotlib
    matplotlib.use("Agg", force=True)

    pop_path = tmp_path / "pop.gpkg"
    boundaries_path = tmp_path / "bounds.gpkg"
    hist_path = tmp_path / "hist.png"
    scatter_path = tmp_path / "scatter.png"

    pop_gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "FR"],
            "total_area": [100.0, 80.0],
            "round_area": [40.0, 30.0],
            "2024_zonal_sum": [1000.0, 900.0],
            "geometry": [Point(0, 0), Point(1, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    pop_gdf.to_file(pop_path, driver="GPKG")

    boundaries = gpd.GeoDataFrame(
        {
            "ISO_A2": ["DE", "FR"],
            "ECONOMY": ["High income", "High income"],
            "geometry": [box(-1, -1, 0.5, 0.5), box(0.5, -1, 2, 0.5)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    boundaries.to_file(boundaries_path, driver="GPKG")

    cfg = {
        "figures": {"approach": 1},
        "zonal_sum_default_column": "2024_zonal_sum",
        "min_country_facility_count": 25,
        "plot_outlier_quantiles": [0.005, 0.995],
        "plot_outlier_iqr_factor": 3.0,
        "paths": {
            "country_boundaries_filepath": str(boundaries_path),
            "composite_histogram_filepath": str(hist_path),
            "composite_scatter_filepath": str(scatter_path),
        },
    }
    module_path = Path(__file__).resolve().parents[3] / "src" / "figures_scripts" / "composite_area_population_plots.py"

    monkeypatch.setattr(sys, "argv", [str(module_path)])
    monkeypatch.setattr(os, "chdir", lambda path: None)
    monkeypatch.setattr(starter_mod, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(starter_mod, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(pipelines_mod, "create_pop_output_paths", lambda _: {"voronoi": {"1": str(pop_path)}})
    monkeypatch.setattr("src.utils.ensure_output_dir_for_file", lambda path: None)

    runpy.run_path(str(module_path), run_name="__main__")

    assert hist_path.exists()
    assert scatter_path.exists()


def test_new04_exportgeotiff_script_executes_with_dummy_data(monkeypatch):
    grid = gpd.GeoDataFrame(
        {"idx": [1], "geometry": [box(0, 0, 10, 10)]},
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

    class _Ax:
        def set_xlim(self, a, b):
            return None

        def set_ylim(self, a, b):
            return None

        def axis(self, value):
            return None

    class _Fig:
        def savefig(self, *args, **kwargs):
            return None

    monkeypatch.setattr("src.utils.ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr("geopandas.read_file", fake_read_file)
    monkeypatch.setattr("matplotlib.pyplot.subplots", lambda *args, **kwargs: (_Fig(), _Ax()))
    monkeypatch.setattr("matplotlib.pyplot.close", lambda fig: None)
    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)

    with patch("builtins.open", mock_open()):
        runpy.run_module("src.annotation_scripts.NEW_04_EXPORTGEOTIFF", run_name="__main__")
