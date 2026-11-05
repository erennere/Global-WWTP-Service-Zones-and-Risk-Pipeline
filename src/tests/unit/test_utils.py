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
