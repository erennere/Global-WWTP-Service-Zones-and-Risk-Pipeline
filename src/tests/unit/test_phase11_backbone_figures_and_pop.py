from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src import download_pop as dp
from src.figures_scripts import composite_area_population_plots as capp
from src.figures_scripts import pop_at_risk_figures as prf


pytestmark = pytest.mark.unit


def test_composite_main_end_to_end_with_sample_files(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")

    pop_path = tmp_path / "pop.gpkg"
    boundaries_path = tmp_path / "bounds.gpkg"
    hist_path = tmp_path / "hist.png"
    scatter_path = tmp_path / "scatter.png"

    pop_gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "DE", "FR", "FR"],
            "total_area": [100.0, 120.0, 80.0, 90.0],
            "round_area": [40.0, 50.0, 30.0, 35.0],
            "2024_zonal_sum": [1000.0, 1200.0, 900.0, 950.0],
            "geometry": [Point(0, 0), Point(0.1, 0.1), Point(1, 0), Point(1.1, 0.1)],
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

    class _Args:
        approach = "1"
        color_col = "ECONOMY"
        zonal_col = None
        hist_lower_q = 0.01
        hist_upper_q = 0.99

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

    monkeypatch.setattr(capp, "parse_args", lambda: _Args())
    monkeypatch.setattr(capp, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(capp, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(capp, "create_pop_output_paths", lambda cfg: {"voronoi": {"1": str(pop_path)}})

    capp.main()

    assert hist_path.exists()
    assert scatter_path.exists()


def test_composite_helpers_error_paths():
    with pytest.raises(KeyError):
        capp.resolve_zonal_sum_column(pd.DataFrame({"x": [1]}), "2024_zonal_sum")

    # Empty/NaN data path in clip_outliers
    clipped = capp.clip_outliers(pd.Series([np.nan, np.inf, -np.inf]), 0.05, 0.95)
    assert clipped.empty


def test_composite_main_raises_when_required_columns_are_missing(monkeypatch, tmp_path):
    class _Args:
        approach = None
        color_col = "ECONOMY"
        zonal_col = None
        hist_lower_q = 0.01
        hist_upper_q = 0.99

    cfg = {
        "figures": {"approach": 1},
        "zonal_sum_default_column": "2024_zonal_sum",
        "min_country_facility_count": 25,
        "plot_outlier_quantiles": [0.005, 0.995],
        "plot_outlier_iqr_factor": 3.0,
        "paths": {
            "country_boundaries_filepath": str(tmp_path / "bounds.gpkg"),
            "composite_histogram_filepath": str(tmp_path / "hist.png"),
            "composite_scatter_filepath": str(tmp_path / "scatter.png"),
        },
    }
    pop_path = os.path.abspath(str(tmp_path / "pop.gpkg"))
    pop_df = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE"],
            "total_area": [10.0],
            "2024_zonal_sum": [20.0],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    boundaries = gpd.GeoDataFrame(
        {"ISO_A2": ["DE"], "ECONOMY": ["High income"], "geometry": [box(-1, -1, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(capp.os, "chdir", lambda path: None)
    monkeypatch.setattr(capp, "parse_args", lambda: _Args())
    monkeypatch.setattr(capp, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(capp, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(capp, "create_pop_output_paths", lambda _: {"voronoi": {"1": pop_path}})
    monkeypatch.setattr(capp.gpd, "read_file", lambda path: pop_df.copy() if path == pop_path else boundaries.copy())

    with pytest.raises(KeyError, match="Missing required columns"):
        capp.main()


def test_composite_main_raises_when_color_column_is_missing(monkeypatch, tmp_path):
    class _Args:
        approach = None
        color_col = "ECONOMY"
        zonal_col = None
        hist_lower_q = 0.01
        hist_upper_q = 0.99

    cfg = {
        "figures": {"approach": 1},
        "zonal_sum_default_column": "2024_zonal_sum",
        "min_country_facility_count": 25,
        "plot_outlier_quantiles": [0.005, 0.995],
        "plot_outlier_iqr_factor": 3.0,
        "paths": {
            "country_boundaries_filepath": str(tmp_path / "bounds.gpkg"),
            "composite_histogram_filepath": str(tmp_path / "hist.png"),
            "composite_scatter_filepath": str(tmp_path / "scatter.png"),
        },
    }
    pop_path = os.path.abspath(str(tmp_path / "pop.gpkg"))
    pop_df = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE"],
            "total_area": [10.0],
            "round_area": [5.0],
            "2024_zonal_sum": [20.0],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    boundaries = gpd.GeoDataFrame(
        {"ISO_A2": ["DE"], "geometry": [box(-1, -1, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(capp.os, "chdir", lambda path: None)
    monkeypatch.setattr(capp, "parse_args", lambda: _Args())
    monkeypatch.setattr(capp, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(capp, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(capp, "create_pop_output_paths", lambda _: {"voronoi": {"1": pop_path}})
    monkeypatch.setattr(capp.gpd, "read_file", lambda path: pop_df.copy() if path == pop_path else boundaries.copy())

    with pytest.raises(KeyError) as excinfo:
        capp.main()

    assert "Color column" in str(excinfo.value)


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


def test_pop_at_risk_create_single_plot_linear_and_log(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    gdf = gpd.GeoDataFrame(
        {
            "tile": [1, 2, 3],
            "pop_sum": [10.0, 100.0, 1000.0],
            "cnt": [5, 1, 9],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    fig1, _ = prf.create_single_plot(
        z8_stats=gdf,
        column="pop_sum",
        title="Log",
        output_filename="log.png",
        output_dir=str(tmp_path),
        scale_type="log",
        show=False,
    )
    assert fig1 is not None

    fig2, _ = prf.create_single_plot(
        z8_stats=gdf,
        column="pop_sum",
        title="Linear",
        output_filename="lin.png",
        output_dir=str(tmp_path),
        scale_type="linear",
        min_count_col="cnt",
        min_count=2,
        show=False,
    )
    assert fig2 is not None
    assert (tmp_path / "log.png").exists()
    assert (tmp_path / "lin.png").exists()


def test_composite_filter_countries_by_facility_count():
    pop_df = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "DE", "FR", "IT"],
            "total_area": [1.0, 2.0, 3.0, 4.0],
            "round_area": [1.0, 2.0, 3.0, 4.0],
            "2024_zonal_sum": [10.0, 20.0, 30.0, 40.0],
            "geometry": [box(0, 0, 1, 1)] * 4,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    boundaries = gpd.GeoDataFrame(
        {"ISO_A2": ["DE", "FR", "IT"], "ECONOMY": ["A", "B", "C"], "geometry": [box(0, 0, 1, 1)] * 3},
        geometry="geometry",
        crs="EPSG:4326",
    )

    filtered_pop, filtered_boundaries, removed = capp.filter_countries_by_facility_count(pop_df, boundaries, "ISO_2", 2)

    assert set(filtered_pop["ISO_2"]) == {"DE"}
    assert set(filtered_boundaries["ISO_A2"]) == {"DE"}
    assert removed == ["FR", "IT"]


def test_pop_at_risk_create_impact_polygon_plots(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(prf, "create_single_plot", lambda **kwargs: calls.append(kwargs))

    tiles = gpd.GeoDataFrame(
        {"tile": [1, 2], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    pop = gpd.GeoDataFrame(
        {"tile": [1, 2], "500_2024_zonal_sum": [100, 200], "1000_2023_zonal_sum": [50, 90], "geometry": [Point(0, 0), Point(1, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    prf.create_impact_polygon_plots(pop, tiles, str(tmp_path))
    assert len(calls) == 2


def test_download_pop_process_single_country_csv_and_tif_paths(tmp_path, monkeypatch):
    base = tmp_path / "population"
    csv_dir = base / "unzipped" / "deu"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "part.csv"
    pd.DataFrame({"lat": [0, 0.001], "lon": [0, 0.001], "pop": [1, 2]}).to_csv(csv_path, index=False)

    tif_dir = base / "unzipped" / "fra"
    tif_dir.mkdir(parents=True, exist_ok=True)

    # Create tiny tif
    import rasterio
    from rasterio.transform import from_origin

    arr = np.ones((5, 5), dtype=np.float32)
    tif_path = tif_dir / "part.tif"
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=from_origin(0, 1, 0.01, 0.01),
    ) as dst:
        dst.write(arr, 1)

    monkeypatch.setattr(dp, "download_save_and_unzip_pops", lambda *args, **kwargs: str(csv_dir))

    mosaics = []
    monkeypatch.setattr(dp, "mosaic_large_rasters", lambda files, out: mosaics.append((tuple(files), out)))

    dp.process_single_country({"deu": ["u"]}, "deu", res=100, data_dir=str(base))
    assert len(mosaics) >= 1

    monkeypatch.setattr(dp, "download_save_and_unzip_pops", lambda *args, **kwargs: str(tif_dir))
    dp.process_single_country({"fra": ["u"]}, "fra", res=100, data_dir=str(base))
    assert len(mosaics) >= 2


def test_download_pop_process_all_and_main(monkeypatch, tmp_path):
    class _Future:
        def __init__(self, fn, args):
            self._fn = fn
            self._args = args

        def result(self):
            return self._fn(*self._args)

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            return _Future(fn, args)

    monkeypatch.setattr(dp, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(dp, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(dp, "tqdm", lambda x, total=None, desc=None: x)

    calls = []
    monkeypatch.setattr(dp, "process_single_country", lambda country_urls, country, res, data_dir: calls.append((country, res, data_dir)))

    urls = {"deu": ["u1"], "fra": ["u2"]}
    dp.process_all_countries(urls, res=120, max_workers=2, data_dir=str(tmp_path))
    assert len(calls) == 2

    monkeypatch.setattr(dp, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(
        dp,
        "load_config",
        lambda **kwargs: {
            "paths": {"pop_dir": str(tmp_path)},
            "start_year": 2015,
            "end_year": 2024,
            "worldpop_2014_url_template": "tpl2014",
            "include_2014": False,
            "worldpop_yearly_url_template": "tplyear",
            "country_limit": 0,
        },
    )
    monkeypatch.setattr(dp, "get_urls", lambda **kwargs: {"a": ["u"], "b": ["u"], "c": ["u"], "d": ["u"]})
    monkeypatch.setattr(dp, "process_all_countries", lambda country_urls, res, max_workers, data_dir: calls.append((len(country_urls), res, max_workers, data_dir)))

    # country_limit=0 means "no limit" - all four countries are processed.
    # (YAML cannot express a literal null here, so 0 is the sentinel.)
    dp.main(res=90, max_workers=1)
    assert calls[-1][0] == 4

    # A positive country_limit really truncates, alphabetically.
    monkeypatch.setattr(
        dp,
        "load_config",
        lambda **kwargs: {
            "paths": {"pop_dir": str(tmp_path)},
            "start_year": 2015,
            "end_year": 2024,
            "worldpop_2014_url_template": "tpl2014",
            "include_2014": False,
            "worldpop_yearly_url_template": "tplyear",
            "country_limit": 3,
        },
    )
    dp.main(res=90, max_workers=1)
    assert calls[-1][0] == 3
