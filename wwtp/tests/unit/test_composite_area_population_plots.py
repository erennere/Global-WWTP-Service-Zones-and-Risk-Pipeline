"""Tests for composite_area_population_plots (moved from the coverage-campaign files)."""
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box
from wwtp.figures_scripts import composite_area_population_plots as capp
from wwtp.figures_scripts import composite_area_population_plots

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


def test_resolve_zonal_sum_column_prefers_requested_or_latest_year():
    df = pd.DataFrame({"2019_zonal_sum": [1], "2024_zonal_sum": [2], "foo": [3]})

    preferred = composite_area_population_plots.resolve_zonal_sum_column(df, "2019_zonal_sum")
    fallback = composite_area_population_plots.resolve_zonal_sum_column(df, "missing_col")

    assert preferred == "2019_zonal_sum"
    assert fallback == "2024_zonal_sum"


def test_resolve_zonal_sum_column_raises_when_no_candidates_exist():
    with pytest.raises(KeyError, match=r"No '\*_zonal_sum' column"):
        composite_area_population_plots.resolve_zonal_sum_column(pd.DataFrame({"x": [1]}), "2024_zonal_sum")


def test_clip_outliers_removes_inf_nan_and_quantile_outliers():
    series = pd.Series([1, 2, 3, 1000, np.inf, -np.inf, np.nan])

    clipped = composite_area_population_plots.clip_outliers(series, lower_q=0.05, upper_q=0.95)

    assert clipped.notna().all()
    assert np.isfinite(clipped).all()
    assert 1000 not in clipped.tolist()


def test_clip_outliers_rejects_invalid_quantile_bounds():
    with pytest.raises(ValueError, match="Quantile bounds"):
        composite_area_population_plots.clip_outliers(pd.Series([1, 2, 3]), lower_q=0.9, upper_q=0.1)


def test_make_category_color_map_is_deterministic_for_unique_categories():
    values = ["B", "A", None, "A"]

    color_map = composite_area_population_plots.make_category_color_map(values)

    assert set(color_map.keys()) == {"A", "B", "Unknown"}
    assert color_map["A"] == composite_area_population_plots.make_category_color_map(values)["A"]


def test_add_one_to_one_line_adds_reference_line_for_non_empty_data():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x_vals = pd.Series([1.0, 2.0, 3.0])
    y_vals = pd.Series([1.5, 2.5, 3.5])

    composite_area_population_plots.add_one_to_one_line(ax, x_vals, y_vals)

    assert len(ax.lines) == 1
    x_line, y_line = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    assert list(x_line) == list(y_line)
    plt.close(fig)


def test_resolve_zonal_sum_column_ignores_unparseable_candidates():
    df = pd.DataFrame({"abc_zonal_sum": [1], "2024_zonal_sum": [2]})

    result = composite_area_population_plots.resolve_zonal_sum_column(df, "missing_col")

    assert result == "2024_zonal_sum"


def test_add_one_to_one_line_skips_empty_series_and_handles_flat_range():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, (ax_empty, ax_flat) = plt.subplots(1, 2)

    composite_area_population_plots.add_one_to_one_line(ax_empty, pd.Series([np.nan]), pd.Series([np.nan]))
    composite_area_population_plots.add_one_to_one_line(ax_flat, pd.Series([5.0]), pd.Series([5.0]))

    assert len(ax_empty.lines) == 0
    assert len(ax_flat.lines) == 1
    x_line = ax_flat.lines[0].get_xdata()
    assert x_line[1] > x_line[0]
    plt.close(fig)
