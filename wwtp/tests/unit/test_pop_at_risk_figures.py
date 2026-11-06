"""Tests for pop_at_risk_figures (moved from the coverage-campaign files)."""
import geopandas as gpd
import pandas as pd
import duckdb
import pytest
from shapely.geometry import Point, Polygon
from wwtp.figures_scripts import pop_at_risk_figures
from shapely.geometry import box
from wwtp.figures_scripts import pop_at_risk_figures as prf
import runpy

pytestmark = pytest.mark.unit


def _simple_plot_gdf(with_crs: bool = True) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {
            "metric": [1.0, 10.0, 100.0],
            "count": [5, 1, 7],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        },
        geometry="geometry",
    )
    if with_crs:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def test_pop_at_risk_helpers_with_dummy_data(monkeypatch, tmp_path):
    low, high = pop_at_risk_figures._robust_bounds([1, 2, 3, 4, 1000], positive_only=False)
    assert high > low

    with pytest.raises(ValueError):
        pop_at_risk_figures._robust_bounds([None, float("nan")], positive_only=True)

    tiles = gpd.GeoDataFrame(
        {
            "tile": ["t1", "t2"],
            "geometry": [Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]), Polygon([(1, 1), (2, 1), (1, 2), (1, 1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    pop = gpd.GeoDataFrame(
        {
            "tile": ["t1", "t2"],
            "500_2020_zonal_sum": [10.0, 20.0],
            "bad_col": [1, 2],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    called = []

    monkeypatch.setattr(
        pop_at_risk_figures,
        "create_single_plot",
        lambda **kwargs: called.append(kwargs["column"]),
    )

    pop_at_risk_figures.create_impact_polygon_plots(pop, tiles, str(tmp_path))

    assert called == ["500_2020_zonal_sum"]


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


def test_robust_bounds_degenerate_positive_path_executes_fallback_branch():
    low, high = prf._robust_bounds([2, 2, 2], positive_only=True)
    assert high > low > 0


def test_create_single_plot_error_paths_cover_missing_branches(tmp_path):
    gdf = _simple_plot_gdf(with_crs=True)

    with pytest.raises(KeyError):
        prf.create_single_plot(gdf, "missing_col", "T", "a.png", output_dir=str(tmp_path), show=False)

    with pytest.raises(KeyError):
        prf.create_single_plot(
            gdf,
            "metric",
            "T",
            "b.png",
            output_dir=str(tmp_path),
            min_count_col="missing_count",
            min_count=2,
            show=False,
        )

    with pytest.raises(ValueError):
        prf.create_single_plot(
            gdf.assign(metric=[0.0, -1.0, 0.0]),
            "metric",
            "T",
            "c.png",
            output_dir=str(tmp_path),
            scale_type="log",
            show=False,
        )

    with pytest.raises(ValueError):
        prf.create_single_plot(
            gdf,
            "metric",
            "T",
            "d.png",
            output_dir=str(tmp_path),
            scale_type="linear",
            vmin=1,
            vmax=1,
            show=False,
        )

    with pytest.raises(ValueError):
        prf.create_single_plot(
            gdf,
            "metric",
            "T",
            "e.png",
            output_dir=str(tmp_path),
            scale_type="bad",
            show=False,
        )


def test_create_single_plot_sets_crs_transforms_values_and_calls_show(tmp_path, monkeypatch):
    called = {"show": False}
    monkeypatch.setattr(prf.plt, "show", lambda: called.__setitem__("show", True))

    gdf = _simple_plot_gdf(with_crs=False)
    fig, _ = prf.create_single_plot(
        z8_stats=gdf,
        column="metric",
        title="T",
        output_filename="ok.png",
        output_dir=str(tmp_path),
        value_transform=lambda s: s * 2,
        legend_kwds={"label": "Custom"},
        author_note="note",
        show=True,
    )

    assert fig is not None
    assert called["show"] is True
    assert (tmp_path / "ok.png").exists()


def test_create_impact_polygon_plots_skip_paths(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(prf.logger, "warning", lambda *args, **kwargs: warnings.append(args[0]))

    tiles = gpd.GeoDataFrame(
        {"tile": [1], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    # No *_zonal_sum columns -> early warning/return
    pop_none = gpd.GeoDataFrame({"tile": [1], "geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")
    prf.create_impact_polygon_plots(pop_none, tiles, str(tmp_path))

    # Malformed column names -> skip warnings inside loop
    pop_bad = gpd.GeoDataFrame(
        {
            "tile": [1],
            "bad_zonal_sum": [10],          # len(parts) < 3
            "abc_2024_zonal_sum": [11],     # non-numeric radius
            "500_abc_zonal_sum": [12],      # non-numeric year
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    prf.create_impact_polygon_plots(pop_bad, tiles, str(tmp_path))

    assert any("No '*_zonal_sum'" in msg for msg in warnings)
    assert any("non-numeric radius/year" in msg for msg in warnings)


def test_create_impact_polygon_plots_skips_non_numeric_column(monkeypatch, tmp_path):
    short_col = "weird_zonal_sum"
    warnings = []
    monkeypatch.setattr(prf.logger, "warning", lambda *args, **kwargs: warnings.append(args[0]))

    tiles = gpd.GeoDataFrame(
        {"tile": [1], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    pop = gpd.GeoDataFrame(
        {
            "tile": [1],
            short_col: [10],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    prf.create_impact_polygon_plots(pop, tiles, str(tmp_path))
    assert any("non-numeric radius/year" in msg for msg in warnings)


def test_pop_at_risk_main_orchestrates_with_patched_io(monkeypatch, tmp_path):
    non_served_csv = tmp_path / "non_served.csv"
    cfg = {
        "paths": {
            "country_boundaries_filepath": "dummy_boundaries.geojson",
            "figures_dir": str(tmp_path),
            "pop_at_risk_output_filepath": str(tmp_path / "pop_at_risk.parquet"),
            "non_served_outpath": str(non_served_csv),
            "non_served_above_threshold_outpath": str(tmp_path / "non_served_above_threshold.gpkg"),
        },
        "zoom_level": 8,
        "save_dpi": 100,
        "threshold_value": 1000,
        "min_display_population": 1000,
        "country_id_column": "ISO_A2",
        "nodata_country_color": "#bdbdbd",
        "nodata_country_label": "NODATA",
        "plot_outlier_quantiles": [0.005, 0.995],
        "plot_outlier_iqr_factor": 3.0,
        "max_workers": 8,
    }

    boundaries = gpd.GeoDataFrame(
        {"ISO_A2": ["DE"], "geometry": [box(0, 0, 2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    pop_at_risk = gpd.GeoDataFrame(
        {"tile": ["t1"], "500_2024_zonal_sum": [100], "geometry": [Point(0.5, 0.5)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    tiles = gpd.GeoDataFrame(
        {"tile": ["t1"], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    calls = {"impact": 0, "single": 0}
    seen = {}

    class _DuckResult:
        def to_df(self):
            return pd.DataFrame({"tile": ["t1"], "pop_sum": [123.0]})

    monkeypatch.setattr(prf, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(prf, "load_config", lambda **kwargs: cfg)
    def _read_file(path, columns=None, **kwargs):
        if str(path).endswith("non_served_above_threshold.gpkg"):
            return gpd.GeoDataFrame(
                {"tile": ["t1"], "pop_sum": [250.0], "geometry": [box(0, 0, 1, 1)]},
                geometry="geometry",
                crs="EPSG:4326",
            )
        return boundaries.copy()

    monkeypatch.setattr(prf.gpd, "read_file", _read_file)
    def _read_parquet(path):
        seen["parquet_path"] = path
        return pop_at_risk.copy()

    monkeypatch.setattr(prf.gpd, "read_parquet", _read_parquet)
    monkeypatch.setattr(prf, "find_tiles_in_countries", lambda *args, **kwargs: tiles.copy())
    monkeypatch.setattr(prf, "create_impact_polygon_plots", lambda *args, **kwargs: calls.__setitem__("impact", calls["impact"] + 1))
    monkeypatch.setattr(prf, "create_single_plot", lambda **kwargs: calls.__setitem__("single", calls["single"] + 1))
    def _duckdb_sql(query):
        seen["duckdb_query"] = query
        return _DuckResult()

    monkeypatch.setattr(duckdb, "sql", _duckdb_sql)

    prf.main()

    assert calls["impact"] == 1
    assert calls["single"] == 1
    assert seen["parquet_path"] == cfg["paths"]["pop_at_risk_output_filepath"]
    assert "duckdb_query" not in seen




def test_pop_at_risk_module_main_guard_executes(monkeypatch, tmp_path):
    import os
    import duckdb
    import geopandas
    import wwtp.starter as starter_mod
    import wwtp.pop_at_risk_river_calculations.find_pop_in_danger_pop as fpip

    cfg = {
        "paths": {
            "country_boundaries_filepath": "dummy_boundaries.geojson",
            "figures_dir": str(tmp_path),
            "pop_at_risk_output_filepath": str(tmp_path / "pop_at_risk.parquet"),
            "non_served_outpath": str(tmp_path / "non_served.csv"),
            "non_served_above_threshold_outpath": str(tmp_path / "non_served_above_threshold.gpkg"),
        },
        "zoom_level": 8,
        "save_dpi": 100,
        "threshold_value": 1,
        "min_display_population": 1,
        "country_id_column": "ISO_A2",
        "nodata_country_color": "#bdbdbd",
        "nodata_country_label": "NODATA",
        "plot_outlier_quantiles": [0.005, 0.995],
        "plot_outlier_iqr_factor": 3.0,
        "max_workers": 8,
    }

    boundaries = gpd.GeoDataFrame(
        {"ISO_A2": ["DE"], "geometry": [box(0, 0, 2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    pop_at_risk = gpd.GeoDataFrame(
        {"tile": ["t1"], "500_2024_zonal_sum": [100], "geometry": [Point(0.5, 0.5)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    tiles = gpd.GeoDataFrame(
        {"tile": ["t1"], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    class _DuckResult:
        def to_df(self):
            return pd.DataFrame({"tile": ["t1"], "pop_sum": [150.0]})

    monkeypatch.setattr(os, "chdir", lambda path: None)
    monkeypatch.setattr(starter_mod, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(starter_mod, "load_config", lambda **kwargs: cfg)
    def _read_file(path, columns=None, **kwargs):
        if str(path).endswith("non_served_above_threshold.gpkg"):
            return gpd.GeoDataFrame(
                {"tile": ["t1"], "pop_sum": [250.0], "geometry": [box(0, 0, 1, 1)]},
                geometry="geometry",
                crs="EPSG:4326",
            )
        return boundaries.copy()

    monkeypatch.setattr(geopandas, "read_file", _read_file)
    monkeypatch.setattr(geopandas, "read_parquet", lambda path: pop_at_risk.copy())
    monkeypatch.setattr(fpip, "find_tiles_in_countries", lambda *args, **kwargs: tiles.copy())
    monkeypatch.setattr(duckdb, "sql", lambda q: _DuckResult())

    module_name = "wwtp.figures_scripts.pop_at_risk_figures"
    runpy.run_module(module_name, run_name="__main__")
