from __future__ import annotations

from pathlib import Path
import runpy

import geopandas as gpd
import matplotlib
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from src.figures_scripts import piechart_interactive as pif
from src.figures_scripts import pop_at_risk_figures as prf


pytestmark = pytest.mark.unit
matplotlib.use("Agg")


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

    monkeypatch.setattr(prf.duckdb, "sql", _duckdb_sql)

    prf.main()

    assert calls["impact"] == 1
    assert calls["single"] == 1
    assert seen["parquet_path"] == cfg["paths"]["pop_at_risk_output_filepath"]
    assert "duckdb_query" not in seen


def test_piechart_interactive_helper_error_and_edge_paths():
    # calculate_size positive path
    assert pif.calculate_size(5, 1, 10, 10, 20) > 10
    assert pif.calculate_size(5, 10, 10, 10, 20) == 10

    # ensure_population_percentage_column preferred exists + error path
    df_pref = pd.DataFrame({"ratio": [0.2]})
    assert pif.ensure_population_percentage_column(df_pref, preferred_col="ratio") == "ratio"

    with pytest.raises(KeyError):
        pif.ensure_population_percentage_column(pd.DataFrame({"x": [1]}), preferred_col="ratio", zonal_sum_col="2024_zonal_sum")

    # resolve_zonal_sum_column preferred path + malformed candidate + no candidates
    df_pref2 = pd.DataFrame({"2020_zonal_sum": [1], "x": [2]})
    assert pif.resolve_zonal_sum_column(df_pref2, "2020_zonal_sum") == "2020_zonal_sum"

    df_malformed = pd.DataFrame({"abc_zonal_sum": [1], "x": [2]})
    assert pif.resolve_zonal_sum_column(df_malformed, "missing") == "abc_zonal_sum"

    with pytest.raises(KeyError):
        pif.resolve_zonal_sum_column(pd.DataFrame({"x": [1]}), "missing")

    svg = pif.get_pie_svg([0, 1], [0, 1], 40)
    assert "<svg" in svg


def test_piechart_interactive_main_end_to_end(tmp_path, monkeypatch):
    boundaries_fp = tmp_path / "boundaries.gpkg"
    pop_fp = tmp_path / "pop.gpkg"
    stats_fp = tmp_path / "stats.csv"
    html_fp = tmp_path / "interactive.html"

    boundaries = gpd.GeoDataFrame(
        {
            "ISO_A2_EH": ["DE", "FR"],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                MultiPolygon([
                    Polygon([(3, 0), (5, 0), (5, 2), (3, 2)]),
                    Polygon([(6, 0), (7, 0), (7, 1), (6, 1)]),
                ]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    boundaries.to_file(boundaries_fp, driver="GPKG")

    pop_gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "DE", "FR", "FR"],
            "category_number": [1, 10, 1, 10],
            "2024_zonal_sum": [1000, 1200, 800, 900],
            "round_area": [30000, 20000, 25000, 30000],
            "wwtp_area_rect_2": [22000, 18000, 21000, 19000],
            "population_served": [700, 900, 500, 700],
            "population_total": [1500, 1500, 1200, 1200],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(3, 0, 4, 1), box(4, 0, 5, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    pop_gdf.to_file(pop_fp, driver="GPKG")

    pd.DataFrame(
        {
            "country": ["DE", "FR"],
            "population_total": [3000, 2400],
            "population_served": [1600, 1200],
        }
    ).to_csv(stats_fp, index=False)

    cfg = {
        "figures": {"approach": 1},
        "min_total_size": 10000,
        "zonal_sum_default_column": "2024_zonal_sum",
        "industrial_category_numbers": [10],
        "mix_use_categories": [],
        "paths": {
            "country_boundaries_filepath": str(boundaries_fp),
            "raster_country_stats_filepath": str(stats_fp),
            "interactive_piechart_html_filepath": str(html_fp),
        },
    }

    monkeypatch.setattr(pif, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(pif, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(pif, "create_pop_output_paths", lambda cfg: {"voronoi": {"1": str(pop_fp)}})

    pif.main()

    assert html_fp.exists()
    text = html_fp.read_text(encoding="utf-8")
    assert "WWTP Type Breakdown" in text


def test_piechart_interactive_main_raises_for_missing_stats_file(tmp_path, monkeypatch):
    boundaries_fp = tmp_path / "boundaries.gpkg"
    pop_fp = tmp_path / "pop.gpkg"

    boundaries = gpd.GeoDataFrame(
        {"ISO_A2_EH": ["DE"], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    boundaries.to_file(boundaries_fp, driver="GPKG")

    pop_gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "DE"],
            "category_number": [1, 10],
            "2024_zonal_sum": [100, 200],
            "round_area": [20000, 20000],
            "wwtp_area_rect_2": [12000, 12000],
            "geometry": [box(0, 0, 0.5, 0.5), box(0.5, 0, 1, 0.5)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    pop_gdf.to_file(pop_fp, driver="GPKG")

    cfg = {
        "figures": {"approach": 1},
        "min_total_size": 10000,
        "zonal_sum_default_column": "2024_zonal_sum",
        "industrial_category_numbers": [10],
        "mix_use_categories": [],
        "paths": {
            "country_boundaries_filepath": str(boundaries_fp),
            "raster_country_stats_filepath": str(tmp_path / "missing.csv"),
            "interactive_piechart_html_filepath": str(tmp_path / "interactive.html"),
        },
    }

    monkeypatch.setattr(pif, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(pif, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(pif, "create_pop_output_paths", lambda cfg: {"voronoi": {"1": str(pop_fp)}})

    with pytest.raises(FileNotFoundError):
        pif.main()


def test_piechart_interactive_import_fallback_block_executes():
    module_path = Path(__file__).resolve().parents[3] / "src" / "figures_scripts" / "piechart_interactive.py"
    module_globals = runpy.run_path(str(module_path), run_name="not_main")
    assert "aggregate_by_country" in module_globals


def test_pop_at_risk_module_main_guard_executes(monkeypatch, tmp_path):
    import os
    import duckdb
    import geopandas
    import src.starter as starter_mod
    import src.pop_at_risk_river_calculations.find_pop_in_danger_pop as fpip

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

    module_path = Path(__file__).resolve().parents[3] / "src" / "figures_scripts" / "pop_at_risk_figures.py"
    runpy.run_path(str(module_path), run_name="__main__")
