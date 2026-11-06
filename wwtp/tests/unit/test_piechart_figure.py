"""Tests for piechart_figure (moved from the coverage-campaign files)."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon
from wwtp.figures_scripts import piechart_figure
import numpy as np
from shapely.geometry import box
from wwtp.figures_scripts import piechart_figure as pf

pytestmark = pytest.mark.unit


class _Wedge:
    def set_edgecolor(self, _):
        return None

    def set_linewidth(self, _):
        return None


class _InsetAx:
    def __init__(self):
        self.transData = object()
        self.transAxes = object()

    def grid(self, *_args, **_kwargs):
        return None

    def set_axis_off(self):
        return None

    def axis(self, *_args, **_kwargs):
        return None

    def pie(self, values, **_kwargs):
        return ([_Wedge() for _ in values], [])

    def add_patch(self, _):
        return None

    def annotate(self, *_args, **_kwargs):
        return None

    def set_title(self, *_args, **_kwargs):
        return None


def test_piechart_figure_helpers_cover_branches():
    poly = Polygon([(0, 0), (1, 0), (0, 1), (0, 0)])
    mpoly = MultiPolygon([poly, Polygon([(10, 10), (12, 10), (10, 12), (10, 10)])])

    x1, y1 = piechart_figure.get_pos(poly)
    x2, y2 = piechart_figure.get_pos(mpoly)
    assert isinstance(x1, float) and isinstance(y1, float)
    assert isinstance(x2, float) and isinstance(y2, float)

    assert piechart_figure.calculate_size(10, 1, 100, 1, 5, scale="linear") > 1
    assert piechart_figure.calculate_size(10, 1, 100, 1, 5, scale="log") > 1
    with pytest.raises(ValueError):
        piechart_figure.calculate_size(10, 1, 100, 1, 5, scale="bad")

    rounded = piechart_figure.round_numbers([100, 1000, 10000], [1, 2, 3])
    assert len(rounded) == 3

    pref = piechart_figure.resolve_zonal_sum_columns(pd.DataFrame({"2020_zonal_sum": [1]}), "2020_zonal_sum")
    assert pref == "2020_zonal_sum"


def test_piechart_aggregation_and_scaling_paths():
    df = pd.DataFrame(
        {
            "country": ["DE", "DE", "FR", "FR"],
            "value": [10, 30, 20, 40],
            "IND/RES": [True, False, True, False],
        }
    )

    pop_agg = pf.aggregate_by_country(df, "country", "value", industrial_column=None, is_pop=True)
    assert "value_sum" in pop_agg.columns

    type_agg = pf.aggregate_by_country(df, "country", "value", industrial_column="IND/RES", is_pop=False)
    assert any(c.startswith("IND_") for c in type_agg.columns)
    assert any(c.startswith("RES_") for c in type_agg.columns)

    sizes = [
        pf.calculate_size(v, min_value=10, max_value=100, min_size=0.1, max_size=1.0, scale="linear")
        for v in [10, 30, 90]
    ]
    assert all(isinstance(s, float) for s in sizes)

    with pytest.raises(ValueError):
        pf.calculate_size(10, 1, 2, 0.1, 1.0, scale="bad")


def test_piechart_position_and_rounding_helpers():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    mpoly = MultiPolygon([poly, box(2, 2, 5, 5)])

    x1, y1 = pf.get_pos(poly)
    x2, y2 = pf.get_pos(mpoly)
    assert isinstance(x1, float)
    assert isinstance(y2, float)

    labels = pf.round_numbers(np.array([10, 100, 1000, np.nan]), [1, 2, 3, 4])
    assert len(labels) == 4


def test_piechart_percentage_and_zonal_resolution_helpers():
    df = pd.DataFrame({"population_served": [50, 30], "population_total": [100, 60]})
    col = pf.ensure_population_percentage_column(
        df, preferred_col="population_served_index", zonal_sum_col="2024_zonal_sum"
    )
    assert col == "population_served_index"
    assert np.isclose(df[col].iloc[0], 0.5)

    df2 = pd.DataFrame({"2023_zonal_sum": [10], "2024_zonal_sum": [11], "population_total": [20]})
    chosen = pf.resolve_zonal_sum_columns(df2, preferred="missing_zonal_sum")
    assert chosen == "2024_zonal_sum"


def test_plot_splitted_piechart_executes(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    pf.plot_splitted_piechart(
        dist_tag1=[20, 80],
        dist_tag2=[45, 55],
        ax=ax,
        size_tag1=100,
        size_tag2=50,
        min_size=5,
        labels=True,
        labels_text=["A", "B", "C"],
    )
    out = tmp_path / "pie.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_piechart_figure_population_helper_branches():
    df_pref = pd.DataFrame({"population_served_index": [0.4]})
    assert pf.ensure_population_percentage_column(df_pref) == "population_served_index"

    df_served = pd.DataFrame({"population_served": [50], "population_total": [100]})
    col = pf.ensure_population_percentage_column(df_served)
    assert col == "population_served_index"
    assert df_served[col].iloc[0] == 0.5

    df_zonal = pd.DataFrame({"2024_zonal_sum_sum": [20], "population_total": [100]})
    col2 = pf.ensure_population_percentage_column(df_zonal)
    assert col2 == "population_served_index"
    assert df_zonal[col2].iloc[0] == 0.2

    with pytest.raises(KeyError):
        pf.ensure_population_percentage_column(pd.DataFrame({"x": [1]}))


def test_piechart_figure_resolve_zonal_sum_fallbacks():
    d1 = pd.DataFrame({"2024_zonal_sum": [1]})
    assert pf.resolve_zonal_sum_columns(d1, "2024_zonal_sum") == "2024_zonal_sum"

    d2 = pd.DataFrame({"2020_zonal_sum": [1], "2023_zonal_sum": [2], "abc_zonal_sum": [3]})
    assert pf.resolve_zonal_sum_columns(d2, "missing_col") == "2023_zonal_sum"

    with pytest.raises(KeyError):
        pf.resolve_zonal_sum_columns(pd.DataFrame({"x": [1]}), "missing_col")


def test_piechart_figure_main_smoke(monkeypatch, tmp_path):
    boundaries_fp = tmp_path / "boundaries.gpkg"
    pop_fp = tmp_path / "pop.gpkg"
    stats_fp = tmp_path / "stats.csv"
    out_fp = tmp_path / "pie.png"

    cfg = {
        "figures": {"approach": "0"},
        "min_total_size": 10000,
        "paths": {
            "country_boundaries_filepath": str(boundaries_fp),
            "raster_country_stats_filepath": str(stats_fp),
            "static_piechart_filepath": str(out_fp),
        },
        "zonal_sum_default_column": "2024_zonal_sum",
        "industrial_category_numbers": [9],
        "mix_use_categories": [],
        "nodata_country_color": "#bdbdbd",
        "nodata_country_label": "NODATA",
    }

    boundaries = gpd.GeoDataFrame(
        {
            "ISO_A2_EH": ["DE", "FR"],
            "geometry": [box(5, 47, 15, 55), box(-5, 43, 8, 51)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    pop_gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "DE"],
            "category_number": [9, 1],
            "2024_zonal_sum": [8_000_000, 7_000_000],
            "round_area": [30_000_000, 28_000_000],
            "wwtp_area_rect_2": [10_000_000, 9_000_000],
            "num_detection_circle": [4, 3],
            "num_detection_rect": [5, 4],
            "total_area": [40_000_000, 37_000_000],
            "geometry": [Point(10, 51), Point(11, 50)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    pd.DataFrame({"country": ["DE", "FR"], "population_total": [80_000_000, 65_000_000]}).to_csv(stats_fp, index=False)

    monkeypatch.setattr(pf, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(pf, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(pf, "create_pop_output_paths", lambda cfg_in: {"voronoi": {"0": str(pop_fp)}})

    def _read_file(path):
        p = str(path)
        if p == str(boundaries_fp):
            return boundaries.copy()
        if p == str(pop_fp):
            return pop_gdf.copy()
        raise FileNotFoundError(p)

    monkeypatch.setattr(pf.gpd, "read_file", _read_file)
    monkeypatch.setattr(gpd.GeoDataFrame, "plot", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(pf, "inset_axes", lambda *args, **kwargs: _InsetAx())

    saved = {"path": None}
    monkeypatch.setattr(pf.plt, "savefig", lambda path, dpi=200: saved.update({"path": str(path), "dpi": dpi}))
    monkeypatch.setattr(pf, "ensure_output_dir_for_file", lambda path: None)

    pf.main()
    assert saved["path"] == str(out_fp)


def test_piechart_figure_main_requires_figures_approach(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "country_boundaries_filepath": str(tmp_path / "boundaries.gpkg"),
            "raster_country_stats_filepath": str(tmp_path / "stats.csv"),
            "static_piechart_filepath": str(tmp_path / "out.png"),
        },
        "zonal_sum_default_column": "2024_zonal_sum",
        "industrial_category_numbers": [9],
        "mix_use_categories": [],
    }

    monkeypatch.setattr(pf, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(pf, "load_config", lambda **kwargs: cfg)

    with pytest.raises(KeyError, match="figures"):
        pf.main()


def test_piechart_figure_main_requires_voronoi_mapping_for_approach(monkeypatch, tmp_path):
    cfg = {
        "figures": {"approach": "1"},
        "paths": {
            "country_boundaries_filepath": str(tmp_path / "boundaries.gpkg"),
            "raster_country_stats_filepath": str(tmp_path / "stats.csv"),
            "static_piechart_filepath": str(tmp_path / "out.png"),
        },
        "zonal_sum_default_column": "2024_zonal_sum",
        "industrial_category_numbers": [9],
        "mix_use_categories": [],
    }

    monkeypatch.setattr(pf, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(pf, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(pf, "create_pop_output_paths", lambda _cfg: {"voronoi": {"0": "pop.gpkg"}})

    with pytest.raises(KeyError, match="1"):
        pf.main()
