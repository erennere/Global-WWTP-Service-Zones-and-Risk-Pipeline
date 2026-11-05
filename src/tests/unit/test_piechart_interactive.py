"""Tests for piechart_interactive (moved from the coverage-campaign files)."""
import json
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from src.annotation_scripts import NEW_01_GENERATEGRIDS as gen_grids
from src.annotation_scripts import NEW_02_EXTRACTOSMDATAFULL_GEOJSON as osm_extract
from src.figures_scripts import piechart_figure, piechart_interactive
from src.figures_scripts import pop_at_risk_figures
import runpy
import matplotlib
from shapely.geometry import MultiPolygon, Point, Polygon, box
from src.figures_scripts import piechart_interactive as pif
from src.figures_scripts import pop_at_risk_figures as prf

pytestmark = pytest.mark.unit


def test_piechart_interactive_helpers_cover_branches():
    df = pd.DataFrame(
        {
            "country": ["DE", "DE", "FR"],
            "val": [1.0, 2.0, 3.0],
            "ind": [True, False, False],
            "population_served": [50.0, 100.0, 75.0],
            "population_total": [100.0, 200.0, 100.0],
        }
    )

    out_pop = piechart_interactive.aggregate_by_country(df, "country", "val", is_pop=True)
    out_mix = piechart_interactive.aggregate_by_country(df, "country", "val", industrial_column="ind", is_pop=False)
    assert set(out_pop["country"]) == {"DE", "FR"}
    assert "IND_val_sum" in out_mix.columns

    assert piechart_interactive.calculate_size(0, 1, 10, 5, 15) == 5

    c1 = df.copy()
    assert piechart_interactive.ensure_population_percentage_column(c1, preferred_col="ratio") == "ratio"
    assert "ratio" in c1.columns

    c2 = pd.DataFrame({"population_total": [100], "2024_zonal_sum_sum": [80]})
    assert piechart_interactive.ensure_population_percentage_column(c2, preferred_col="ratio", zonal_sum_col="2024_zonal_sum") == "ratio"

    assert piechart_interactive.resolve_zonal_sum_column(pd.DataFrame({"2019_zonal_sum": [1], "2024_zonal_sum": [2]}), "missing") == "2024_zonal_sum"

    svg = piechart_interactive.get_pie_svg([1, 2], [3, 4], 32)
    assert "<svg" in svg and "path" in svg


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
