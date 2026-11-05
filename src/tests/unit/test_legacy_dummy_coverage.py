from __future__ import annotations

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


pytestmark = pytest.mark.unit


def test_point_to_square_handles_none_and_builds_bounds():
    assert gen_grids.point_to_square(None, 5) is None
    square = gen_grids.point_to_square(Point(10, 20), 2)
    assert square.bounds == pytest.approx((8, 18, 12, 22))


def test_generategrids_main_filters_points_and_writes(monkeypatch, tmp_path):
    src = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0), Point(1, 1), LineString([(0, 0), (1, 1)])],
            "id": [1, 2, 3],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    captured = {}
    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, driver=None, **kwargs):
        captured["rows"] = len(self)
        captured["path"] = path
        captured["driver"] = driver
        captured["geom_type"] = self.geometry.iloc[0].geom_type

    monkeypatch.setattr(gen_grids.gpd, "read_file", lambda path: src.copy())
    monkeypatch.setattr(gen_grids, "ensure_output_dir_for_file", lambda path: captured.setdefault("ensured", path))
    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        gen_grids.main(100.0, 50.0, "in.gpkg", str(tmp_path / "out.gpkg"))
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["rows"] == 2
    assert captured["driver"] == "GPKG"
    assert captured["geom_type"] == "Polygon"


def test_clean_columns_normalizes_names_and_values():
    gdf = gpd.GeoDataFrame(
        {
            "bad col!": [1],
            "another-col": [None],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    out = osm_extract.clean_columns(gdf)

    assert "bad_col_" in out.columns
    assert "another_col" in out.columns
    assert out["bad_col_"].iloc[0] == "1"


def test_query_overpass_handles_success_rate_limit_and_bad_json(monkeypatch):
    class _Response:
        def __init__(self, status, payload=None, bad_json=False):
            self.status_code = status
            self._payload = payload or {}
            self._bad_json = bad_json

        def json(self):
            if self._bad_json:
                raise json.JSONDecodeError("bad", "x", 0)
            return self._payload

    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(429)
        if calls["n"] == 2:
            return _Response(200, payload={"elements": []})
        return _Response(200, bad_json=True)

    class _Session:
        def get(self, url, params=None, timeout=None):
            return fake_get(url, params=params, timeout=timeout)

    monkeypatch.setattr(osm_extract, "requests_session_with_retries", lambda *a, **k: _Session())
    monkeypatch.setattr(osm_extract.time, "sleep", lambda _: None)

    ok = osm_extract.query_overpass("0,0,1,1", {"waterway": "river"}, "http://dummy")
    bad = osm_extract.query_overpass("0,0,1,1", {"waterway": "river"}, "http://dummy")

    assert ok == {"elements": []}
    assert bad is None


def test_elements_to_gdf_splits_line_and_polygon_content():
    data = {
        "elements": [
            {"type": "node", "id": 1, "lon": 0.0, "lat": 0.0},
            {"type": "node", "id": 2, "lon": 1.0, "lat": 0.0},
            {"type": "node", "id": 3, "lon": 1.0, "lat": 1.0},
            {"type": "node", "id": 4, "lon": 0.0, "lat": 1.0},
            {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"waterway": "river"}},
            {"type": "way", "id": 11, "nodes": [1, 2, 3, 4, 1], "tags": {"landuse": "industrial"}},
        ]
    }

    lines, polys = osm_extract.elements_to_gdf(data)

    assert len(lines) == 1
    assert len(polys) == 1
    assert lines.geometry.iloc[0].geom_type == "LineString"
    assert polys.geometry.iloc[0].geom_type == "Polygon"


def test_find_bbox_handles_invalid_and_empty_geometries():
    assert osm_extract.find_bbox(None) is None

    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    bbox = osm_extract.find_bbox(invalid)
    assert isinstance(bbox, str)


def test_create_tasks_batches_and_rotates_urls(monkeypatch):
    monkeypatch.setattr(osm_extract, "urls", ["u1", "u2", "u3"])
    gdf = gpd.GeoDataFrame({"idx": [1, 2, 3, 4], "bbox": ["a", "b", "c", "d"]}, geometry=[Point(0, 0)] * 4)

    batches = list(osm_extract.create_tasks(gdf, batch_size=2))

    assert len(batches) == 2
    assert batches[0][0] == ("a", 1, "u1")
    assert batches[0][1] == ("b", 2, "u2")


def test_row_operation_skips_none_bbox_and_writes_outputs(monkeypatch, tmp_path):
    captured = {"writes": 0}

    monkeypatch.setattr(osm_extract, "query_overpass", lambda *args, **kwargs: {"elements": []})
    monkeypatch.setattr(
        osm_extract,
        "elements_to_gdf",
        lambda data: (
            gpd.GeoDataFrame({"x": [1], "geometry": [LineString([(0, 0), (1, 1)])]}, geometry="geometry", crs="EPSG:4326"),
            gpd.GeoDataFrame({"y": [1], "geometry": [Polygon([(0, 0), (1, 0), (0, 1), (0, 0)])]}, geometry="geometry", crs="EPSG:4326"),
        ),
    )
    monkeypatch.setattr(osm_extract, "clean_columns", lambda gdf: gdf)
    monkeypatch.setattr(osm_extract, "ensure_output_dir_for_file", lambda path: None)

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, *args, **kwargs):
        captured["writes"] += 1

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        osm_extract.row_operation(None, 7, "url", str(tmp_path))
        osm_extract.row_operation("0,0,1,1", 8, "url", str(tmp_path))
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["writes"] == 2


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
