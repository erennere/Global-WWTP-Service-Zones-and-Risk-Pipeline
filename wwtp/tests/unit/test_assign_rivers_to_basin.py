"""Tests for assign_rivers_to_basin (moved from the coverage-campaign files)."""
import sys
import geopandas as gpd
import pandas as pd
import os
import pytest
from shapely.geometry import LineString, Polygon
from wwtp.pop_at_risk_river_calculations import assign_rivers_to_basin

pytestmark = pytest.mark.unit


def test_extract_first_digit_handles_strings_and_missing_values():
    frame = pd.DataFrame({"HYRIV_ID": [" 123", "9", None]})

    result = assign_rivers_to_basin.extract_first_digit(frame.copy(), "HYRIV_ID", "continent")

    assert result["continent"].iloc[0] == "1"
    assert result["continent"].iloc[1] == "9"
    assert pd.isna(result["continent"].iloc[2])


def test_extract_first_digit_requires_source_column():
    frame = pd.DataFrame({"other": ["123"]})

    with pytest.raises(KeyError, match="source column"):
        assign_rivers_to_basin.extract_first_digit(frame.copy(), "HYRIV_ID", "continent")


def test_assign_hybas_id_by_length_handles_empty_inputs(tiny_watershed_gdf):
    lines = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    out = assign_rivers_to_basin.assign_hybas_id_by_length(lines, tiny_watershed_gdf)

    assert out.empty


def test_assign_hybas_id_by_length_picks_longest_intersection_for_multi_match():
    polygons = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [1, 2],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(1.5, 0), (4, 0), (4, 2), (1.5, 2)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    lines = gpd.GeoDataFrame(
        {
            "river": ["single", "multi"],
            "geometry": [
                LineString([(0.1, 1), (1.0, 1)]),
                LineString([(0.2, 1), (2.2, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = assign_rivers_to_basin.assign_hybas_id_by_length(lines.copy(), polygons, id_col="HYBAS_ID")

    assert result["HYBAS_ID"].tolist() == [1, 1]


def test_orchestrate_intersections_discards_unmatched_regions_and_combines_results(monkeypatch):
    hybas = gpd.GeoDataFrame(
        {
            "HYBAS_ID": ["1A"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {
            "HYRIV_ID": ["1x", "2y"],
            "geometry": [
                LineString([(0, 0.5), (1, 0.5)]),
                LineString([(2, 0.5), (3, 0.5)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _Future(fn(*args, **kwargs))

    monkeypatch.setattr(assign_rivers_to_basin, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(assign_rivers_to_basin, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(assign_rivers_to_basin, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(
        assign_rivers_to_basin,
        "assign_hybas_id_by_length",
        lambda r_chunk, h_chunk, col: r_chunk.assign(**{col: [h_chunk.iloc[0][col]] * len(r_chunk)}),
    )

    out = assign_rivers_to_basin.orchestrate_intersections(
        hybas,
        rivers,
        hybas_col="HYBAS_ID",
        hyshed_col="HYRIV_ID",
        new_col="continent",
        max_workers=1,
    )

    # Region "2" has no basin polygon, so its rivers are discarded rather than
    # emitted with an empty HYBAS_ID.
    assert out["HYRIV_ID"].tolist() == ["1x"]
    assert out["HYBAS_ID"].iloc[0] == "1A"


def test_orchestrate_intersections_raises_when_every_region_fails(monkeypatch):
    hybas = gpd.GeoDataFrame(
        {"HYBAS_ID": ["1A"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {"HYRIV_ID": ["1x"], "geometry": [LineString([(0, 0.5), (1, 0.5)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    class _Future:
        def result(self):
            raise RuntimeError("worker exploded")

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _Future()

    monkeypatch.setattr(assign_rivers_to_basin, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(assign_rivers_to_basin, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(assign_rivers_to_basin, "tqdm", lambda iterable, **kwargs: iterable)

    with pytest.raises(RuntimeError, match="produced no assigned rivers"):
        assign_rivers_to_basin.orchestrate_intersections(
            hybas,
            rivers,
            hybas_col="HYBAS_ID",
            hyshed_col="HYRIV_ID",
            new_col="continent",
            max_workers=1,
        )


def test_assign_rivers_orchestrate_intersections_accepts_leading_n_region(monkeypatch):
    hybas = gpd.GeoDataFrame(
        {
            "HYBAS_ID": ["nA"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {
            "HYRIV_ID": ["n1"],
            "geometry": [LineString([(0, 0.5), (1, 0.5)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _Future(fn(*args, **kwargs))

    monkeypatch.setattr(assign_rivers_to_basin, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(assign_rivers_to_basin, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(assign_rivers_to_basin, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(
        assign_rivers_to_basin,
        "assign_hybas_id_by_length",
        lambda r_chunk, h_chunk, col: r_chunk.assign(**{col: [h_chunk.iloc[0][col]] * len(r_chunk)}),
    )

    out = assign_rivers_to_basin.orchestrate_intersections(
        hybas,
        rivers,
        hybas_col="HYBAS_ID",
        hyshed_col="HYRIV_ID",
        new_col="continent",
        max_workers=1,
    )

    assert out["HYBAS_ID"].iloc[0] == "nA"


def test_assign_rivers_orchestrate_intersections_rejects_non_positive_workers():
    hybas = gpd.GeoDataFrame(
        {"HYBAS_ID": ["1A"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {"HYRIV_ID": ["1x"], "geometry": [LineString([(0, 0.5), (1, 0.5)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="max_workers"):
        assign_rivers_to_basin.orchestrate_intersections(
            hybas,
            rivers,
            hybas_col="HYBAS_ID",
            hyshed_col="HYRIV_ID",
            new_col="continent",
            max_workers=0,
        )


def test_assign_hybas_id_by_length_requires_id_column():
    lines = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    polygons = gpd.GeoDataFrame(
        {"NOT_HYBAS": [1], "geometry": [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(KeyError, match="HYBAS_ID"):
        assign_rivers_to_basin.assign_hybas_id_by_length(lines, polygons, id_col="HYBAS_ID")


def test_assign_rivers_main_uses_default_workers_for_non_digit_argv(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "watershed": str(tmp_path / "watershed.gpkg"),
            "rivershed": str(tmp_path / "rivers.gpkg"),
            "rivershed_output_path": str(tmp_path / "out.gpkg"),
        }
    }
    captured = {}
    gdf = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(sys, "argv", ["assign_rivers_to_basin.py"])
    monkeypatch.setattr(os, "chdir", lambda path: None)
    monkeypatch.setattr(assign_rivers_to_basin, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(assign_rivers_to_basin, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(assign_rivers_to_basin.gpd, "read_file", lambda path: gdf.copy())
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: captured.setdefault("ensured", path))

    def fake_orchestrate(hybas_gdf, rivers_gdf, hybas_col, hyshed_col, new_col, max_workers=2):
        captured["workers"] = max_workers
        return rivers_gdf

    monkeypatch.setattr(
        assign_rivers_to_basin,
        "orchestrate_intersections",
        fake_orchestrate,
    )

    original_to_file = gpd.GeoDataFrame.to_file
    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, path, driver=None, index=None, **kwargs: captured.setdefault("written", path))
        assign_rivers_to_basin.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["workers"] == 2
    assert captured["ensured"] == cfg["paths"]["rivershed_output_path"]
    assert captured["written"] == cfg["paths"]["rivershed_output_path"]
