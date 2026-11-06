"""Tests for annotation_wastewater_join (moved from the coverage-campaign files)."""
import geopandas as gpd
import pandas as pd
import random
import pytest
from shapely.geometry import Point, Polygon
from shapely import to_wkb
from wwtp.annotation_scripts import NEW_03_WASTEWATERJOIN_GEOJSON as ww_join
from wwtp import download_pop
import concurrent.futures
import os
import runpy
from pathlib import Path
import wwtp.create_voronoi as create_voronoi
import wwtp.starter as starter
from wwtp.annotation_scripts import NEW_03_WASTEWATERJOIN_GEOJSON as ww3
import argparse
import sys
import types
import duckdb
import numpy as np
from shapely.geometry import LineString, Point, box
from wwtp import download_pop as dp
from wwtp.annotation_scripts import NEW_01_GENERATEGRIDS as n1
from wwtp.annotation_scripts import NEW_02_EXTRACTOSMDATAFULL_GEOJSON as n2
from wwtp.annotation_scripts import download_bing_annotate as dba
from wwtp.figures_scripts import piechart_figure as pf
from wwtp import starter
from wwtp.pop_at_risk_river_calculations import create_rasters as cr
from wwtp.pop_at_risk_river_calculations import find_intersection_river as fir
from wwtp.pop_at_risk_river_calculations import impact_polygons_pop as ipp
from wwtp.industrial_analysis import download_and_vectorize as dv

pytestmark = pytest.mark.unit


class _Future:
    def __init__(self, fn=None, args=(), kwargs=None, value=None):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs or {}
        self._value = value

    def result(self):
        if self._fn is None:
            return self._value
        return self._fn(*self._args, **self._kwargs)

class _Executor:
    def __init__(self, max_workers=None, eager_results=True):
        self.max_workers = max_workers
        self.eager_results = eager_results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        if self.eager_results:
            return _Future(fn, args, kwargs)
        return _Future(value=None)


def test_new03_geometry_helpers_cover_core_paths():
    bad = ww_join.from_wkb_modified(b"not_wkb")
    assert bad is None

    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]), Point()]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    out = ww_join.compute_centroids(gdf)
    assert out["centroid"].iloc[0] is not None

    points = [Point(0, 0), Point(0.01, 0), Point(2, 2), Point()]
    tree = ww_join.build_spatial_index(points)
    clusters = ww_join.cluster_points(points, tree, distance_threshold=0.1)
    assert len(clusters) >= 2

    src = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]),
                Polygon([(0.2, 0), (1.2, 0), (0.2, 1), (0.2, 0)]),
            ]
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    bboxes = ww_join.clusters_to_bboxes(src, [{0, 1}], label="wastewater")
    assert len(bboxes) == 1
    assert bboxes["man_name"].iloc[0] == "wastewater"


def test_new03_load_geodata_and_column_sanitize(monkeypatch):
    df = pd.DataFrame({"geom": [to_wkb(Point(0, 0))], "A": [1]})
    monkeypatch.setattr(ww_join.pd, "read_parquet", lambda path: df.copy())

    gdf = ww_join.load_geodata("dummy.parquet")
    assert "geometry" in gdf.columns
    assert gdf.geometry.iloc[0].geom_type == "Point"

    mixed = gpd.GeoDataFrame({"A": [1], "a": [2], "geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")
    clean = ww_join.sanitize_gdf_columns(mixed)
    assert list(clean.columns).count("a") == 1


def test_new03_duckdb_schema_helpers(monkeypatch):
    class _Conn:
        def execute(self, sql):
            return self

        def df(self):
            return pd.DataFrame({"name": ["geometry", "NAME"], "type": ["BLOB", "VARCHAR"]})

        def close(self):
            return None

    cols, types = ww_join.get_parquet_schema_info(_Conn(), "tmp")
    assert cols == ["geometry", "NAME"]
    assert types["geometry"] == "BLOB"

    assert ww_join.build_cast_expr("x", "TEXT").startswith("CAST")
    assert ww_join.build_cast_expr("x", "INTEGER") == '"x"'

    monkeypatch.setattr(ww_join.duckdb, "connect", lambda *args, **kwargs: _Conn())
    monkeypatch.setattr(ww_join, "get_parquet_schema_info", lambda conn, tbl: (["col1"], {"col1": "VARCHAR"}))

    pf, grid, names, ctype = ww_join.discover_parquet_schema("idx_123_demo.parquet")
    assert pf.endswith(".parquet")
    assert grid == "123"
    assert names == ["col1"]


def test_new03_main_clusters_output(monkeypatch):
    input_gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]), Polygon([(3, 3), (4, 3), (3, 4), (3, 3)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    captured = {}
    monkeypatch.setattr(ww_join, "load_geodata", lambda path: input_gdf.copy())
    monkeypatch.setattr(ww_join, "write_geodata", lambda gdf, path, driver="GeoJSON": captured.setdefault("rows", len(gdf)))

    ww_join.main("in.parquet", "out.geojson", distance_threshold=0.1)
    assert captured["rows"] == 2




def test_write_geodata_ensures_parent_and_writes(monkeypatch, tmp_path):
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")
    captured = {}

    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: captured.setdefault("ensured", path))
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, path, driver="GeoJSON": captured.setdefault("written", (path, driver)))

    ww3.write_geodata(gdf, str(tmp_path / "out.geojson"))

    assert captured["ensured"].endswith("out.geojson")
    assert captured["written"][1] == "GeoJSON"


def test_cluster_points_skips_empty_neighbor_candidates():
    points = [Point(0, 0), Point()]

    class _Tree:
        def query(self, buffered, predicate=None):
            return [1]

    clusters = ww3.cluster_points(points, _Tree(), distance_threshold=1.0)

    assert clusters == [{0}]


def test_convert_geojson_to_parquet_success_path(monkeypatch, tmp_path):
    captured = {"queries": []}

    class _Conn:
        def execute(self, query):
            captured["queries"].append(str(query))
            return self

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(ww3.os.path, "exists", lambda path: False)
    monkeypatch.setattr(ww3, "ensure_output_dir_for_file", lambda path: captured.setdefault("ensured", path))
    monkeypatch.setattr(ww3.duckdb, "connect", lambda *a, **k: _Conn())

    result = ww3.convert_geojson_to_parquet("idx_1_polygons.geojson", str(tmp_path), overwrite=False)

    assert result.endswith("idx_1_polygons.parquet")
    assert captured["ensured"].endswith("idx_1_polygons.parquet")
    assert captured["closed"] is True
    assert any("ST_Read('idx_1_polygons.geojson')" in query for query in captured["queries"])


def test_convert_geojson_to_parquet_failure_and_existing_skip(monkeypatch, tmp_path):
    class _FailConn:
        def execute(self, query):
            raise RuntimeError("duckdb failed")

        def close(self):
            return None

    monkeypatch.setattr(ww3.os.path, "exists", lambda path: False)
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(ww3.duckdb, "connect", lambda *a, **k: _FailConn())

    assert ww3.convert_geojson_to_parquet("idx_2_polygons.geojson", str(tmp_path), overwrite=False) is None

    monkeypatch.setattr(ww3.os.path, "exists", lambda path: True)
    monkeypatch.setattr(ww3.duckdb, "connect", lambda *a, **k: pytest.fail("duckdb.connect should not be called"))

    skipped = ww3.convert_geojson_to_parquet("idx_3_polygons.geojson", str(tmp_path), overwrite=False)

    assert skipped.endswith("idx_3_polygons.parquet")


def test_parallel_convert_geojsons_handles_future_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(ww3, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(ww3, "as_completed", lambda futures: list(futures))

    def fake_convert(path, temp_parquet_dir, overwrite=False):
        if "bad" in path:
            raise RuntimeError("conversion failed")
        return str(Path(temp_parquet_dir) / (Path(path).stem + ".parquet"))

    monkeypatch.setattr(ww3, "convert_geojson_to_parquet", fake_convert)

    result = ww3.parallel_convert_geojsons(["good.geojson", "bad.geojson"], str(tmp_path), max_workers=1)

    assert result == [str(tmp_path / "good.parquet")]


def test_merge_parquets_sql_ignores_schema_discovery_failures(monkeypatch):
    class _Conn:
        def __init__(self):
            self.commands = []

        def execute(self, query):
            self.commands.append(str(query))
            return self

    def fake_discover(parquet_file):
        if parquet_file == "bad.parquet":
            raise RuntimeError("schema failure")
        return parquet_file, "1", ["Geom", "Name"], {"Geom": "BLOB", "Name": "TEXT"}

    monkeypatch.setattr(ww3, "ThreadPoolExecutor", _Executor)
    monkeypatch.setattr(ww3, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(ww3, "discover_parquet_schema", fake_discover)

    conn = _Conn()
    with pytest.raises(KeyError, match="bad.parquet"):
        ww3.merge_parquets_sql(conn, ["good.parquet", "bad.parquet"], max_workers=1, insert_batch_size=1)

    assert any("CREATE TABLE dataset" in command for command in conn.commands)
    assert any("INSERT INTO dataset" in command for command in conn.commands)


def test_new03_merge_parquets_sql_happy_path(monkeypatch):
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

    class _Conn:
        def __init__(self):
            self.sql = []

        def execute(self, query):
            self.sql.append(str(query))
            return self

    def _discover(pf):
        if pf.endswith("a.parquet"):
            return pf, "001", ["geom", "name"], {"geom": "BLOB", "name": "VARCHAR"}
        return pf, "002", ["geom", "power"], {"geom": "BLOB", "power": "TEXT"}

    monkeypatch.setattr(ww3, "ThreadPoolExecutor", _Exec)
    monkeypatch.setattr(ww3, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(ww3, "discover_parquet_schema", _discover)

    conn = _Conn()
    mapping = ww3.merge_parquets_sql(conn, ["a.parquet", "b.parquet"], max_workers=2, insert_batch_size=1)

    assert mapping == {"a.parquet": "001", "b.parquet": "002"}
    assert any("CREATE TABLE dataset" in q for q in conn.sql)
    assert sum(1 for q in conn.sql if "INSERT INTO dataset" in q) == 2


def test_new03_merge_bboxes_sql_branches(monkeypatch, tmp_path):
    out = tmp_path / "merged.parquet"

    # Branch: no matching input files.
    monkeypatch.setattr(ww3.glob, "glob", lambda pattern: [])
    ww3.merge_bboxes_sql(str(tmp_path), "*_polygons.geojson", str(out))

    # Branch: normal merge path with cleanup.
    inputs = [str(tmp_path / "idx_1_polygons.geojson")]
    monkeypatch.setattr(ww3.glob, "glob", lambda pattern: inputs)

    class _Conn:
        def __init__(self):
            self.commands = []

        def execute(self, query):
            self.commands.append(str(query))
            return self

        def close(self):
            return None

    def _connect(*_args, **_kwargs):
        return _Conn()

    monkeypatch.setattr(ww3.duckdb, "connect", _connect)
    monkeypatch.setattr(ww3, "parallel_convert_geojsons", lambda *args, **kwargs: [str(tmp_path / "idx_1_polygons.parquet")])
    monkeypatch.setattr(ww3, "merge_parquets_sql", lambda *args, **kwargs: {"x": "1"})
    monkeypatch.setattr("wwtp.utils.ensure_output_dir_for_file", lambda path: None)

    ww3.merge_bboxes_sql(
        polygons_dir=str(tmp_path),
        prototype="*_polygons.geojson",
        output_filepath=str(out),
        temp_parquet_dir=str(tmp_path / "tmp_parquets"),
        max_workers=1,
        insert_batch_size=1,
        duckdb_threads=1,
        overwrite=True,
    )

    # The scratch database is owned by utils.duckdb_connection; this module must
    # neither name nor delete one.
    assert not list(tmp_path.glob("temp_*.db"))


def test_new03_load_geodata_missing_geometry_column_raises(monkeypatch, tmp_path):
    # FIX [C-1]: missing geometry/geom columns should raise a clear contract error.
    parquet_path = tmp_path / "no_geom.parquet"
    pd_no_geom = pd.DataFrame({"id": [1], "name": ["x"]})

    monkeypatch.setattr(ww3.pd, "read_parquet", lambda _path: pd_no_geom.copy())
    with pytest.raises(ValueError, match="geometry|geom"):
        ww3.load_geodata(str(parquet_path))


def test_new03_merge_bboxes_sql_empty_conversion_short_circuits(monkeypatch, tmp_path):
    # FIX [F-2]: empty parquet conversion output should short-circuit before merge/export.
    out = tmp_path / "merged.parquet"
    inputs = [str(tmp_path / "idx_1_polygons.geojson")]
    monkeypatch.setattr(ww3.glob, "glob", lambda pattern: inputs)

    class _Conn:
        def execute(self, _query):
            return self

        def close(self):
            return None

    monkeypatch.setattr(random, "randint", lambda a, b: 456)
    monkeypatch.setattr(ww3.duckdb, "connect", lambda *a, **k: _Conn())
    monkeypatch.setattr(ww3, "parallel_convert_geojsons", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        ww3,
        "merge_parquets_sql",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("merge_parquets_sql should not be called")),
    )

    ww3.merge_bboxes_sql(
        polygons_dir=str(tmp_path),
        prototype="*_polygons.geojson",
        output_filepath=str(out),
        temp_parquet_dir=str(tmp_path / "tmp_parquets"),
        max_workers=1,
        insert_batch_size=1,
        duckdb_threads=1,
        overwrite=True,
    )
