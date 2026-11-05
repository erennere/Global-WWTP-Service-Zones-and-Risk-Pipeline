from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon
from shapely import to_wkb

from src.annotation_scripts import NEW_03_WASTEWATERJOIN_GEOJSON as ww_join
from src import download_pop


pytestmark = pytest.mark.unit


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


def test_download_pop_small_helpers_and_finders(tmp_path):
    assert download_pop.extract_first_wildcard("abc_123", r"abc_(\d+)") == "123"
    assert download_pop.try_extract_country("population_deu.geotiff.zip", [r"population_([a-z]{3})"]) == "deu"

    urls = {}
    download_pop.add_country_url(urls, "deu", "u1")
    download_pop.add_country_url(urls, "deu", "u2")
    assert urls["deu"] == ["u1", "u2"]

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.tif").write_text("x", encoding="utf-8")
    (nested / "b.csv").write_text("x", encoding="utf-8")

    tif_files = download_pop.find_type(str(tmp_path), ".tif")
    assert len(tif_files) == 1

    files, is_tif = download_pop.find_files(str(tmp_path))
    assert is_tif is True
    assert len(files) == 1


def test_download_file_and_unzip_dispatch(monkeypatch, tmp_path):
    class _Resp:
        headers = {"Content-Length": "10"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"abc"

    class _Session:
        def get(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(download_pop, "requests_session_with_retries", lambda *a, **k: _Session())
    out = tmp_path / "f.bin"
    assert download_pop.download_file("http://x", str(out)) is True
    assert out.exists()

    monkeypatch.setattr(download_pop, "download_save_and_unzip_pop", lambda url, country, data_dir: str(tmp_path) if "ok" in url else None)
    result = download_pop.download_save_and_unzip_pops({"deu": ["bad", "ok"]}, "deu", data_dir=str(tmp_path))
    assert result == str(tmp_path)


def test_get_urls_contains_multiple_years():
    urls = download_pop.get_urls()
    key = next(iter(urls.keys()))
    assert len(urls[key]) >= 10
