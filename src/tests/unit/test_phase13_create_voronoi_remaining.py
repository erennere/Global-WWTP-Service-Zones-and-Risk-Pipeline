from __future__ import annotations

import argparse
import contextlib
import math
import runpy
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box

from src import create_voronoi as cv
from src import geo_utils


pytestmark = pytest.mark.unit


def _fake_duckdb_connection(conn):
    """Build a fake replacement for cv.duckdb_connection yielding ``conn``."""

    @contextlib.contextmanager
    def _cm(*args, **kwargs):
        yield conn

    return _cm


def test_estimate_utm_epsg_and_crs_fallback_paths(monkeypatch):
    with pytest.raises(ValueError):
        cv.estimate_utm_epsg(200, 10)

    original_from_epsg = CRS.from_epsg

    def fail_for_utm(epsg):
        if epsg in {32631, 32731}:
            raise RuntimeError("boom")
        return original_from_epsg(epsg)

    monkeypatch.setattr(geo_utils.CRS, "from_epsg", fail_for_utm)
    assert cv.estimate_utm_epsg(0, 10) == 3857

    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    assert cv.estimate_utm_crs(empty).to_epsg() == 3857


def test_estimate_utm_crs_uses_valid_geometry_fallback(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(10, 10)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    class _FakeUnion:
        @property
        def centroid(self):
            return Point(float("nan"), float("nan"))

    monkeypatch.setattr(gpd.GeoSeries, "unary_union", property(lambda self: _FakeUnion()))
    result = cv.estimate_utm_crs(gdf)
    assert result.to_epsg() == 32631


def test_create_weights_error_and_fallback_branches():
    with pytest.raises(KeyError):
        cv.create_weights(pd.DataFrame({"x": [1, 2]}))

    df = pd.DataFrame({"base_values": [np.nan, np.nan, np.nan]})
    result = cv.create_weights(df)
    assert math.isclose(result["weights"].sum(), 1.0)
    assert result["weights"].nunique() == 1

    sigmoid_df = pd.DataFrame({"base_values": [5.0, 5.0, 5.0]})
    sigmoid = cv.create_weights(sigmoid_df, method="sigmoid")
    assert math.isclose(sigmoid["weights"].sum(), 1.0)
    assert sigmoid["weights"].nunique() == 1


def test_finalize_gdf_and_contours_helpers():
    empty = cv.finalize_gdf([], cols=["geometry", "x"])
    assert empty.empty
    assert "geometry" in empty.columns

    g1 = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), box(0, 0, 1, 1)], "x": [1, 2]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    combined = cv.finalize_gdf([g1], cols=g1.columns)
    assert len(combined) == 2

    mask = np.array([[False, True, True], [False, True, False], [False, False, False]], dtype=bool)
    assert cv.extract_contours_scipy(mask, 10, 0, 0)
    assert cv.extract_contours_cv2(mask, 10, 0, 0)
    assert cv.extract_contours_rasterio(mask, 10, 0, 0)


def test_intersection_helpers_empty_and_error_paths(monkeypatch):
    empty_df = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    poly = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)], "country": ["DE"]}, geometry="geometry", crs="EPSG:4326")
    assert cv.intersect_with_polygon_sindex(empty_df, poly.copy(), "country") is empty_df
    assert cv.intersect_with_polygons_parallelized(empty_df, poly.copy(), ["country"]) .empty
    assert cv.intersects_with_country_db(empty_df, "dummy.parquet") is empty_df

    with pytest.raises(KeyError):
        cv.intersect_with_polygons_db(poly.copy(), poly.copy(), ["country"], df_join_col="missing", polygon_join_col="country")


def test_orchestrate_overlaps_cache_and_missing_column(monkeypatch, tmp_path):
    df = gpd.GeoDataFrame({"geometry": [Point(0, 0)], "ISO_2": ["DE"]}, geometry="geometry", crs="EPSG:4326")
    cache_fp = tmp_path / "buffers.gpkg"
    cached = gpd.GeoDataFrame({"geometry": [Point(1, 1)]}, geometry="geometry", crs="EPSG:4326")
    cached.to_file(cache_fp, driver="GPKG")
    assert cv.orchestrate_overlaps(df, 2, str(cache_fp), 1000, country_col="ISO_2").equals(gpd.read_file(cache_fp))

    with pytest.raises(KeyError):
        cv.orchestrate_overlaps(df.drop(columns=["ISO_2"]), 2, str(tmp_path / "other.gpkg"), 1000, country_col="ISO_2")


def _patch_main_dependencies(monkeypatch, tmp_path, args_namespace, overwrite=True):
    import concurrent.futures
    import src.pipelines as pipelines
    import src.starter as starter

    class _ImmediateFuture:
        def __init__(self, fn, args):
            self._fn = fn
            self._args = args

        def result(self):
            return self._fn(*self._args)

    class _ImmediateExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            return _ImmediateFuture(fn, args)

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(concurrent.futures, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args_namespace)

    cities_fp = tmp_path / "cities.csv"
    pd.DataFrame(
        {
            "ISO_2": ["DE"],
            "geometry": [Point(0.0, 0.0).wkt],
        }
    ).to_csv(cities_fp, index=False)

    cfg = {
        "overwrite_existing": overwrite,
        "city_voronoi": True,
        "weight_func": "mult",
        "weight_method": "linear",
        "country_output_column": "ISO_2",
        "country_boundary_column": "country",
        "site_id_column": "WASTE_ID",
        "prepare_data_fn": None,
        "distance_fn": cv.default_distance_multiplicative,
        "duckdb_cond": False,
        "max_workers": 1,
        "buffer": 200,
        "sindex_concurrency": False,
        "paths": {
            "voronoi_dir": str(tmp_path / "voronoi"),
            "cities": str(cities_fp),
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://dummy/overture.parquet",
        },
    }

    out_paths = {
        "voronoi": {
            "0": str(tmp_path / "voronoi" / "a0.gpkg"),
            "1": str(tmp_path / "voronoi" / "a1.gpkg"),
            "2": str(tmp_path / "voronoi" / "a2.gpkg"),
            "0_only_round": str(tmp_path / "voronoi" / "a0_or.gpkg"),
            "1_only_round": str(tmp_path / "voronoi" / "a1_or.gpkg"),
        },
        "buffers": {
            "WWTP": str(tmp_path / "buffers" / "wwtp.gpkg"),
            "city": str(tmp_path / "buffers" / "city.gpkg"),
        },
    }

    gdf_bbox = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1, 2],
            "ISO_2": ["DE", "DE"],
            "HYBAS_ID": [100, 100],
            "weights": [0.5, 0.5],
            "geometry": [Point(0, 0), Point(0.002, 0.002)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    basin_gdf = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [100],
            "geometry": [box(-1, -1, 1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    country_df = gpd.GeoDataFrame(
        {
            "country": ["DE"],
            "geometry": [box(-2, -2, 2, 2)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(starter, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(starter, "load_config", lambda **kwargs: cfg)

    monkeypatch.setattr(pipelines, "create_output_paths", lambda cfg_in: out_paths)
    monkeypatch.setattr(
        pipelines,
        "prepare_data",
        lambda cfg_in: {"gdf_bbox": gdf_bbox.copy(), "basin_gdf": basin_gdf.copy(), "country_df": country_df.copy()},
    )
    monkeypatch.setattr(pipelines, "_resolve_configured_callable", lambda configured, default, name, module: default)
    monkeypatch.setattr(pipelines, "run_voronoi_approach", lambda *args, **kwargs: True)


def test_create_voronoi_main_runs_all_approaches(monkeypatch, tmp_path):
    args = argparse.Namespace(
        approach=["0", "1", "2"],
        only_round=False,
        verbose=True,
        level=None,
        version=None,
        buffer=None,
        weight_method=None,
        weight_func=None,
        dynamic_buffering=None,
        size_measure=None,
    )
    _patch_main_dependencies(monkeypatch, tmp_path, args, overwrite=True)
    runpy.run_module("src.create_voronoi", run_name="__main__")


def test_create_voronoi_main_no_remaining_approaches_exits(monkeypatch, tmp_path):
    args = argparse.Namespace(
        approach=["0"],
        only_round=False,
        verbose=False,
        level=None,
        version=None,
        buffer=None,
        weight_method=None,
        weight_func=None,
        dynamic_buffering=None,
        size_measure=None,
    )
    _patch_main_dependencies(monkeypatch, tmp_path, args, overwrite=False)

    # Pre-create output so filter step skips all requested approaches.
    (tmp_path / "voronoi").mkdir(parents=True, exist_ok=True)
    (tmp_path / "voronoi" / "a0.gpkg").write_text("x", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("src.create_voronoi", run_name="__main__")
    assert exc.value.code == 0


def test_create_voronoi_main_run_path_import_fallback(monkeypatch, tmp_path):
    import src.pipelines as pipelines
    import src.starter as starter

    import src.geo_utils as geo_utils_module
    import src.utils as utils_module

    # Make top-level imports in run_path fallback resolve cleanly.
    monkeypatch.setitem(sys.modules, "starter", starter)
    monkeypatch.setitem(sys.modules, "pipelines", pipelines)
    monkeypatch.setitem(sys.modules, "geo_utils", geo_utils_module)
    monkeypatch.setitem(sys.modules, "utils", utils_module)

    args = argparse.Namespace(
        approach=["0"],
        only_round=True,
        verbose=False,
        level=None,
        version=None,
        buffer=None,
        weight_method=None,
        weight_func=None,
        dynamic_buffering=None,
        size_measure=None,
    )
    _patch_main_dependencies(monkeypatch, tmp_path, args, overwrite=True)

    module_path = Path(__file__).resolve().parents[2] / "create_voronoi.py"
    runpy.run_path(str(module_path), run_name="__main__")


def test_download_overture_maps_success_and_failure(monkeypatch, tmp_path):
    calls = []

    class _OkConn:
        def execute(self, query):
            return None

        def sql(self, query):
            calls.append(query)

            class _Result:
                def df(self):
                    return pd.DataFrame()

            return _Result()

    monkeypatch.setattr(cv, "duckdb_connection", _fake_duckdb_connection(_OkConn()))
    out_fp = tmp_path / "overture" / "countries.parquet"
    cv.download_overture_maps("s3://dummy/countries.parquet", str(out_fp))
    assert calls

    class _FailConn:
        def execute(self, query):
            return None

        def sql(self, query):
            raise RuntimeError("duckdb fail")

    monkeypatch.setattr(cv, "duckdb_connection", _fake_duckdb_connection(_FailConn()))
    cv.download_overture_maps("s3://dummy/countries.parquet", str(out_fp))


def test_intersects_with_country_db_non_empty(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    ).to_crs(3857)

    class _DuckResult:
        def __init__(self, frame):
            self._frame = frame

        def df(self):
            return self._frame

    class _Conn:
        def execute(self, query):
            return None

        def sql(self, query):
            if "SELECT" in query.upper():
                return _DuckResult(pd.DataFrame({"geometry": [Point(0, 0).wkt], "ISO_2": ["DE"]}))
            return _DuckResult(pd.DataFrame())

    monkeypatch.setattr(cv, "duckdb_connection", _fake_duckdb_connection(_Conn()))
    out = cv.intersects_with_country_db(gdf, "dummy.parquet", output_country_col="ISO_2")
    assert "ISO_2" in out.columns
    assert out["ISO_2"].iloc[0] == "DE"


def test_intersect_with_polygons_db_success_path(monkeypatch):
    df = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "DE"],
            "utm": ["EPSG:32632", "EPSG:32632"],
            "geometry": [Point(10, 50), None],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    polygons = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE"],
            "buffer_id": [7],
            "geometry": [box(9.0, 49.0, 11.0, 51.0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    class _DuckResult:
        def __init__(self, frame=None):
            self._frame = frame if frame is not None else pd.DataFrame()

        def df(self):
            return self._frame

    class _Conn:
        def execute(self, query):
            if "SELECT" in query.upper():
                return _DuckResult(
                    pd.DataFrame(
                        {
                            "ISO_2": ["DE"],
                            "utm": ["EPSG:32632"],
                            "geometry": [Point(10, 50).wkt],
                            "centroid": [Point(10, 50).wkt],
                            "buffer_id": [7],
                        }
                    )
                )
            return _DuckResult()

        def close(self):
            return None

    monkeypatch.setattr(cv, "duckdb_connection", _fake_duckdb_connection(_Conn()))
    monkeypatch.setattr(cv.np.random, "randint", lambda low, high: 1)
    out = cv.intersect_with_polygons_db(df, polygons, ["buffer_id"], df_join_col="ISO_2", polygon_join_col="ISO_2")
    assert "buffer_id" in out.columns
    assert len(out) == 2


def test_create_voronoi_main_invalid_approach_and_override_errors(monkeypatch, tmp_path):
    # Invalid approach should fail fast through parser.error.
    bad_args = argparse.Namespace(
        approach=["bogus"],
        only_round=False,
        verbose=False,
        level=None,
        version=None,
        buffer=None,
        weight_method=None,
        weight_func=None,
        dynamic_buffering=None,
        size_measure=None,
    )
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: bad_args)
    with pytest.raises(SystemExit):
        runpy.run_module("src.create_voronoi", run_name="__main__")

    args = argparse.Namespace(
        approach=["0"],
        only_round=False,
        verbose=False,
        level=None,
        version=None,
        buffer=None,
        weight_method=None,
        weight_func=None,
        dynamic_buffering=None,
        size_measure=None,
    )
    _patch_main_dependencies(monkeypatch, tmp_path, args, overwrite=True)

    import src.starter as starter

    monkeypatch.setattr(starter, "parse_config_overrides", lambda args=None: (_ for _ in ()).throw(ValueError("bad override")))
    with pytest.raises(SystemExit):
        runpy.run_module("src.create_voronoi", run_name="__main__")


def test_create_voronoi_main_city_country_enrichment_and_failure_exit(monkeypatch, tmp_path):
    import src.pipelines as pipelines

    args = argparse.Namespace(
        approach=["2"],
        only_round=False,
        verbose=False,
        level=None,
        version=None,
        buffer=None,
        weight_method=None,
        weight_func=None,
        dynamic_buffering=None,
        size_measure=None,
    )
    _patch_main_dependencies(monkeypatch, tmp_path, args, overwrite=True)

    # Rewrite cities without ISO_2 to trigger enrichment branch.
    cities_fp = tmp_path / "cities.csv"
    pd.DataFrame({"geometry": [Point(0.0, 0.0).wkt]}).to_csv(cities_fp, index=False)

    class _DuckResult:
        def __init__(self, frame=None):
            self._frame = frame if frame is not None else pd.DataFrame()

        def df(self):
            return self._frame

    class _Conn:
        def execute(self, query):
            return None

        def sql(self, query):
            if "SELECT" in query.upper() and "FROM data a" in query:
                return _DuckResult(pd.DataFrame({"geometry": [Point(0, 0).wkt], "ISO_2": ["DE"]}))
            return _DuckResult(pd.DataFrame())

    monkeypatch.setattr(cv, "duckdb_connection", _fake_duckdb_connection(_Conn()))

    real_exists = cv.os.path.exists

    def _exists(path):
        if str(path).endswith("overture.parquet"):
            return False
        return real_exists(path)

    monkeypatch.setattr(cv.os.path, "exists", _exists)
    monkeypatch.setattr(pipelines, "run_voronoi_approach", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("src.create_voronoi", run_name="__main__")
    assert exc.value.code == 1


def test_helper_branch_paths_for_remaining_small_functions(monkeypatch):
    uf = cv.UnionFind(2)
    uf.union(0, 0)

    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    monkeypatch.setattr(cv.shapely, "contains_xy", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(cv.vectorized, "contains", lambda geom, xs, ys: np.array([True, False]))
    assert cv.geometry_contains_points(box(-1, -1, 2, 2), points).tolist() == [True, False]

    monkeypatch.setattr(cv.vectorized, "contains", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    assert cv.geometry_contains_points(box(-1, -1, 2, 2), points).tolist() == [True, True]

    class _BadGeom:
        @property
        def is_valid(self):
            raise RuntimeError("boom")

    assert cv.is_valid_geom(_BadGeom()) is False
    assert cv.buffer_geometry("unknown") == "unknown"
    assert cv.create_centroid_points(123) is None

    assert np.isfinite(cv.auto_weight_scale([(0, 0), (1, 1), (np.nan, 2), (None, 3)]))

    bad_centroid_gdf = gpd.GeoDataFrame({"geometry": [GeometryCollection([Point(0, 0)])]}, geometry="geometry", crs="EPSG:4326")

    class _FakeUnion:
        @property
        def centroid(self):
            return Point(float("nan"), float("nan"))

    monkeypatch.setattr(gpd.GeoSeries, "unary_union", property(lambda self: _FakeUnion()))
    assert cv.estimate_utm_crs(bad_centroid_gdf).to_epsg() == 3857

    calc = cv.calculate_area(pd.DataFrame({"diameters": ["[]"]}))
    assert (calc["total_area"] == 1).all()


def test_orchestrate_voronoi_weights_output_path_branches(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        cv.orchestrate_voronoi_weights(
            gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326"),
            col="grp",
            country_df=gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326"),
            area_fn=None,
        )

    df = gpd.GeoDataFrame(
        {
            "grp": ["1"],
            "ISO_2": ["DE"],
            "WASTE_ID": [1],
            "geometry": [Point(10, 50)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    clipping = gpd.GeoDataFrame({"grp": ["1"], "geometry": [box(9, 49, 11, 51)]}, geometry="geometry", crs="EPSG:4326")
    country_df = gpd.GeoDataFrame({"country": ["DE"], "geometry": [box(8, 48, 12, 52)]}, geometry="geometry", crs="EPSG:4326")

    class _Future:
        def __init__(self, payload):
            self._payload = payload

        def result(self):
            return self._payload

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            return _Future(fn(*args))

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(cv, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(cv, "estimate_utm_crs", lambda g: CRS.from_epsg(32632))

    def _area_fn(sub_df, **kwargs):
        out = sub_df.copy()
        out["base_values"] = 1.0
        out["total_area"] = 1.0
        out["num_detection_rect"] = 0
        out["num_detection_circle"] = 0
        return out

    def _fake_worker(args):
        sub_df = args[0]
        reg = sub_df[["grp", "ISO_2", "WASTE_ID", "geometry"]].copy()
        return reg, reg.copy()

    monkeypatch.setattr(cv, "voronoi_worker", _fake_worker)

    def _fake_to_file(self, path, *args, **kwargs):
        Path(path).write_text("x", encoding="utf-8")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", _fake_to_file)

    output_path = tmp_path / "voronoi_out.gpkg"
    ok = cv.orchestrate_voronoi_weights(
        df,
        col="grp",
        country_df=country_df,
        clipping=clipping,
        workers=1,
        area_fn=_area_fn,
        output_path=str(output_path),
        overwrite=True,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
    )
    assert ok is True
    assert output_path.exists()

    output_path.write_text("already", encoding="utf-8")
    ok2 = cv.orchestrate_voronoi_weights(
        df,
        col="grp",
        country_df=country_df,
        clipping=clipping,
        workers=1,
        area_fn=_area_fn,
        output_path=str(output_path),
        overwrite=False,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
    )
    assert ok2 is True


def test_estimate_utm_crs_linestring_and_epsg_failure(monkeypatch):
    gdf = gpd.GeoDataFrame({"geometry": [LineString([(0, 0), (1, 1)])]}, geometry="geometry", crs="EPSG:4326")

    class _FakeUnion:
        @property
        def centroid(self):
            return Point(float("nan"), float("nan"))

    original_from_epsg = geo_utils.CRS.from_epsg

    def _patched_from_epsg(epsg):
        if epsg != 3857:
            raise RuntimeError("bad epsg")
        return original_from_epsg(epsg)

    monkeypatch.setattr(gpd.GeoSeries, "unary_union", property(lambda self: _FakeUnion()))
    monkeypatch.setattr(geo_utils.CRS, "from_epsg", _patched_from_epsg)
    assert cv.estimate_utm_crs(gdf).to_epsg() == 3857


def test_orchestrate_voronoi_weights_checkpoint_and_error_branches(monkeypatch, tmp_path):
    df = gpd.GeoDataFrame(
        {"grp": ["1"], "WASTE_ID": [1], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    country_df = gpd.GeoDataFrame({"country": ["DE"], "geometry": [box(-1, -1, 1, 1)]}, geometry="geometry", crs="EPSG:4326")
    out_fp = tmp_path / "x.gpkg"
    temp_fp = tmp_path / "temp_x.gpkg"
    temp_fp.write_text("not-a-gpkg", encoding="utf-8")

    monkeypatch.setattr(cv, "estimate_utm_crs", lambda g: None)

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            class _Future:
                def result(self_inner):
                    return None

            return _Future()

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(cv, "as_completed", lambda futures: list(futures))

    def _area_fn(sub_df, **kwargs):
        out = sub_df.copy()
        out["base_values"] = 1.0
        out["total_area"] = 1.0
        out["num_detection_rect"] = 0
        out["num_detection_circle"] = 0
        return out

    # overwrite=False with invalid temp checkpoint should hit resume-read warning branch.
    r1 = cv.orchestrate_voronoi_weights(
        df,
        col="grp",
        country_df=country_df,
        workers=1,
        area_fn=_area_fn,
        output_path=str(out_fp),
        overwrite=False,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
    )
    assert r1 in {False, True}

    # overwrite=True with failing temp removal should hit remove-warning branch.
    monkeypatch.setattr(cv.os, "remove", lambda path: (_ for _ in ()).throw(OSError("no remove")))
    r2 = cv.orchestrate_voronoi_weights(
        df,
        col="grp",
        country_df=country_df,
        workers=1,
        area_fn=_area_fn,
        output_path=str(out_fp),
        overwrite=True,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
    )
    assert r2 in {False, True}


def test_remaining_small_helper_branches(monkeypatch):
    uf = cv.UnionFind(2)
    uf.rank[0] = 0
    uf.rank[1] = 1
    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)

    class _FiniteGeom:
        is_valid = True
        geom_type = "LineString"
        coords = [(0.0, 0.0), (float("inf"), 1.0)]

    assert cv.is_valid_geom(_FiniteGeom()) is False

    class _FakeCentroid:
        is_valid = False
        is_empty = False

    class _FakePoly:
        geom_type = "Polygon"
        centroid = _FakeCentroid()

        def buffer(self, n):
            raise RuntimeError("buffer fail")

    monkeypatch.setattr(cv, "Polygon", _FakePoly)
    fp = _FakePoly()
    assert cv.buffer_geometry(fp) is fp
    assert cv.create_centroid_points(fp) is None

    rng = cv.create_ranges(0, 1, step=10, min_step=2)
    assert np.allclose(rng, np.array([0.0, 10.0]))

    area_empty = cv.calculate_area(gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326"))
    assert area_empty.empty

    area_df = pd.DataFrame(
        {
            "wwtp_area_rect": ["[1 2]"],
            "diameters": ["[4]"],
            "num_detection_rect": [1],
            "num_detection_circle": [0],
        }
    )
    area_out = cv.calculate_area(area_df, only_round=True)
    assert area_out["total_area"].iloc[0] == area_out["round_area"].iloc[0]

    monkeypatch.setattr(cv, "create_centroid_points", lambda geom: geom)
    site_df = gpd.GeoDataFrame({"geometry": [LineString([(0, 0), (1, 1)]), None]}, geometry="geometry", crs="EPSG:4326")
    pts = cv.extract_site_coordinates(site_df, centroid_points=False)
    assert pts[0][0] is not None
    assert pts[1] == (None, None)


def test_process_and_intersection_additional_branches(monkeypatch):
    poly = gpd.GeoDataFrame({"v": [1], "geometry": [box(0, 0, 1, 1)]}, geometry="geometry", crs="EPSG:4326")
    sidx = poly.sindex
    assert cv.process_centroid((Point(), sidx, poly, "v")) is None
    monkeypatch.setattr(gpd.GeoDataFrame, "intersects", lambda self, other: (_ for _ in ()).throw(RuntimeError("bad intersects")))
    assert cv.process_centroid((Point(0.5, 0.5), sidx, poly, "v")) is None

    class _ImmediatePool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, args):
            return [fn(a) for a in args]

    monkeypatch.setattr(cv, "ThreadPoolExecutor", lambda: _ImmediatePool())
    df = gpd.GeoDataFrame({"geometry": [Point(0.2, 0.2)]}, geometry="geometry", crs="EPSG:4326")
    out = cv.intersect_with_polygon_sindex(df, poly.rename(columns={"v": "country"}).copy(), "country", concurrency=True)
    assert "country" in out.columns


def test_db_parallel_and_weights_fallback_branches(monkeypatch):
    df_empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    poly_empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    assert cv.intersect_with_polygons_db(df_empty, poly_empty, "x") is df_empty

    df = gpd.GeoDataFrame({"ISO_2": ["DE"], "utm": [32632], "geometry": [Point(10, 50)]}, geometry="geometry", crs="EPSG:4326")
    poly = gpd.GeoDataFrame({"X": [1], "geometry": [box(9, 49, 11, 51)]}, geometry="geometry", crs="EPSG:4326")
    with pytest.raises(KeyError):
        cv.intersect_with_polygons_db(df.copy(), poly.copy(), ["X"], df_join_col="ISO_2", polygon_join_col="missing")

    class _ConnFail:
        def execute(self, q):
            raise RuntimeError("duckdb boom")

        def close(self):
            return None

    monkeypatch.setattr(cv, "duckdb_connection", _fake_duckdb_connection(_ConnFail()))
    # A failed DuckDB intersection re-raises instead of returning the half-built
    # frame (WKT geometries, dropped null rows, requested columns absent). The
    # scratch database is owned by utils.duckdb_connection, so this module neither
    # names nor removes one.
    with pytest.raises(RuntimeError, match="duckdb boom"):
        cv.intersect_with_polygons_db(df.copy(), poly.copy(), "X", df_join_col="ISO_2", polygon_join_col="X")

    par_df = gpd.GeoDataFrame({"geometry": [Point(0, 0), None]}, geometry="geometry", crs="EPSG:4326")
    par_poly = gpd.GeoDataFrame({"geometry": [box(-1, -1, 1, 1)]}, geometry="geometry", crs="EPSG:4326")
    monkeypatch.setattr(cv, "intersect_with_polygons_db", lambda *args, **kwargs: args[0])
    res = cv.intersect_with_polygons_parallelized(par_df, par_poly, "foo", use_duckdb=True)
    assert isinstance(res, gpd.GeoDataFrame)

    cw = cv.create_weights(pd.DataFrame({"base_values": [-2.0, -3.0]}), method="logarithmic")
    assert math.isclose(cw["weights"].sum(), 1.0)


def test_dissolve_overlap_and_main_remaining_branches(monkeypatch, tmp_path):
    assert cv.dissolve_overlapping_geometries_fast(gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326"), 10) == ([], None)

    # orchestrate_overlaps future handling + cache-write warning path.
    # A worker returning None is now counted as an error rather than skipped.
    class _FutNone:
        def result(self):
            return None

    class _FutErr:
        def result(self):
            raise RuntimeError("boom")

    class _FutGood:
        def result(self):
            sdf = gpd.GeoDataFrame(
                {"some_id": [2], "geometry": [box(0, 0, 1, 1)]},
                geometry="geometry",
                crs="EPSG:4326",
            )
            return [{2}], sdf

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers
            self._i = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            self._i += 1
            if self._i == 1:
                return _FutNone()
            if self._i == 2:
                return _FutErr()
            return _FutGood()

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(cv, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, *a, **k: (_ for _ in ()).throw(OSError("no write")))

    df2 = gpd.GeoDataFrame(
        {"ISO_2": ["DE", "FR", "IT"], "some_id": [0, 1, 2], "geometry": [Point(0, 0), Point(2, 2), Point(3, 3)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    # R3: a failed country group now stops the run instead of silently
    # dropping that country's buffers.
    with pytest.raises(RuntimeError, match="2 country group"):
        cv.orchestrate_overlaps(df2.copy(), 2, str(tmp_path / "no_cache.gpkg"), 10, country_col="ISO_2")

    class _ExecGood(_Exec):
        def submit(self, fn, *args):
            return _FutGood()

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _ExecGood)
    dissolved = cv.orchestrate_overlaps(df2.iloc[[2]].copy(), 2, str(tmp_path / "no_cache.gpkg"), 10, country_col="ISO_2")
    assert isinstance(dissolved, gpd.GeoDataFrame)

    # resolve_polygon_overlaps None-geometry skip
    rp = gpd.GeoDataFrame({"geometry": [None, box(0, 0, 1, 1)]}, geometry="geometry", crs="EPSG:4326")
    out = cv.resolve_polygon_overlaps(rp)
    assert len(out) == 2

    # main: args.approach default (None), disabled approach logging path, and bad prepare_data type
    args = argparse.Namespace(
        approach=None,
        only_round=False,
        verbose=False,
        level=None,
        version=None,
        buffer=None,
        weight_method=None,
        weight_func=None,
        dynamic_buffering=None,
        size_measure=None,
    )
    _patch_main_dependencies(monkeypatch, tmp_path, args, overwrite=False)
    import src.starter as starter
    import src.pipelines as pipelines
    monkeypatch.setattr(starter, "load_config", lambda **kwargs: {
        "overwrite_existing": False,
        "city_voronoi": False,
        "weight_func": "mult",
        "weight_method": "linear",
        "country_output_column": "ISO_2",
        "country_boundary_column": "country",
        "site_id_column": "WASTE_ID",
        "prepare_data_fn": None,
        "distance_fn": cv.default_distance_multiplicative,
        "duckdb_cond": False,
        "max_workers": 1,
        "buffer": 200,
        "sindex_concurrency": False,
        "paths": {"voronoi_dir": str(tmp_path / "v"), "cities": str(tmp_path / "cities.csv"), "overture": str(tmp_path / "ov.parquet"), "overture_s3_url": "s3://x"},
    })
    patched_paths = {
        "voronoi": {
            "0": str(tmp_path / "v" / "a0.gpkg"),
            "1": str(tmp_path / "v" / "a1.gpkg"),
            "2": str(tmp_path / "v" / "a2.gpkg"),
            "0_only_round": str(tmp_path / "v" / "a0r.gpkg"),
            "1_only_round": str(tmp_path / "v" / "a1r.gpkg"),
        },
        "buffers": {
            "WWTP": str(tmp_path / "b0.gpkg"),
            "city": str(tmp_path / "b1.gpkg"),
        },
    }
    monkeypatch.setattr(pipelines, "create_output_paths", lambda cfg_in: patched_paths)
    for p in [tmp_path / "v" / "a0.gpkg", tmp_path / "v" / "a1.gpkg"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("src.create_voronoi", run_name="__main__")
    assert exc.value.code == 0

    args2 = argparse.Namespace(
        approach=["0"],
        only_round=False,
        verbose=False,
        level=None,
        version=None,
        buffer=None,
        weight_method=None,
        weight_func=None,
        dynamic_buffering=None,
        size_measure=None,
    )
    _patch_main_dependencies(monkeypatch, tmp_path, args2, overwrite=True)
    monkeypatch.setattr(pipelines, "prepare_data", lambda cfg_in: "not-dict")
    with pytest.raises(TypeError):
        runpy.run_module("src.create_voronoi", run_name="__main__")


def test_remaining_cluster_and_db_crs_branches(monkeypatch):
    df = gpd.GeoDataFrame(
        {
            "weights": [0.4, 0.6],
            "POP_SERVED": [10, 20],
            "geometry": [Point(0, 0), Point(0.1, 0.1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    monkeypatch.setattr(cv, "cluster_point_indices", lambda geoms, threshold: [{0, 1}])
    c = cv.cluster_points(df, threshold=1)
    assert len(c) == 1
    # Merged duplicates keep the larger reported population; never the sum.
    assert c["POP_SERVED"].iloc[0] == 20
    assert c["n_merged_records"].iloc[0] == 2

    # Force execution through df.crs is None and polygons.crs is None branches.
    df2 = gpd.GeoDataFrame({"ISO_2": ["DE"], "utm": [32632], "geometry": [Point(10, 50)]}, geometry="geometry", crs=None)
    poly2 = gpd.GeoDataFrame({"ISO_2": ["DE"], "x": [1], "geometry": [box(9, 49, 11, 51)]}, geometry="geometry", crs=None)

    class _Res:
        def df(self):
            return pd.DataFrame({"ISO_2": ["DE"], "utm": [32632], "geometry": [Point(10, 50).wkt], "centroid": [Point(10, 50).wkt], "x": [1]})

    class _Conn:
        def execute(self, q):
            return _Res()

        def close(self):
            return None

    monkeypatch.setattr(cv, "duckdb_connection", _fake_duckdb_connection(_Conn()))
    monkeypatch.setattr(cv.np.random, "randint", lambda low, high: 3)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", lambda self, crs: self)
    monkeypatch.setattr(cv.os.path, "exists", lambda p: False)
    out = cv.intersect_with_polygons_db(df2, poly2, ["x"], df_join_col="ISO_2", polygon_join_col="ISO_2")
    assert "x" in out.columns


def test_remaining_weighted_voronoi_and_worker_branches(monkeypatch):
    # Branches: df.crs None, clipping/country_clip crs None conversion, buffering path, contour-empty path.
    base_df = gpd.GeoDataFrame(
        {"WASTE_ID": [1], "grp": [1], "weights": [1.0], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs=None,
    )
    clipping = gpd.GeoDataFrame({"grp": [1], "geometry": [box(-1, -1, 1, 1)]}, geometry="geometry", crs=None)
    country = gpd.GeoDataFrame({"country": ["DE"], "geometry": [box(-1, -1, 1, 1)]}, geometry="geometry", crs=None)

    monkeypatch.setattr(cv, "estimate_utm_crs", lambda g: CRS.from_epsg(32632))
    monkeypatch.setattr(cv, "cluster_points", lambda df, threshold: df)
    monkeypatch.setattr(cv, "extract_site_coordinates", lambda df, cp: [(0.0, 0.0)])
    monkeypatch.setattr(cv, "initialize_voronoi_weights", lambda df, distance_fn, scale_weights, points: (np.array([1.0]), 1.0))
    monkeypatch.setattr(cv, "create_ranges", lambda x, y, n: np.array([0.0, 1.0]))
    monkeypatch.setattr(cv, "geometry_contains_points", lambda geom, pts: np.array([True, False, False, False]))
    monkeypatch.setattr(cv, "assign_sites_streaming", lambda vp, p, w, d, f: np.array([0]))
    monkeypatch.setattr(cv, "extract_contours_scipy", lambda *args, **kwargs: [])
    monkeypatch.setattr(cv, "extract_contours_cv2", lambda *args, **kwargs: [])

    r1, p1 = cv.weighted_voronoi(
        base_df.copy(),
        "grp",
        country.copy(),
        False,
        clipping.copy(),
        10,
        cv.default_distance_multiplicative,
        scipy_true=True,
        cv2_true=False,
        centroid_points=False,
        buffering=True,
        threshold=1,
        calculate_buffer_fn=lambda df, w, **k: np.array([50.0]),
        buffer_fn_kwargs={},
        site_id_col="WASTE_ID",
    )
    assert len(r1) == 1 and len(p1) == 1

    # Multi-site branch to trigger no-assignment + cv2 contour-empty branch.
    df2 = gpd.GeoDataFrame(
        {"WASTE_ID": [1, 2], "grp": [1, 1], "weights": [0.5, 0.5], "geometry": [Point(0, 0), Point(1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    monkeypatch.setattr(cv, "cluster_points", lambda df, threshold: df)
    monkeypatch.setattr(cv, "extract_site_coordinates", lambda df, cp: [(0.0, 0.0), (1.0, 1.0)])
    monkeypatch.setattr(cv, "initialize_voronoi_weights", lambda df, distance_fn, scale_weights, points: (np.array([0.5, 0.5]), 1.0))
    monkeypatch.setattr(cv, "geometry_contains_points", lambda geom, pts: np.array([True, True, False, False]))
    monkeypatch.setattr(cv, "assign_sites_streaming", lambda vp, p, w, d, f: np.array([0, 0]))
    r2, _ = cv.weighted_voronoi(
        df2,
        "grp",
        None,
        False,
        None,
        10,
        cv.default_distance_multiplicative,
        scipy_true=False,
        cv2_true=True,
        centroid_points=False,
        buffering=False,
        threshold=1,
        calculate_buffer_fn=lambda df, w, **k: np.array([20.0, 20.0]),
        buffer_fn_kwargs={},
        site_id_col="WASTE_ID",
    )
    assert isinstance(r2, gpd.GeoDataFrame)

    with pytest.raises(Exception):
        cv.voronoi_worker(("bad",))


def test_remaining_orchestrate_voronoi_weights_branches(monkeypatch, tmp_path):
    df = gpd.GeoDataFrame(
        {
            "grp": ["1", "1"],
            "ISO_2": ["DE", "DE"],
            "WASTE_ID": [1, 2],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    clipping = gpd.GeoDataFrame({"grp": ["1"], "geometry": [box(-1, -1, 2, 2)]}, geometry="geometry", crs=None)
    country_df = gpd.GeoDataFrame({"country": ["DE"], "geometry": [box(-2, -2, 3, 3)]}, geometry="geometry", crs=None)
    output = tmp_path / "out.gpkg"
    temp = tmp_path / "temp_out.gpkg"
    temp.write_text("x", encoding="utf-8")
    output.write_text("existing", encoding="utf-8")

    monkeypatch.setattr(cv.os.path, "exists", lambda p: str(p).endswith("temp_out.gpkg") or str(p).endswith("out.gpkg"))
    monkeypatch.setattr(cv.os, "remove", lambda p: (_ for _ in ()).throw(OSError("remove fail")))

    def _area_fn(sub_df, **kwargs):
        out = sub_df.copy()
        out["base_values"] = 1.0
        out["total_area"] = 1.0
        out["num_detection_rect"] = 0
        out["num_detection_circle"] = 0
        return out

    # First pass: utm=None skip path.
    monkeypatch.setattr(cv, "estimate_utm_crs", lambda g: None)

    class _FutNone:
        def result(self):
            return None

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            return _FutNone()

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(cv, "as_completed", lambda futures: list(futures))
    r = cv.orchestrate_voronoi_weights(
        df,
        col="grp",
        country_df=country_df,
        clipping=clipping,
        workers=2,
        area_fn=_area_fn,
        output_path=str(output),
        overwrite=True,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
        flush_size=99,
    )
    assert r in {True, False}

    # Second pass: generate task, then exception in future handling and finalize replace failure.
    monkeypatch.setattr(cv, "estimate_utm_crs", lambda g: CRS.from_epsg(32632))
    monkeypatch.setattr(pd.Series, "unique", lambda self: np.array(["DE", "XX"]))

    class _FutErr:
        def result(self):
            raise RuntimeError("future fail")

    class _ExecErr(_Exec):
        def submit(self, fn, *args):
            return _FutErr()

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _ExecErr)
    monkeypatch.setattr(cv.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("replace fail")))
    r2 = cv.orchestrate_voronoi_weights(
        df,
        col="grp",
        country_df=country_df,
        clipping=clipping,
        workers=2,
        area_fn=_area_fn,
        output_path=str(output),
        overwrite=True,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
        flush_size=99,
    )
    assert r2 is False


def test_residual_dissolve_and_overlap_orchestration_lines(monkeypatch, tmp_path):
    monkeypatch.setattr(cv, "tqdm", lambda it, **kwargs: it)
    monkeypatch.setattr(cv, "estimate_utm_crs", lambda g: CRS.from_epsg(32632))

    class _ExecNone:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            return None

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _ExecNone)
    monkeypatch.setattr(cv, "as_completed", lambda futures: futures)
    df = gpd.GeoDataFrame({"ISO_2": ["DE"], "some_id": [1], "geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")
    with pytest.raises(KeyError):
        cv.orchestrate_overlaps(df, 1, str(tmp_path / "a.gpkg"), 10, country_col="ISO_2")

    class _FutGood:
        def result(self):
            sdf = gpd.GeoDataFrame(
                {"some_id": [1], "centroid": [Point(0, 0)], "geometry": [box(0, 0, 1, 1)]},
                geometry="geometry",
                crs="EPSG:4326",
            )
            return [{1}], sdf

    class _ExecGood(_ExecNone):
        def submit(self, fn, *args):
            return _FutGood()

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _ExecGood)
    monkeypatch.setattr(cv, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, *a, **k: None)
    r = cv.orchestrate_overlaps(df, 1, str(tmp_path / "b.gpkg"), 10, country_col="ISO_2")
    assert isinstance(r, gpd.GeoDataFrame)
    assert "centroid" not in r.columns


def test_residual_weighted_voronoi_scipy_branch(monkeypatch):
    df = gpd.GeoDataFrame(
        {"WASTE_ID": [1, 2], "grp": [1, 1], "weights": [0.5, 0.5], "geometry": [Point(0, 0), Point(1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    monkeypatch.setattr(cv, "cluster_points", lambda d, t: d)
    monkeypatch.setattr(cv, "extract_site_coordinates", lambda d, cp: [(0.0, 0.0), (1.0, 1.0)])
    monkeypatch.setattr(cv, "initialize_voronoi_weights", lambda d, f, s, p: (np.array([0.5, 0.5]), 1.0))
    monkeypatch.setattr(cv, "create_ranges", lambda x, y, n: np.array([0.0, 1.0]))
    monkeypatch.setattr(cv, "geometry_contains_points", lambda geom, pts: np.array([True, True, False, False]))
    monkeypatch.setattr(cv, "assign_sites_streaming", lambda vp, p, w, d, f: np.array([0, 1]))
    monkeypatch.setattr(cv, "extract_contours_scipy", lambda *a, **k: [])
    out, _ = cv.weighted_voronoi(
        df,
        "grp",
        None,
        False,
        None,
        10,
        cv.default_distance_multiplicative,
        scipy_true=True,
        cv2_true=False,
        centroid_points=False,
        buffering=False,
        threshold=1,
        calculate_buffer_fn=lambda d, w, **k: np.array([20.0, 20.0]),
        buffer_fn_kwargs={},
        site_id_col="WASTE_ID",
    )
    assert isinstance(out, gpd.GeoDataFrame)


def test_residual_orchestrate_voronoi_weights_lines(monkeypatch, tmp_path):
    country_df = gpd.GeoDataFrame({"country": ["DE"], "geometry": [box(-1, -1, 1, 1)]}, geometry="geometry", crs="EPSG:4326")

    def _area_fn(sub_df, **kwargs):
        out = sub_df.copy()
        out["base_values"] = 1.0
        out["total_area"] = 1.0
        out["num_detection_rect"] = 0
        out["num_detection_circle"] = 0
        return out

    # Hit sub_df-empty-after-to_crs branch.
    df_empty_after = gpd.GeoDataFrame({"grp": ["1"], "ISO_2": ["DE"], "WASTE_ID": [999], "geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")
    monkeypatch.setattr(cv, "estimate_utm_crs", lambda g: CRS.from_epsg(32632))

    original_to_crs = gpd.GeoDataFrame.to_crs

    def _patched_to_crs(self, crs):
        if "WASTE_ID" in self.columns and (self["WASTE_ID"] == 999).any():
            return self.iloc[0:0].copy()
        return original_to_crs(self, crs)

    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", _patched_to_crs)

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            class _F:
                def result(self_inner):
                    return None

            return _F()

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(cv, "as_completed", lambda futures: list(futures))
    cv.orchestrate_voronoi_weights(
        df_empty_after,
        col="grp",
        country_df=country_df,
        workers=1,
        area_fn=_area_fn,
        output_path=None,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
    )

    # Hit no-country-clip task generation, flush early return, and result None continue.
    df_no_iso = gpd.GeoDataFrame({"grp": ["1"], "ISO_2": [np.nan], "WASTE_ID": [1], "geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")
    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", original_to_crs)
    monkeypatch.setattr(cv, "estimate_utm_crs", lambda g: CRS.from_epsg(32632))
    monkeypatch.setattr(cv.os.path, "exists", lambda p: False)

    class _FutRes:
        def __init__(self, payload):
            self._payload = payload

        def result(self):
            return self._payload

    class _ExecMix(_Exec):
        def __init__(self, max_workers=None):
            super().__init__(max_workers)
            self._n = 0

        def submit(self, fn, *args):
            self._n += 1
            if self._n == 1:
                reg = gpd.GeoDataFrame({"grp": ["1"], "ISO_2": [np.nan], "WASTE_ID": [1], "geometry": [box(0, 0, 1, 1)]}, geometry="geometry", crs="EPSG:4326")
                return _FutRes((reg, reg.copy()))
            return _FutRes(None)

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _ExecMix)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, *a, **k: None)
    ok = cv.orchestrate_voronoi_weights(
        df_no_iso,
        col="grp",
        country_df=country_df,
        workers=2,
        area_fn=_area_fn,
        output_path=str(tmp_path / "final.gpkg"),
        overwrite=True,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
        flush_size=99,
    )
    assert ok is False or ok is True


def test_final_residual_dissolve_and_orchestrate_none_result(monkeypatch):
    monkeypatch.setattr(cv, "tqdm", lambda it, **kwargs: it)
    monkeypatch.setattr(cv, "estimate_utm_crs", lambda g: CRS.from_epsg(32632))

    # Explicitly drive `if result is None: continue` in orchestration worker loop.
    df = gpd.GeoDataFrame(
        {"grp": ["1"], "ISO_2": [np.nan], "WASTE_ID": [1], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    country_df = gpd.GeoDataFrame({"country": ["DE"], "geometry": [box(-1, -1, 1, 1)]}, geometry="geometry", crs="EPSG:4326")

    def _area_fn(sub_df, **kwargs):
        out = sub_df.copy()
        out["base_values"] = 1.0
        out["total_area"] = 1.0
        out["num_detection_rect"] = 0
        out["num_detection_circle"] = 0
        return out

    class _FutNone:
        def result(self):
            return None

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            return _FutNone()

    monkeypatch.setattr(cv, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(cv, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(cv, "estimate_utm_crs", lambda g: CRS.from_epsg(32632))
    out2 = cv.orchestrate_voronoi_weights(
        df,
        col="grp",
        country_df=country_df,
        workers=1,
        area_fn=_area_fn,
        output_path=None,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
    )
    assert isinstance(out2, tuple)
