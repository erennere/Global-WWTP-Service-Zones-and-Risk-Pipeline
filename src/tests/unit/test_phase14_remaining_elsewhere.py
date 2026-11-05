from __future__ import annotations

import argparse
import os
import runpy
import sys
import types

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from src import download_pop as dp
from src.annotation_scripts import NEW_01_GENERATEGRIDS as n1
from src.annotation_scripts import NEW_02_EXTRACTOSMDATAFULL_GEOJSON as n2
from src.annotation_scripts import NEW_03_WASTEWATERJOIN_GEOJSON as ww3
from src.annotation_scripts import download_bing_annotate as dba
from src.figures_scripts import piechart_figure as pf
from src import starter
from src.pop_at_risk_river_calculations import create_rasters as cr
from src.pop_at_risk_river_calculations import find_intersection_river as fir
from src.pop_at_risk_river_calculations import impact_polygons_pop as ipp
from src.industrial_analysis import download_and_vectorize as dv


pytestmark = pytest.mark.unit


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


def test_download_pop_get_urls_from_hdx_branches(monkeypatch):
    class _DatasetObj:
        def __init__(self, resources):
            self._resources = resources

        def get_resources(self):
            return self._resources

    class _Dataset:
        @staticmethod
        def search_in_hdx(_q):
            return [
                _DatasetObj(
                    [
                        {"download_url": "http://x/deu.zip", "name": "population_deu.geotiff.zip"},
                        {"download_url": "http://x/deu2.zip", "name": "population_deu.geotiff.zip"},
                        {"download_url": "http://x/anr.zip", "name": "ANR_men_geotiff.zip"},
                    ]
                )
            ]

    class _Configuration:
        @staticmethod
        def create(**_kwargs):
            return None

    mod_easy = types.ModuleType("hdx.utilities.easy_logging")
    mod_easy.setup_logging = lambda: None
    mod_cfg = types.ModuleType("hdx.api.configuration")
    mod_cfg.Configuration = _Configuration
    mod_dataset = types.ModuleType("hdx.data.dataset")
    mod_dataset.Dataset = _Dataset

    monkeypatch.setitem(sys.modules, "hdx.utilities.easy_logging", mod_easy)
    monkeypatch.setitem(sys.modules, "hdx.api.configuration", mod_cfg)
    monkeypatch.setitem(sys.modules, "hdx.data.dataset", mod_dataset)

    urls = dp.get_urls_from_hdx()
    assert "deu" in urls
    assert "anr" in urls
    assert len(urls["deu"]) == 1


def test_download_pop_download_file_error_branch(monkeypatch, tmp_path):
    class _ReqErr(dp.requests.RequestException):
        pass

    monkeypatch.setattr(dp.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(_ReqErr("x")))
    ok = dp.download_file("http://bad", str(tmp_path / "out.bin"))
    assert ok is False


def test_download_bing_log_preview_and_bbox_error_branches(monkeypatch, tmp_path):
    gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
    dba.log_gdf_preview("x", gdf, ["missing"])
    dba.log_gdf_preview("x", gdf, ["geometry"])

    bbox = box(0, 0, 10, 10)
    poly = gpd.GeoDataFrame({"grid": [1], "geometry": [box(1, 1, 2, 2)]}, geometry="geometry", crs="EPSG:3857")
    line = gpd.GeoDataFrame({"grid": [1], "geometry": [LineString([(1, 1), (2, 2)])]}, geometry="geometry", crs="EPSG:3857")

    monkeypatch.setattr(dba, "get_image", lambda img_idx, images_dir: None)
    idx, n, err = dba.process_bbox(1, bbox, 77, poly, ["man_made"], line, ["waterway"], str(tmp_path), str(tmp_path))
    assert idx == 1 and n == 0 and "Image not found" in err

    monkeypatch.setattr(dba, "get_image", lambda img_idx, images_dir: object())
    idx2, n2, err2 = dba.process_bbox(2, bbox, 1, poly, ["man_made"], line, ["waterway"], str(tmp_path), str(tmp_path))
    assert idx2 == 2 and n2 == 0 and err2 is not None


def test_download_bing_annotate_parallel_error_logging_branch(monkeypatch, tmp_path):
    bbox = gpd.GeoDataFrame({"idx": [1], "img_idx": [1], "geometry": [box(0, 0, 1, 1)]}, geometry="geometry", crs="EPSG:3857")
    poly = gpd.GeoDataFrame({"grid": [1], "geometry": [box(0, 0, 1, 1)]}, geometry="geometry", crs="EPSG:3857")
    line = gpd.GeoDataFrame({"grid": [1], "geometry": [LineString([(0, 0), (1, 1)])]}, geometry="geometry", crs="EPSG:3857")

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

        def submit(self, fn, *args, **kwargs):
            return _Future((1, 0, "fail"))

    monkeypatch.setattr(dba, "ThreadPoolExecutor", _Exec)
    monkeypatch.setattr(dba, "as_completed", lambda futures: list(futures))
    with pytest.raises(RuntimeError, match="bbox annotation task"):
        dba.annotate_bboxes_parallel(bbox, poly, ["man_made"], line, ["waterway"], str(tmp_path), str(tmp_path), set())


def test_download_bing_main_smoke(monkeypatch, tmp_path):
    args = argparse.Namespace(
        instance_id=0,
        num_instances=1,
        split_seed=42,
        level=None,
        version=None,
        buffer=None,
        weight_method=None,
        weight_func=None,
        dynamic_buffering=None,
        dynamic_buffer_k=None,
    )
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)

    images_dir = tmp_path / "images"
    grid_dir = tmp_path / "grid"
    osm_dir = tmp_path / "osm"
    out_dir = tmp_path / "out"
    for d in [images_dir, grid_dir, osm_dir, out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = {
        "annotations": {
            "cell_size": 3072,
            "factor": 1.194,
            "image_size_px": 3072,
            "zoom_level": 17,
            "base_z17_resolution": 1.1943285669555664,
            "max_workers": 1,
            "georeferenced": False,
            "fontsize": 24,
            "dpi": 300,
            "target_size_px": 1024,
            "mercator_tile_size_px": 256,
            "imagery_reference_tile_size_px": 3072,
            "earth_radius_m": 6378137,
        },
        "paths": {
            "annotations_images_dir": str(images_dir),
            "annotations_grid_dir": str(grid_dir),
            "annotations_temp_parquet_dir": str(tmp_path / "tmp_parquet"),
            "annotations_by_osm_dir": str(osm_dir),
            "corrected_all_filepath": str(tmp_path / "all.csv"),
            "annotated_images_output_dir": str(out_dir),
        }
    }

    monkeypatch.setattr(starter, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(starter, "load_config", lambda **kwargs: cfg)

    grids = gpd.GeoDataFrame({"idx": [1], "geometry": [box(0, 0, 10, 10)]}, geometry="geometry", crs="EPSG:4326")
    points = gpd.GeoDataFrame({"idx": [1], "geometry": [Point(5, 5)]}, geometry="geometry", crs="EPSG:4326")

    def _read_file(path):
        p = str(path)
        if p.endswith("ref.geojson"):
            return points.copy()
        return grids.copy()

    monkeypatch.setattr(gpd, "read_file", _read_file)

    # Pretend one polygon and one line geojson exist for grid 1.
    monkeypatch.setattr(dba.os, "listdir", lambda p: ["idx_1_polygons.geojson", "idx_1_lines.geojson"] if str(p) == str(osm_dir) else [])

    # Seed a source image so process_bbox can load it.
    dba.Image.new("RGB", (64, 64), "black").save(images_dir / "0.png")

    class _QRes:
        def __init__(self, frame):
            self._frame = frame

        def df(self):
            return self._frame

    class _Conn:
        def execute(self, query):
            q = str(query)
            if "DESCRIBE SELECT * FROM ST_READ" in q:
                return _QRes(pd.DataFrame({"column_name": ["geom", "man_made", "waterway"]}))
            if "_polygons.geojson" in q and "ST_AsText(geom) AS geometry" in q:
                return _QRes(pd.DataFrame({"man_made": ["wwtp"], "geometry": [box(0, 0, 1, 1).wkt], "grid": ["1"]}))
            if "_lines.geojson" in q and "ST_AsText(geom) AS geometry" in q:
                return _QRes(pd.DataFrame({"waterway": ["river"], "geometry": [LineString([(0, 0), (1, 1)]).wkt], "grid": ["1"]}))
            return _QRes(pd.DataFrame())

        def close(self):
            return None

    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: _Conn())
    runpy.run_module("src.annotation_scripts.download_bing_annotate", run_name="__main__")


def test_new01_point_to_square_none_and_valid():
    assert n1.point_to_square(None, 5) is None

    poly = n1.point_to_square(Point(1, 2), 3)
    assert poly is not None
    assert poly.bounds == (-2.0, -1.0, 4.0, 5.0)


def test_new01_main_script_smoke(monkeypatch, tmp_path):
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "annotations": {"cell_size": 10, "factor": 2},
        "paths": {
            "corrected_all_filepath": str(tmp_path / "all_points.gpkg"),
            "annotations_grid_dir": str(grid_dir),
        },
    }

    monkeypatch.setattr(starter, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(starter, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(n1.os, "chdir", lambda path: None)

    src = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    monkeypatch.setattr(gpd, "read_file", lambda path: src.copy())

    captured = {}

    def _to_file(self, path, driver=None, **kwargs):
        captured["path"] = str(path)
        captured["driver"] = driver
        captured["rows"] = len(self)

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", _to_file)

    runpy.run_module("src.annotation_scripts.NEW_01_GENERATEGRIDS", run_name="__main__")

    assert captured["driver"] == "GPKG"
    assert captured["rows"] == 2
    assert captured["path"].endswith("grids_all_points.gpkg")


def test_download_pop_mosaic_large_rasters_multifile_paths(monkeypatch, tmp_path):
    class _Bounds:
        def __init__(self, left, bottom, right, top):
            self.left = left
            self.bottom = bottom
            self.right = right
            self.top = top

        def __iter__(self):
            return iter((self.left, self.bottom, self.right, self.top))

    class _Window:
        def __init__(self, row_off, col_off, height, width):
            self.row_off = row_off
            self.col_off = col_off
            self.height = height
            self.width = width

        def round_offsets(self):
            return self

        def round_lengths(self):
            return self

    class _Src:
        def __init__(self, fp):
            self.fp = fp
            self.bounds = _Bounds(0, 0, 2, 2)
            self.crs = "EPSG:4326"
            self.dtypes = ["float32"]
            self.res = (1, 1) if "res1" in fp else (2, 2)
            self.count = 2
            self.transform = object()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, *_args, out_shape=None, **_kwargs):
            if out_shape is None:
                return np.ones((2, 2), dtype=np.float32)
            h, w = out_shape[1], out_shape[2]
            return np.ones((1, h + 1, w + 1), dtype=np.float32)

    class _Dst:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, arr):
            self.arr = arr

    def _open(fp, mode="r", **_kwargs):
        if mode == "w+":
            return _Dst()
        return _Src(str(fp))

    monkeypatch.setattr(dp.rasterio, "open", _open)
    monkeypatch.setattr(dp.rasterio.transform, "from_origin", lambda *args, **kwargs: object())
    monkeypatch.setattr(dp, "from_bounds", lambda *args, **kwargs: _Window(0, 0, 1, 1))
    monkeypatch.setattr(dp, "resample_raster", lambda src, tt, ts, crs: np.ones(ts, dtype=np.float32))
    monkeypatch.setattr(dp, "ensure_output_dir_for_file", lambda path: None)

    files = [str(tmp_path / "a_res1.tif"), str(tmp_path / "b_res2.tif")]
    dp.mosaic_large_rasters(files, str(tmp_path / "out.tif"))


def test_download_pop_process_single_country_and_pool_branches(monkeypatch, tmp_path):
    country_urls = {"deu": ["u1"], "fra": ["u2"]}

    # Branch: early return when extraction fails.
    monkeypatch.setattr(dp, "download_save_and_unzip_pops", lambda *args, **kwargs: None)
    assert dp.process_single_country(country_urls, "deu", data_dir=str(tmp_path)) is None

    # Branch: csv workflow with failed rasterize and no valid files.
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dp, "download_save_and_unzip_pops", lambda *args, **kwargs: str(extract_dir))
    monkeypatch.setattr(dp, "find_files", lambda folder: ([str(tmp_path / "a.csv")], False))
    monkeypatch.setattr(dp.pd, "read_csv", lambda path: pd.DataFrame({"lat": [0], "lon": [0], "pop": [1]}))
    monkeypatch.setattr(dp, "rasterize_csv", lambda df, output_path, res=30: None)
    monkeypatch.setattr(dp.os.path, "exists", lambda p: True if str(p).endswith("merged") or "rasterized" in str(p) else False)
    monkeypatch.setattr(dp.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(dp, "mosaic_large_rasters", lambda *args, **kwargs: None)
    dp.process_single_country(country_urls, "deu", data_dir=str(tmp_path))

    class _Future:
        def __init__(self, err=False):
            self.err = err

        def result(self):
            if self.err:
                raise RuntimeError("boom")
            return None

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            country = args[1]
            return _Future(err=(country == "fra"))

    monkeypatch.setattr(dp, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(dp, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(dp, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(dp, "process_single_country", lambda *args, **kwargs: None)
    dp.process_all_countries(country_urls, res=25, max_workers=2, data_dir=str(tmp_path))


def test_download_pop_main_smoke(monkeypatch, tmp_path):
    cfg = {
        "paths": {"pop_dir": str(tmp_path / "pop")},
        "start_year": 2015,
        "end_year": 2024,
        "country_limit": 0,
        "worldpop_2014_url_template": "tpl2014",
        "worldpop_yearly_url_template": "tplyear",
    }

    monkeypatch.setattr(dp.os, "chdir", lambda path: None)
    monkeypatch.setattr(dp, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(dp, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(
        dp,
        "get_urls",
        lambda **kwargs: {
            "deu": ["u1"],
            "fra": ["u2"],
            "usa": ["u3"],
            "bra": ["u4"],
        },
    )

    seen = {}
    monkeypatch.setattr(
        dp,
        "process_all_countries",
        lambda country_urls, res, max_workers, data_dir: seen.update(
            {
                "countries": sorted(country_urls.keys()),
                "res": res,
                "max_workers": max_workers,
                "data_dir": data_dir,
            }
        ),
    )

    dp.main(res=12, max_workers=3)
    assert seen["countries"] == ["bra", "deu", "fra", "usa"]
    assert seen["res"] == 12
    assert seen["max_workers"] == 3


def test_create_rasters_main_smoke(monkeypatch, tmp_path):
    args = argparse.Namespace(job_index=0, total_jobs=1)
    monkeypatch.setattr(cr, "parse_args", lambda: args)
    monkeypatch.setattr(cr, "parse_config_overrides", lambda args=None: {})

    output_tif_dir = tmp_path / "out_tifs"
    watershed_path = tmp_path / "watershed.geojson"
    overture_path = tmp_path / "overture.parquet"

    cfg = {
        "annotations": {"max_workers": 2, "random_seed": 11},
        "min_pixels": 3,
        "served_all_touched": False,
        "zoom_level": 8,
        "figures": {"approach": "1"},
        "paths": {
            "pop_tif_dir": str(tmp_path / "pop_tifs"),
            "WWTP_tif_dir": str(output_tif_dir),
            "non_served_outpath": str(tmp_path / "non_served.gpkg"),
            "csv_output_filepath": str(tmp_path / "stats.gpkg"),
            "watershed": str(watershed_path),
            "watershed_with_countries": str(tmp_path / "watershed_with_countries.gpkg"),
            "overture": str(overture_path),
            "overture_s3_url": "s3://dummy/overture",
        },
        "country_output_column": "iso3",
        "country_boundary_column": "country_name",
        "basin_column_name": "HYBAS_ID",
    }
    monkeypatch.setattr(cr, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(cr, "create_pop_output_paths", lambda cfg_: {"voronoi": {"1": str(tmp_path / "voronoi.gpkg")}})

    voronoi = gpd.GeoDataFrame(
        {"iso3": ["DEU"], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    watershed = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "country_name": ["Germany"], "geometry": [box(0, 0, 2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    def _read_file(path, crs=None):
        if str(path).endswith("voronoi.gpkg"):
            return voronoi.copy()
        return watershed.copy()

    monkeypatch.setattr(cr.gpd, "read_file", _read_file)
    monkeypatch.setattr(cr, "find_newest_country_tif_files", lambda countries, tif_dir: {"DEU": str(tmp_path / "deu.tif")})
    monkeypatch.setattr(cr.os, "chdir", lambda path: None)
    monkeypatch.setattr(cr.os.path, "exists", lambda p: False)
    monkeypatch.setattr(cr.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(cr, "download_overture_maps", lambda s3, out: None)
    monkeypatch.setattr(
        cr,
        "intersects_with_country_db",
        lambda gdf, overture, polygon_country_col, output_country_col: gdf.assign(**{output_country_col: ["DEU"]}),
    )
    monkeypatch.setattr(cr, "ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, *args, **kwargs: None)

    seen = {}
    monkeypatch.setattr(
        cr,
        "orchestrate_intersections",
        lambda tif_dict, gdf, watershed_gdf, output_tif_dir, csv_output_filepath, non_served_outpath, max_workers, **kwargs: seen.update(
            {
                "tif_count": len(tif_dict),
                "gdf_rows": len(gdf),
                "watershed_rows": len(watershed_gdf),
                "max_workers": max_workers,
            }
        ),
    )

    cr.main()
    assert seen["tif_count"] == 1
    assert seen["gdf_rows"] == 1
    assert seen["watershed_rows"] == 1
    assert seen["max_workers"] == 2


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
    monkeypatch.setattr(ww3, "ensure_output_dir_for_file", lambda path: None)

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

    monkeypatch.setattr(ww3.random, "randint", lambda a, b: 456)
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


def test_new03_script_skips_main_when_output_file_missing(monkeypatch, tmp_path):
    # FIX [D-1]: script entrypoint should skip clustering if merged parquet is absent.
    cfg = {
        "overwrite_existing": False,
        "paths": {
            "corrected_all_filepath": str(tmp_path / "points.gpkg"),
            "annotations_grid_dir": str(tmp_path / "grids"),
            "annotations_by_osm_dir": str(tmp_path / "osm"),
            "annotations_temp_parquet_dir": str(tmp_path / "tmp_parquets"),
        },
    }

    os.makedirs(cfg["paths"]["annotations_grid_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["annotations_by_osm_dir"], exist_ok=True)

    merged_path = os.path.join(os.path.abspath(os.path.join(cfg["paths"]["annotations_grid_dir"], "..")), "data", "merged_polygons.parquet")
    merged_lines_path = merged_path.replace("polygons", "lines")
    original_exists = os.path.exists
    seen = {"merged_checks": 0, "lines_checks": 0}

    def _exists(path):
        abs_path = os.path.abspath(str(path))
        if abs_path == os.path.abspath(merged_path):
            seen["merged_checks"] += 1
            # First check: appear present to skip merge task creation.
            # Second check: appear absent so guarded main call is skipped.
            return seen["merged_checks"] == 1
        if abs_path == os.path.abspath(merged_lines_path):
            seen["lines_checks"] += 1
            # Keep lines output present to avoid creating any merge task.
            return True
        return original_exists(path)

    monkeypatch.setattr(starter, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(starter, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(os.path, "exists", _exists)

    runpy.run_path(str(ww3.__file__), run_name="__main__")


def test_impact_polygons_main_smoke_writes_outputs(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "non_served_nxt_river_outpath": str(tmp_path / "non_served.gpkg"),
            "rivershed_output_path": str(tmp_path / "rivers.gpkg"),
            "impact_pop_polygons_outpath": str(tmp_path / "impact.gpkg"),
        },
        "impact_polygons_pop_params": {
            "width": 12,
            "c_limit": 5,
            "org_per_pop": 60,
            "base_k": 0.23,
            "theta": 1.047,
            "step_m": 100,
            "least_discharge_cms": 0.269,
            "impact_radii": [1000, 2000],
        },
    }

    pop_gdf = gpd.GeoDataFrame(
        {
            "NXT_DIS": [1],
            "pop_sum": [20],
            "country": ["DE"],
            "utm": [32632],
            "geometry": [Point(10, 50)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    river_gdf = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1],
            "NEXT_DOWN": [0],
            "MAIN_RIV": [1],
            "DIS_AV_CMS": [1.0],
            "geometry": [LineString([(10, 50), (10.1, 50.1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(ipp.os, "chdir", lambda path: None)
    monkeypatch.setattr(ipp, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(ipp, "load_config", lambda **kwargs: cfg)

    def _read_file(path, columns=None, **kwargs):
        if str(path).endswith("non_served.gpkg"):
            return pop_gdf.copy()
        return river_gdf.copy()

    monkeypatch.setattr(ipp.gpd, "read_file", _read_file)
    monkeypatch.setattr(ipp, "batch_estimate_utm_epsg", lambda gdf: (np.array([32632]), np.array([50.0])))
    monkeypatch.setattr(ipp, "create_dicts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ipp,
        "orchestrate_logic",
        lambda *args, **kwargs: {
            1000.0: gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, geometry="geometry", crs="EPSG:4326"),
            2000.0: gpd.GeoDataFrame({"geometry": [box(0, 0, 2, 2)]}, geometry="geometry", crs="EPSG:4326"),
        },
    )
    monkeypatch.setattr(ipp, "ensure_output_dir_for_file", lambda path: None)

    written = []
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, path, driver="GPKG": written.append(str(path)))
    monkeypatch.setattr(sys, "argv", ["prog", "--max-workers", "8"])

    ipp.main()
    assert len(written) == 2
    assert any(p.endswith("impact_1000.gpkg") for p in written)
    assert any(p.endswith("impact_2000.gpkg") for p in written)


def test_impact_polygons_main_no_output_branch(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "non_served_nxt_river_outpath": str(tmp_path / "non_served.gpkg"),
            "rivershed_output_path": str(tmp_path / "rivers.gpkg"),
            "impact_pop_polygons_outpath": str(tmp_path / "impact.gpkg"),
        },
        "impact_polygons_pop_params": {
            "org_per_pop": 60,
            "width": 12,
            "c_limit": 5,
            "base_k": 0.23,
            "theta": 1.047,
            "step_m": 100,
            "least_discharge_cms": 0.269,
            "impact_radii": [1000, 2000],
        },
    }

    pop_gdf = gpd.GeoDataFrame(
        {
            "NXT_DIS": [1],
            "pop_sum": [10],
            "country": ["DE"],
            "utm": [32632],
            "geometry": [Point(10, 50)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    river_gdf = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1],
            "NEXT_DOWN": [0],
            "MAIN_RIV": [1],
            "DIS_AV_CMS": [1.0],
            "geometry": [LineString([(10, 50), (10.1, 50.1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(ipp.os, "chdir", lambda path: None)
    monkeypatch.setattr(ipp, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(ipp, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(ipp.gpd, "read_file", lambda path, columns=None, **kwargs: pop_gdf.copy() if str(path).endswith("non_served.gpkg") else river_gdf.copy())
    monkeypatch.setattr(ipp, "batch_estimate_utm_epsg", lambda gdf: (np.array([32632]), np.array([50.0])))
    monkeypatch.setattr(ipp, "create_dicts", lambda *args, **kwargs: None)
    monkeypatch.setattr(ipp, "orchestrate_logic", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["prog"])

    ipp.main()


def test_new02_elements_timer_and_find_bbox_branches(monkeypatch):
    # elements_to_gdf: empty/no-elements branch
    l0, p0 = n2.elements_to_gdf(None)
    assert l0.empty and p0.empty

    # elements_to_gdf: relation-with-outer polygon branch
    data = {
        "elements": [
            {"type": "node", "id": 1, "lon": 0.0, "lat": 0.0},
            {"type": "node", "id": 2, "lon": 1.0, "lat": 0.0},
            {"type": "node", "id": 3, "lon": 1.0, "lat": 1.0},
            {"type": "node", "id": 4, "lon": 0.0, "lat": 1.0},
            {"type": "way", "id": 10, "nodes": [1, 2, 3, 4, 1], "tags": {}},
            {
                "type": "relation",
                "id": 20,
                "members": [{"type": "way", "role": "outer", "ref": 10}],
                "tags": {"landuse": "industrial"},
            },
        ]
    }
    l1, p1 = n2.elements_to_gdf(data)
    assert l1.empty
    assert len(p1) >= 1

    # find_bbox: invalid that stays invalid after buffer => None branch
    class _BadGeom:
        is_empty = False
        is_valid = False

        def buffer(self, _):
            return self

    assert n2.find_bbox(_BadGeom()) is None

    # timer decorator branch
    wrapped = n2.timer("demo")(lambda x: x + 1)
    assert wrapped(2) == 3


def test_new02_main_smoke_paths(monkeypatch, tmp_path):
    cfg = {
        "overwrite_existing": False,
        "annotations": {
            "retries": 1,
            "max_workers": 1,
            "overpass_urls": [
                "https://overpass.kumi.systems/api/interpreter",
            ],
            "overpass_pause_seconds": 0.1,
        },
        "paths": {
            "corrected_all_filepath": str(tmp_path / "all.gpkg"),
            "annotations_grid_dir": str(tmp_path / "grid"),
            "annotations_by_osm_dir": str(tmp_path / "osm"),
        },
    }

    grid = gpd.GeoDataFrame(
        {"idx": [1, 2], "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(n2.os, "chdir", lambda path: None)
    monkeypatch.setattr(n2, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(n2, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(n2.gpd, "read_file", lambda path: grid.copy())
    monkeypatch.setattr(n2.os, "makedirs", lambda *args, **kwargs: None)

    class _F:
        def __init__(self, value=None):
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
            fn(*args, **kwargs)
            return _F(None)

    monkeypatch.setattr(n2, "ThreadPoolExecutor", _Exec)
    monkeypatch.setattr(n2, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(n2, "row_operation", lambda *args, **kwargs: None)

    # First pass: one idx already processed, one remains for execution.
    monkeypatch.setattr(n2.os, "listdir", lambda p: ["idx_1_lines.geojson"])
    n2.main()

    # Second pass: all done -> early return branch.
    monkeypatch.setattr(n2.os, "listdir", lambda p: ["idx_1_lines.geojson", "idx_2_polygons.geojson"])
    n2.main()


def test_find_intersection_river_branch_helpers_and_empty_paths(monkeypatch):
    graph = {1: 2, 2: 0, 3: 4, 4: 0}
    assert fir.find_intersection_id(1, 3, graph) is None
    assert fir.find_common_intersection([1, 3], graph) is None

    empty_poly = gpd.GeoDataFrame({"river_list": [], "geometry": []}, geometry="geometry", crs="EPSG:4326")
    rivers = gpd.GeoDataFrame({"HYRIV_ID": [], "NEXT_DOWN": [], "geometry": []}, geometry="geometry", crs="EPSG:4326")
    out = fir.assign_river_juncture(empty_poly.copy(), rivers)
    assert "NXT_DIS" in out.columns


def test_find_intersection_orchestration_and_main_smoke(monkeypatch, tmp_path):
    class _Future:
        def __init__(self, value=None, err=None):
            self._value = value
            self._err = err

        def result(self):
            if self._err is not None:
                raise self._err
            return self._value

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            try:
                return _Future(value=fn(*args, **kwargs))
            except Exception as e:
                return _Future(err=e)

    polygons = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [10],
            "NEXT_DOWN": [0],
            "MAIN_RIV": [100],
            "HYBAS_ID": [1],
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(fir, "estimate_utm_epsg_for_geom", lambda geom: 3857)
    monkeypatch.setattr(fir, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(fir, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(fir, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(fir, "optimize_river_lookup", lambda *args, **kwargs: gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326"))

    out = fir.orchestrate_settlement_river_intersections(polygons.copy(), rivers.copy(), x_distance=1000, max_workers=1)
    assert "river_list" in out.columns
    assert out["river_list"].iloc[0] == []

    # Main smoke path
    cfg = {
        "x_distance": 5000,
        "basin_column_name": "HYBAS_ID",
        "paths": {
            "non_served_above_threshold_outpath": str(tmp_path / "poly.gpkg"),
            "rivershed_output_path": str(tmp_path / "rivers.gpkg"),
            "non_served_nxt_river_outpath": str(tmp_path / "out.gpkg"),
        }
    }
    monkeypatch.setattr(fir.os, "chdir", lambda path: None)
    monkeypatch.setattr(fir, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(fir, "load_config", lambda **kwargs: cfg)

    def _read_file(path, columns=None, **kwargs):
        if str(path).endswith("poly.gpkg"):
            return gpd.GeoDataFrame(
                {"HYBAS_ID": [1], "geometry": [box(0, 0, 1, 1)]},
                geometry="geometry",
                crs="EPSG:4326",
            )
        return gpd.GeoDataFrame(
            {
                "HYRIV_ID": [10],
                "NEXT_DOWN": [0],
                "MAIN_RIV": [100],
                "HYBAS_ID": [1],
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

    monkeypatch.setattr(fir.gpd, "read_file", _read_file)
    monkeypatch.setattr(fir, "orchestrate_settlement_river_intersections", lambda p, r, x_distance, max_workers=1: p.assign(river_list=[[10]]))
    monkeypatch.setattr(fir, "assign_main_riv", lambda p, r: p.assign(MAIN_RIV=[100]))
    monkeypatch.setattr(fir, "orchestrate_river_assignment", lambda p, r, max_workers=1: p.assign(NXT_DIS=[10]))
    monkeypatch.setattr(fir, "ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, path, driver="GPKG", index=False: None)
    monkeypatch.setattr(sys, "argv", ["prog"])

    fir.main()


def test_find_intersection_main_raises_when_missing_crs(monkeypatch, tmp_path):
    cfg = {
        "x_distance": 5000,
        "basin_column_name": "HYBAS_ID",
        "paths": {
            "non_served_above_threshold_outpath": str(tmp_path / "poly.gpkg"),
            "rivershed_output_path": str(tmp_path / "rivers.gpkg"),
            "non_served_nxt_river_outpath": str(tmp_path / "out.gpkg"),
        }
    }
    monkeypatch.setattr(fir.os, "chdir", lambda path: None)
    monkeypatch.setattr(fir, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(fir, "load_config", lambda **kwargs: cfg)

    no_crs_poly = gpd.GeoDataFrame({"HYBAS_ID": [1], "geometry": [box(0, 0, 1, 1)]}, geometry="geometry")
    ok_rivers = gpd.GeoDataFrame(
        {"HYRIV_ID": [1], "NEXT_DOWN": [0], "MAIN_RIV": [1], "HYBAS_ID": [1], "geometry": [LineString([(0, 0), (1, 1)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(fir.gpd, "read_file", lambda path, columns=None, **kwargs: no_crs_poly.copy() if str(path).endswith("poly.gpkg") else ok_rivers.copy())
    monkeypatch.setattr(sys, "argv", ["prog", "--max-workers", "4"])

    with pytest.raises(ValueError):
        fir.main()


def test_find_intersection_main_rejects_non_positive_workers(monkeypatch, tmp_path):
    cfg = {
        "x_distance": 5000,
        "basin_column_name": "HYBAS_ID",
        "paths": {
            "non_served_above_threshold_outpath": str(tmp_path / "poly.gpkg"),
            "rivershed_output_path": str(tmp_path / "rivers.gpkg"),
            "non_served_nxt_river_outpath": str(tmp_path / "out.gpkg"),
        }
    }
    monkeypatch.setattr(fir.os, "chdir", lambda path: None)
    monkeypatch.setattr(fir, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(fir, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(sys, "argv", ["prog", "--max-workers", "0"])
    monkeypatch.setattr(
        fir.gpd,
        "read_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("read_file should not be called for invalid workers")),
    )

    with pytest.raises(ValueError, match="max_workers"):
        fir.main()


def test_download_and_vectorize_boundary_and_repair_branches(monkeypatch, tmp_path):
    # _repair_geometry basic branches
    assert dv._repair_geometry(None) is None
    bad_poly = Point(0, 0).buffer(1).buffer(0)
    assert dv._repair_geometry(bad_poly) is not None

    industrial = gpd.GeoDataFrame({"geometry": [Point(0, 0).buffer(0.2)]}, geometry="geometry", crs="EPSG:4326")
    watershed = gpd.GeoDataFrame({"geometry": [Point(0, 0).buffer(1)]}, geometry="geometry", crs="EPSG:4326")

    with pytest.raises(KeyError):
        dv.add_boundary_info(
            industrial,
            watershed,
            overture_path=str(tmp_path / "ovt.parquet"),
            overture_s3_url="s3://x",
            basin_col="HYBAS_ID",
            sindex_concurrency=False,
        )

    watershed_ok = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "geometry": [Point(0, 0).buffer(1)]},
        geometry="geometry",
        crs="EPSG:3857",
    )
    monkeypatch.setattr(dv.os.path, "exists", lambda path: False)
    monkeypatch.setattr(dv, "download_overture_maps", lambda s3, out: None)
    monkeypatch.setattr(dv, "intersects_with_country_db", lambda gdf, *_args, **_kwargs: gdf.assign(ISO_2=["DE"]))
    monkeypatch.setattr(dv, "intersect_with_polygon_sindex", lambda gdf, *_args, **_kwargs: gdf.assign(HYBAS_ID=[1]))

    enriched = dv.add_boundary_info(
        industrial,
        watershed_ok,
        overture_path=str(tmp_path / "ovt.parquet"),
        overture_s3_url="s3://x",
        basin_col="HYBAS_ID",
        sindex_concurrency=False,
        country_boundary_col="country",
        country_output_col="ISO_2",
    )
    assert "ISO_2" in enriched.columns
    assert "HYBAS_ID" in enriched.columns


def test_download_and_vectorize_main_persist_rasters_path(monkeypatch, tmp_path):
    vectorized_path = str(tmp_path / "industrial_merged.parquet")
    raster_dir = str(tmp_path / "rasters")
    watershed_path = str(tmp_path / "watershed.gpkg")

    cfg = {
        "paths": {
            "industrial_merged_filepath": vectorized_path,
            "watershed": watershed_path,
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://example/overture.parquet",
            "industrial_raster_persistent_dir": raster_dir,
        },
        "overwrite_existing": True,
        "industrial_min_cells": 20,
        "industrial_persist_rasters": True,
        "industrial_simplify_tolerance": 0.01,
        "industrial_batch_size": 2000,
        "industrial_download_chunk_size": 1048576,
        "max_workers": 2,
        "basin_column_name": "HYBAS_ID",
        "sindex_concurrency": False,
        "country_boundary_column": "country",
        "country_output_column": "ISO_2",
        "industrial_zenodo_url": "https://example.com/industrial.zip",
    }

    merged = gpd.GeoDataFrame({"geometry": [Point(0, 0).buffer(1)]}, geometry="geometry", crs="EPSG:4326")
    watershed = gpd.GeoDataFrame({"HYBAS_ID": [1], "geometry": [Point(0, 0).buffer(2)]}, geometry="geometry", crs="EPSG:4326")
    enriched = gpd.GeoDataFrame({"ISO_2": ["DE"], "HYBAS_ID": [1], "geometry": [Point(0, 0).buffer(1)]}, geometry="geometry", crs="EPSG:4326")

    state = {"vectorized_exists": False}
    original_exists = dv.os.path.exists

    def _exists(path):
        if path == vectorized_path:
            return state["vectorized_exists"]
        return original_exists(path)

    monkeypatch.setattr(dv, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(dv, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(dv.os.path, "exists", _exists)
    monkeypatch.setattr(dv, "download_file", lambda *a, **k: None)
    monkeypatch.setattr(dv, "_find_raster_dirs", lambda base: [str(tmp_path / "extracted")])
    monkeypatch.setattr(dv, "_vectorize_and_merge", lambda *args, **kwargs: merged.copy())
    monkeypatch.setattr(dv.gpd, "read_file", lambda path, driver=None: watershed.copy())
    monkeypatch.setattr(dv, "add_boundary_info", lambda *args, **kwargs: enriched.copy())

    class _Zip:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extractall(self, out):
            os.makedirs(out, exist_ok=True)

    monkeypatch.setattr(dv.zipfile, "ZipFile", _Zip)

    removed = []
    replaced = []
    monkeypatch.setattr(dv.os, "remove", lambda path: removed.append(str(path)))
    monkeypatch.setattr(dv.os, "replace", lambda src, dst: replaced.append((str(src), str(dst))) or state.update({"vectorized_exists": True}))

    original_to_parquet = gpd.GeoDataFrame.to_parquet

    def _to_parquet(self, path, index=False, **kwargs):
        state["vectorized_exists"] = True

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", _to_parquet)
        assert dv.main() is True
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", original_to_parquet)

    assert removed == []
    assert replaced == [
        (f"{vectorized_path}.tmp", vectorized_path),
        (f"{vectorized_path}.tmp", vectorized_path),
    ]


def test_add_tile_into_mosaic_clips_on_row_and_column_axes():
    """Overhanging tiles are clipped by rows/cols, not by band/row counts.

    A single-band tile is (1, rows, cols). Trimming with shape[0] (bands) and
    shape[1] (rows) - as the previous arithmetic did - either fails to trim at
    all or trims columns by a row count, shifting tile contents in the mosaic.
    """
    import numpy as np

    from src import download_pop as dp

    mosaic = np.zeros((1, 3, 6), dtype="float32")
    tile = np.arange(1, 13, dtype="float32").reshape(1, 3, 4)

    # Window starts at column 4, so only the tile's first 2 columns fit.
    dp.add_tile_into_mosaic(mosaic, tile, row_off=0, col_off=4, height=3, width=4)

    assert np.allclose(mosaic[0, :, 4:], np.array([[1, 2], [5, 6], [9, 10]]))
    assert np.allclose(mosaic[0, :, :4], 0.0)


def test_add_tile_into_mosaic_clips_overhanging_rows():
    import numpy as np

    from src import download_pop as dp

    mosaic = np.zeros((1, 3, 4), dtype="float32")
    tile = np.ones((1, 3, 4), dtype="float32")

    # Window starts at row 2, so only the tile's first row fits.
    dp.add_tile_into_mosaic(mosaic, tile, row_off=2, col_off=0, height=3, width=4)

    assert np.allclose(mosaic[0, 2, :], 1.0)
    assert np.allclose(mosaic[0, :2, :], 0.0)


def test_mosaic_large_rasters_places_tiles_side_by_side(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from src import download_pop as dp

    def write_tile(path, x_origin, fill):
        profile = {
            "driver": "GTiff", "height": 3, "width": 5, "count": 1,
            "dtype": "float32", "crs": "EPSG:32632",
            "transform": from_origin(x_origin, 3, 1, 1),
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.full((1, 3, 5), fill, dtype="float32"))

    left, right = tmp_path / "left.tif", tmp_path / "right.tif"
    write_tile(left, 0, 1.0)
    write_tile(right, 5, 2.0)

    out = tmp_path / "mosaic.tif"
    dp.mosaic_large_rasters([str(left), str(right)], str(out))

    with rasterio.open(out) as src:
        data = src.read(1)

    assert data.shape == (3, 10)
    assert np.allclose(data[:, :5], 1.0)
    assert np.allclose(data[:, 5:], 2.0)
