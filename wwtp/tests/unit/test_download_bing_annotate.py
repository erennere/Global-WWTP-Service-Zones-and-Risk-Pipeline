"""Tests for download_bing_annotate (moved from the coverage-campaign files)."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw, ImageFont
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from wwtp.annotation_scripts import download_bing_annotate as dba
from wwtp.figures_scripts import piechart_figure as pf
from wwtp.industrial_analysis import download_and_vectorize as dav
import argparse
import os
import runpy
import sys
import types
import duckdb
from shapely.geometry import LineString, Point, box
from wwtp import download_pop as dp
from wwtp.annotation_scripts import NEW_01_GENERATEGRIDS as n1
from wwtp.annotation_scripts import NEW_02_EXTRACTOSMDATAFULL_GEOJSON as n2
from wwtp.annotation_scripts import NEW_03_WASTEWATERJOIN_GEOJSON as ww3
from wwtp import starter
from wwtp.pop_at_risk_river_calculations import create_rasters as cr
from wwtp.pop_at_risk_river_calculations import find_intersection_river as fir
from wwtp.pop_at_risk_river_calculations import impact_polygons_pop as ipp
from wwtp.industrial_analysis import download_and_vectorize as dv

pytestmark = pytest.mark.unit


def test_image_helpers_and_drawing(tmp_path):
    in_dir = tmp_path / "images"
    in_dir.mkdir(parents=True, exist_ok=True)
    file_img = in_dir / "3.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(file_img)
    assert dba.get_image(3, str(in_dir)) is not None
    assert dba.get_image(4, str(in_dir)) is None

    canvas = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    dba.draw_text_with_padding(draw, (150, 150), "txt", font, "white", "black", pad=1)
    dba.draw_rotated_text_with_padding(canvas, (150, 150), "rot", 35, font, "white", "blue", pad=1)

    angle = dba.linestring_angle(LineString([(0, 0), (2, 0)]))
    assert angle == pytest.approx(0.0)


def test_bing_process_bbox_georeferenced_output(tmp_path, monkeypatch):
    monkeypatch.setattr(dba, "get_image", lambda idx, images_dir: Image.new("RGB", (3072, 3072), (0, 0, 0)))
    monkeypatch.setattr(dba, "FONTSIZE", 16)
    monkeypatch.setattr(dba, "GEOREFERENCED", True)
    monkeypatch.setattr(dba, "draw_annotations", lambda image, annotations, fontsize=12: image)

    bbox_geom = box(0, 0, 100, 100)
    poly_gdf = gpd.GeoDataFrame(
        {
            "grid": [1],
            "man_made": ["wastewater_plant"],
            "geometry": [box(10, 10, 20, 20)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    line_gdf = gpd.GeoDataFrame(
        {
            "grid": [1],
            "waterway": ["river"],
            "geometry": [LineString([(5, 5), (95, 95)])],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    idx, n, err = dba.process_bbox(
        1,
        bbox_geom,
        11,
        poly_gdf,
        ["man_made"],
        line_gdf,
        ["waterway"],
        str(tmp_path),
        str(tmp_path),
    )
    assert idx == 1
    assert err is None
    assert n >= 1
    assert (tmp_path / "bbox_1.tif").exists()


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
        size_measure=None,
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
    runpy.run_module("wwtp.annotation_scripts.download_bing_annotate", run_name="__main__")
