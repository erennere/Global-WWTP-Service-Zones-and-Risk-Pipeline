from __future__ import annotations


import geopandas as gpd
import pytest
from PIL import Image, ImageFont
from shapely.geometry import LineString, box

from wwtp.annotation_scripts import download_bing_annotate as dba


pytestmark = pytest.mark.unit


def test_safe_wkt_and_projection_helpers():
    assert dba.safe_wkt_load(None) is None
    assert dba.safe_wkt_load("bad_wkt") is None
    assert dba.safe_wkt_load("POINT (1 2)").geom_type == "Point"

    px, py = dba.mercator_to_pixel(0, 0, 0, 0, dba.IMAGE_SIZE)
    assert isinstance(px, int) and isinstance(py, int)

    xmin, ymin, xmax, ymax, res = dba.image_bounds_mercator(0.0, 0.0)
    assert xmax > xmin and ymax > ymin and res > 0

    angle = dba.linestring_angle(LineString([(0, 0), (1, 1)]))
    assert angle == pytest.approx(45.0)


def test_split_grids_for_instance_is_deterministic_and_validates():
    grids = list(range(20))
    split_a = dba.split_grids_for_instance(grids, instance_id=1, num_instances=4, split_seed=123)
    split_b = dba.split_grids_for_instance(grids, instance_id=1, num_instances=4, split_seed=123)

    assert split_a == split_b
    with pytest.raises(ValueError):
        dba.split_grids_for_instance(grids, instance_id=0, num_instances=0, split_seed=1)
    with pytest.raises(ValueError):
        dba.split_grids_for_instance(grids, instance_id=5, num_instances=4, split_seed=1)


def test_draw_annotations_and_georef_write(monkeypatch, tmp_path):
    default_font = ImageFont.load_default()
    monkeypatch.setattr(ImageFont, "truetype", lambda *args, **kwargs: default_font)

    image = Image.new("RGB", (256, 256), (10, 10, 10))
    annotations = [
        {"x": 50, "y": 60, "text": "pipe", "style": "line", "angle": 30},
        {"x": 80, "y": 90, "text": "plant", "style": "man_made", "angle": None},
    ]

    out = dba.draw_annotations(image, annotations, fontsize=12)
    assert out.size == image.size

    geotiff_path = tmp_path / "anno.tif"
    dba.georef_write(out.convert("RGB"), 0.0, 0.0, str(geotiff_path))
    assert geotiff_path.exists()


def test_process_bbox_and_parallel_dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr(dba, "get_image", lambda idx, images_dir: Image.new("RGB", (3072, 3072), (0, 0, 0)))
    monkeypatch.setattr(dba, "draw_annotations", lambda image, annotations, fontsize=12: image)
    monkeypatch.setattr(dba, "ensure_output_dir_for_file", lambda path: None)

    bbox = box(0, 0, 100, 100)
    poly = gpd.GeoDataFrame(
        {
            "grid": [1],
            "man_made": ["wastewater_plant"],
            "geometry": [box(10, 10, 20, 20)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    line = gpd.GeoDataFrame(
        {
            "grid": [1],
            "waterway": ["river"],
            "geometry": [LineString([(5, 5), (95, 95)])],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    idx, n, err = dba.process_bbox(
        idx=1,
        bbox_geom=bbox,
        img_idx=42,
        poly_gdf=poly,
        cols=["man_made"],
        line_gdf=line,
        line_cols=["waterway"],
        output_dir=str(tmp_path),
        images_dir=str(tmp_path),
    )
    assert idx == 1
    assert err is None
    assert n >= 1

    bbox_gdf = gpd.GeoDataFrame(
        {"idx": [1], "img_idx": [42], "geometry": [bbox]},
        geometry="geometry",
        crs="EPSG:3857",
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

    monkeypatch.setattr(dba, "ThreadPoolExecutor", _Exec)
    monkeypatch.setattr(dba, "as_completed", lambda futures: list(futures))

    dba.annotate_bboxes_parallel(
        bbox_gdf=bbox_gdf,
        poly_gdf=poly,
        cols=["man_made"],
        line_gdf=line,
        line_cols=["waterway"],
        output_dir=str(tmp_path),
        images_dir=str(tmp_path),
        files=set(),
    )


def test_annotate_bboxes_parallel_requires_idx_and_img_idx_columns(tmp_path):
    bbox_gdf = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 100, 100)]},
        geometry="geometry",
        crs="EPSG:3857",
    )
    empty_poly = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
    empty_line = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")

    with pytest.raises(KeyError, match="required columns"):
        dba.annotate_bboxes_parallel(
            bbox_gdf=bbox_gdf,
            poly_gdf=empty_poly,
            cols=["man_made"],
            line_gdf=empty_line,
            line_cols=["waterway"],
            output_dir=str(tmp_path),
            images_dir=str(tmp_path),
            files=set(),
        )
