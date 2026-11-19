"""Tests for download_bing_annotate (moved from the coverage-campaign files)."""

import geopandas as gpd
import pandas as pd
import pytest
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, box
from wwtp.annotation_scripts import download_bing_annotate as dba
import argparse
import os
import runpy
from shapely.geometry import Point, Polygon
from wwtp import starter

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
    monkeypatch.setattr(dba, "LABEL_FONT_PX", 16)
    monkeypatch.setattr(dba, "GEOREFERENCED", True)
    monkeypatch.setattr(dba, "draw_annotations", lambda image, annotations, **kw: image)

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


def test_download_bing_bbox_error_branches(monkeypatch, tmp_path):
    bbox = box(0, 0, 10, 10)
    poly = gpd.GeoDataFrame({"grid": [1], "geometry": [box(1, 1, 2, 2)]}, geometry="geometry", crs="EPSG:3857")
    line = gpd.GeoDataFrame(
        {"grid": [1], "geometry": [LineString([(1, 1), (2, 2)])]}, geometry="geometry", crs="EPSG:3857"
    )

    monkeypatch.setattr(dba, "get_image", lambda img_idx, images_dir: None)
    idx, n, err = dba.process_bbox(1, bbox, 77, poly, ["man_made"], line, ["waterway"], str(tmp_path), str(tmp_path))
    assert idx == 1 and n == 0 and "Image not found" in err

    monkeypatch.setattr(dba, "get_image", lambda img_idx, images_dir: object())
    idx2, n2, err2 = dba.process_bbox(2, bbox, 1, poly, ["man_made"], line, ["waterway"], str(tmp_path), str(tmp_path))
    assert idx2 == 2 and n2 == 0 and err2 is not None


def _osm_world(tmp_path, monkeypatch, n=6):
    """``n`` images at 5 deg E, 5 deg N (512 px of 0.6 m) with a wastewater plant polygon on each tile."""
    osm_dir, out_dir = tmp_path / "osm", tmp_path / "out"
    osm_dir.mkdir()
    monkeypatch.setattr(dba, "IMAGE_SIZE", [512, 512])
    monkeypatch.setattr(dba, "IMAGERY_REFERENCE_TILE_SIZE", 512)
    monkeypatch.setattr(dba, "get_image", lambda img_idx, images_dir: Image.new("RGB", (512, 512), "black"))
    square = Polygon([(4.9999, 4.9999), (5.0001, 4.9999), (5.0001, 5.0001), (4.9999, 5.0001)])
    for idx in range(n):
        gpd.GeoDataFrame({"man_made": ["wastewater_plant"], "id": ["11"], "ACRES": ["2"], "Acres": ["1"]},
                         geometry=[square], crs=4326).to_file(osm_dir / f"idx_{idx}_polygons.geojson")  # fmt: skip
    centre = gpd.GeoSeries([Point(5, 5)], crs=4326).to_crs(3857).iloc[0]
    images = gpd.GeoDataFrame({"idx": range(n), "img_idx": range(n), "osm_tiles": [(i,) for i in range(n)]},
                              geometry=[centre] * n, crs=3857)  # fmt: skip
    shared = dict(osm_dir=str(osm_dir), osm_files=set(os.listdir(osm_dir)), poly_cols=["man_made", "id"],
                  line_cols=[], poly_labels=["man_made"], line_labels=[], output_dir=str(out_dir),
                  images_dir=str(tmp_path))  # fmt: skip
    return images, shared, osm_dir, out_dir


def test_annotate_images_never_stops(monkeypatch, tmp_path):
    """Images in parallel processes, each with the OSM features of its tiles. A missing image or unreadable
    OSM files are reported for that image; all others are written and nothing is raised."""
    images, shared, osm_dir, out_dir = _osm_world(tmp_path, monkeypatch)
    report = dba.annotate_images(images, 3, **shared)
    assert sorted(p.name for p in out_dir.iterdir()) == [f"bbox_{i}.png" for i in range(6)]
    assert report["labels"].tolist() == [1] * 6 and report["osm_features_on_image"].tolist() == [1] * 6
    assert report["error"].isna().all()
    assert len(dba.annotate_images(images.iloc[:2], 1, **shared)) == 2  # one process: no pool

    for p in out_dir.iterdir():
        p.unlink()
    (osm_dir / "idx_4_polygons.geojson").write_text("not json")
    no_image = lambda img_idx, images_dir: None if img_idx == 5 else Image.new("RGB", (512, 512))  # noqa: E731
    monkeypatch.setattr(dba, "get_image", no_image)
    report = dba.annotate_images(images, 3, **shared).set_index("idx")
    assert report.loc[4, "error"].startswith("OSM files:") and "Image not found" in report.loc[5, "error"]
    assert sorted(p.name for p in out_dir.iterdir()) == [f"bbox_{i}.png" for i in range(4)]


def test_a_feature_of_two_tiles_counts_once(monkeypatch, tmp_path):
    images, shared, _, _ = _osm_world(tmp_path, monkeypatch, n=2)
    report = dba.annotate_images(images.iloc[:1].assign(osm_tiles=[(0, 1)]), 1, **shared)
    assert report["osm_features"].tolist() == [1] and report["labels"].tolist() == [1]


def test_osm_tiles_are_found_by_place(monkeypatch):
    """The OSM files of an image come from the tile at its place: its own tile when that is centred on it,
    else another tile centred on it (tiles numbered differently), else all tiles overlapping it."""
    monkeypatch.setattr(dba, "IMAGE_SIZE", [3072, 3072])
    monkeypatch.setattr(dba, "IMAGERY_REFERENCE_TILE_SIZE", 256)  # 1.194 m/px: the image is NEW_01's tile
    pts = gpd.GeoSeries([Point(5, 5), Point(6, 6), Point(7, 7), Point(30, 30)], crs=4326).to_crs(3857)
    images = gpd.GeoDataFrame({"idx": [1, 2, 3, 4]}, geometry=pts.to_numpy(), crs=3857)
    frame = lambda p, dx=0.0: dba.image_frame(p.x + dx, p.y)  # noqa: E731
    half = dba.image_frame(0, 0).bounds[2]
    tiles = gpd.GeoDataFrame({"idx": [1, 2, 7, 8, 9, 10]}, crs=3857, geometry=[
        frame(pts[0]),               # image 1's own tile
        frame(pts[0], 5 * half),     # tile "2" elsewhere: image 2 is at tile 7
        frame(pts[1]),
        frame(pts[2], -half),        # image 3: two tiles half off either side, together all of it
        frame(pts[2], half),
        frame(pts[3]),               # centred on image 4, but without OSM files
    ])  # fmt: skip
    files = {"idx_1_polygons.geojson", "idx_2_lines.geojson", "idx_7_empty.txt", "idx_8_lines.geojson",
             "idx_9_polygons.geojson"}  # fmt: skip
    m = dba.match_osm_tiles(images, tiles, files, 10)
    assert m["osm_source"].tolist() == ["own_tile", "tile_at_image", "overlapping_tiles", "none"]
    assert m["osm_tiles"].tolist() == [(1,), (7,), (8, 9), ()]
    assert m["osm_coverage"].tolist() == [1.0, 1.0, 1.0, 0.0]
    half_cover = dba.match_osm_tiles(images.iloc[[2]], tiles[tiles["idx"] != 9], files, 10)
    assert half_cover["osm_tiles"].tolist() == [(8,)] and half_cover["osm_coverage"].tolist() == [0.5]


def test_download_bing_main_smoke(monkeypatch, tmp_path, capsys):
    """The new plants' set as it is: ref.geojson sorted west to east, <n>.png the image of the feature whose
    idx is n; one plant's tile numbered differently. Each image gets the OSM features of its own place."""
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
        "max_image_offset_m": 10,
        "annotations": {
            "cell_size": 3072,
            "factor": 1.194,
            "image_size_px": 3072,
            "zoom_level": 17,
            "base_z17_resolution": 1.1943285669555664,
            "max_workers": 1,
            "georeferenced": False,
            "label_font_px": 8,
            "label_halo_px": 1,
            "dpi": 300,
            "target_size_px": 1024,
            "mercator_tile_size_px": 256,
            "imagery_reference_tile_size_px": 256,
            "image_names": "ref_idx",
            "earth_radius_m": 6378137,
        },
        "paths": {
            "annotations_images_dir": str(images_dir),
            "annotations_grid_dir": str(grid_dir),
            "annotations_temp_parquet_dir": str(tmp_path / "tmp_parquet"),
            "annotations_by_osm_dir": str(osm_dir),
            "annotation_points_filepath": str(tmp_path / "all.csv"),
            "annotated_images_output_dir": str(out_dir),
            "images_ref_list": str(tmp_path / "imaged_list.csv"),
            "match_report_dir": str(tmp_path / "report"),
        },
    }
    pd.DataFrame({"idx": [7, 1]}).to_csv(tmp_path / "imaged_list.csv", index=False)  # list row 0 = plant 7

    monkeypatch.setattr(starter, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(starter, "load_config", lambda **kwargs: cfg)

    # ref.geojson west to east: feature 0 is list row 1 (plant 1 at 5 E), feature 1 is list row 0 (plant 7 at 6 E)
    points = gpd.GeoDataFrame({"idx": [1, 0]}, geometry=[Point(5, 5), Point(6, 5)], crs=4326)
    plants = gpd.GeoDataFrame({"idx": [7, 1]}, geometry=[Point(6, 5), Point(5, 5)], crs=4326)
    # tile 1 at plant 1; plant 7's tile carries another number (99)
    grids = gpd.GeoDataFrame({"idx": [1, 99]}, geometry=[box(4.98, 4.98, 5.02, 5.02), box(5.98, 4.98, 6.02, 5.02)],
                             crs=4326)  # fmt: skip

    # Tile 1's OSM files as osm_planet wrote them on the cluster, with the two things that stopped the overlay
    # there (18 Nov): tag columns alike but for case (DuckDB: "duplicate column name") and a polygon whose ring
    # crosses itself (GEOS: "TopologyException: side location conflict").
    square = Polygon([(4.9999, 4.9999), (5.0001, 4.9999), (5.0001, 5.0001), (4.9999, 5.0001)])
    bow_tie = Polygon([(4.9999, 4.9999), (5.0001, 5.0001), (5.0001, 4.9999), (4.9999, 5.0001)])
    polygons = {"man_made": ["wastewater_plant", "nan"], "Acres": ["3", "nan"], "ID": ["a", "nan"], "id": ["1", "2"],
                "landuse": ["nan", "industrial"], "ACRES": ["nan", "4"]}  # fmt: skip
    gpd.GeoDataFrame(polygons, geometry=[square, bow_tie], crs=4326).to_file(osm_dir / "idx_1_polygons.geojson")
    lines = {"man_made": ["pipeline"], "id": ["3"], "Id": ["b"]}
    river = LineString([(4.9995, 5.0), (5.0005, 5.0)])
    gpd.GeoDataFrame(lines, geometry=[river], crs=4326).to_file(osm_dir / "idx_1_lines.geojson")
    tank = Polygon([(5.9999, 4.9999), (6.0001, 4.9999), (6.0001, 5.0001), (5.9999, 5.0001)])
    gpd.GeoDataFrame({"man_made": ["storage_tank"], "id": ["4"]}, geometry=[tank], crs=4326).to_file(
        osm_dir / "idx_99_polygons.geojson"
    )
    real_read_file = gpd.read_file

    def _read_file(path, **kwargs):
        p = str(path)
        if p.endswith("ref.geojson"):
            return points.copy()
        if os.path.basename(p) == "all.csv":
            return plants.copy()
        if p.startswith(str(osm_dir)):
            return real_read_file(path, **kwargs)
        return grids.copy()

    monkeypatch.setattr(gpd, "read_file", _read_file)

    # 1.png shows plant 1 (red), 0.png plant 7 (blue)
    dba.Image.new("RGB", (3072, 3072), "red").save(images_dir / "1.png")
    dba.Image.new("RGB", (3072, 3072), "blue").save(images_dir / "0.png")

    runpy.run_module("wwtp.annotation_scripts.download_bing_annotate", run_name="__main__")
    err = capsys.readouterr().err
    assert "bbox 1 done (3 tags)" in err and "bbox 7 done (1 tags)" in err
    assert Image.open(out_dir / "bbox_1.png").convert("RGB").getpixel((5, 5)) == (255, 0, 0)
    assert Image.open(out_dir / "bbox_7.png").convert("RGB").getpixel((5, 5)) == (0, 0, 255)
    (report_file,) = (tmp_path / "report").iterdir()  # <last two folders of the images dir>_shard<k>.csv
    assert report_file.name == f"{tmp_path.name}-images_shard0.csv"
    report = pd.read_csv(report_file, dtype={"osm_tiles": str}).set_index("idx")
    assert report.loc[1, ["image", "osm_source", "osm_tiles", "labels"]].tolist() == ["1.png", "own_tile", "1", 3]
    assert report.loc[7, ["image", "osm_source", "osm_tiles", "labels"]].tolist() == ["0.png", "tile_at_image", "99", 1]
    assert report["image_to_plant_m"].max() < 0.01 and report["error"].isna().all()


def test_ref_geojson_rows_become_plant_numbers(tmp_path):
    """Sukanya's ref.geojson numbers the plants by their row in the list she got (cluster, 18 Nov: 13,932
    of 17,519 images "not in the plant list")."""
    from wwtp.annotation_scripts.images_ref import plant_idx_of_rows

    pd.DataFrame({"idx": [47999, 8869, 64862]}).to_csv(tmp_path / "list.csv", index=False)
    ref = gpd.GeoDataFrame({"idx": [2, 0]}, geometry=[Point(2.77, 41.76), Point(45.07, 12.89)], crs=4326)
    out = plant_idx_of_rows(ref, tmp_path / "list.csv")
    assert out["idx"].tolist() == [64862, 47999] and out["list_row"].tolist() == [2, 0]
    assert ref["idx"].tolist() == [2, 0]  # the input stays as it is
    with pytest.raises(ValueError, match="1 images of ref.geojson have a row outside the 3 rows"):
        plant_idx_of_rows(ref.assign(idx=[3, 0]), tmp_path / "list.csv")


def test_the_overlay_stops_when_the_images_do_not_fit(tmp_path):
    """Images made for another plant list, or of another size, stop the overlay before it draws."""
    from wwtp.annotation_scripts.images_ref import check_image_size, check_images_ref

    plants = gpd.GeoDataFrame({"idx": [1, 2]}, geometry=[Point(8, 47), Point(9, 47)], crs="EPSG:4326")
    ref = gpd.GeoDataFrame({"idx": [2, 1]}, geometry=[Point(9, 47), Point(8, 47.0001)], crs="EPSG:4326")
    check_images_ref(ref.to_crs(3857), plants, 20, "ref.geojson")  # 11 m: the same plant
    with pytest.raises(ValueError, match="1 of 2 images are farther than 5 m"):
        check_images_ref(ref, plants, 5, "ref.geojson")
    with pytest.raises(ValueError, match="not in the plant list"):
        check_images_ref(gpd.GeoDataFrame({"idx": [3]}, geometry=[Point(0, 0)], crs=4326), plants, 5, "ref")

    dba.Image.new("RGB", (512, 512), "black").save(tmp_path / "1.png")
    with pytest.raises(ValueError, match="is 512 x 512 pixels; the overlay expects 3072 x 3072"):
        check_image_size(str(tmp_path), [0, 1], 3072)
    check_image_size(str(tmp_path), [1], 512)
    with pytest.raises(FileNotFoundError):
        check_image_size(str(tmp_path), [5], 512)
