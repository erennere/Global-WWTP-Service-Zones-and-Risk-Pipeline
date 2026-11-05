from __future__ import annotations


import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw, ImageFont
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from src.annotation_scripts import download_bing_annotate as dba
from src.figures_scripts import piechart_figure as pf
from src.industrial_analysis import download_and_vectorize as dav


pytestmark = pytest.mark.unit


def test_download_file_stream_and_vectorize_raster(tmp_path, monkeypatch):
    class _Resp:
        headers = {"content-length": "12"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"abc"
            yield b"def"
            yield b"ghi"
            yield b"jkl"

    class _Session:
        def get(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(dav, "requests_session_with_retries", lambda *a, **k: _Session())

    out = tmp_path / "download.bin"
    dav.download_file("https://example.test/file", str(out), chunk_size=1024)
    assert out.exists()
    assert out.read_bytes() == b"abcdefghijkl"

    raster_path = tmp_path / "industrial.tif"
    import rasterio

    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[1:5, 1:5] = 1
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs="EPSG:4326",
        transform=from_origin(-1, 1, 0.01, 0.01),
    ) as dst:
        dst.write(arr, 1)

    gdf = dav.vectorize_raster_file(str(raster_path), crs="EPSG:4326", min_cells=2)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) >= 1


def test_vectorize_rasters_parallel_and_merge(tmp_path):
    import rasterio

    for i in range(2):
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[2:8, 2:8] = 1
        rp = tmp_path / f"tile_{i}.tif"
        with rasterio.open(
            rp,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            crs="EPSG:4326",
            transform=from_origin(-1 + i * 0.2, 1, 0.01, 0.01),
        ) as dst:
            dst.write(arr, 1)

    gdfs = dav.vectorize_rasters_parallel(str(tmp_path), max_workers=2, crs="EPSG:4326", min_cells=3)
    assert len(gdfs) == 2

    merged = dav.merge_geodataframes(gdfs, simplify_tolerance=0.0001, max_workers=2)
    assert isinstance(merged, gpd.GeoDataFrame)
    assert len(merged) >= 1


def test_geometry_repair_and_overlap_dissolve_paths():
    # self-intersecting bow-tie polygon
    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    repaired = dav._repair_geometry(invalid)
    assert repaired is not None

    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                box(0, 0, 1, 1),
                box(0.5, 0.5, 1.5, 1.5),
                box(3, 3, 4, 4),
            ]
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    dissolved = dav._dissolve_by_overlap_groups(gdf)
    assert isinstance(dissolved, gpd.GeoDataFrame)
    assert len(dissolved) >= 1


def test_add_boundary_info_preserves_original_geometry(monkeypatch):
    industrial = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 1, 1), box(2, 2, 3, 3)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    watersheds = gpd.GeoDataFrame(
        {"HYBAS_ID": [1, 2], "geometry": [box(-1, -1, 1.2, 1.2), box(1.8, 1.8, 3.2, 3.2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(dav.os.path, "exists", lambda p: True)
    monkeypatch.setattr(dav, "download_overture_maps", lambda *args, **kwargs: None)

    def _fake_country(df, *_args, **_kwargs):
        out = df.copy()
        out["ISO_2"] = ["DE", "FR"]
        return out

    def _fake_basin(df, ws, basin_col, **kwargs):
        out = df.copy()
        out[basin_col] = [1, 2]
        return out

    monkeypatch.setattr(dav, "intersects_with_country_db", _fake_country)
    monkeypatch.setattr(dav, "intersect_with_polygon_sindex", _fake_basin)

    enriched = dav.add_boundary_info(
        industrial,
        watersheds,
        overture_path="dummy.parquet",
        overture_s3_url="s3://dummy",
        basin_col="HYBAS_ID",
        sindex_concurrency=False,
        country_boundary_col="country",
        country_output_col="ISO_2",
    )

    assert "ISO_2" in enriched.columns
    assert "HYBAS_ID" in enriched.columns
    assert enriched.geometry.equals(industrial.geometry)


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


def test_piechart_aggregation_and_scaling_paths():
    df = pd.DataFrame(
        {
            "country": ["DE", "DE", "FR", "FR"],
            "value": [10, 30, 20, 40],
            "IND/RES": [True, False, True, False],
        }
    )

    pop_agg = pf.aggregate_by_country(df, "country", "value", industrial_column=None, is_pop=True)
    assert "value_sum" in pop_agg.columns

    type_agg = pf.aggregate_by_country(df, "country", "value", industrial_column="IND/RES", is_pop=False)
    assert any(c.startswith("IND_") for c in type_agg.columns)
    assert any(c.startswith("RES_") for c in type_agg.columns)

    sizes = [pf.calculate_size(v, min_value=10, max_value=100, min_size=0.1, max_size=1.0, scale="linear") for v in [10, 30, 90]]
    assert all(isinstance(s, float) for s in sizes)

    with pytest.raises(ValueError):
        pf.calculate_size(10, 1, 2, 0.1, 1.0, scale="bad")


def test_piechart_position_and_rounding_helpers():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    mpoly = MultiPolygon([poly, box(2, 2, 5, 5)])

    x1, y1 = pf.get_pos(poly)
    x2, y2 = pf.get_pos(mpoly)
    assert isinstance(x1, float)
    assert isinstance(y2, float)

    labels = pf.round_numbers(np.array([10, 100, 1000, np.nan]), [1, 2, 3, 4])
    assert len(labels) == 4


def test_piechart_percentage_and_zonal_resolution_helpers():
    df = pd.DataFrame({"population_served": [50, 30], "population_total": [100, 60]})
    col = pf.ensure_population_percentage_column(df, preferred_col="population_served_index", zonal_sum_col="2024_zonal_sum")
    assert col == "population_served_index"
    assert np.isclose(df[col].iloc[0], 0.5)

    df2 = pd.DataFrame({"2023_zonal_sum": [10], "2024_zonal_sum": [11], "population_total": [20]})
    chosen = pf.resolve_zonal_sum_columns(df2, preferred="missing_zonal_sum")
    assert chosen == "2024_zonal_sum"


def test_plot_splitted_piechart_executes(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    pf.plot_splitted_piechart(
        dist_tag1=[20, 80],
        dist_tag2=[45, 55],
        ax=ax,
        size_tag1=100,
        size_tag2=50,
        min_size=5,
        labels=True,
        labels_text=["A", "B", "C"],
    )
    out = tmp_path / "pie.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
