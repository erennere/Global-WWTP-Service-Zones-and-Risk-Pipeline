"""Unconnected industrial map: clean input, zoom by area, tiles that decode back, an HTML that points at them."""
import sys
import zipfile

import geopandas as gpd
import mapbox_vector_tile
import pytest
from shapely.geometry import Polygon, box

from src.figures_scripts import interactive_unconnected_industrial_map as im

pytestmark = pytest.mark.unit

BOWTIE = Polygon([(8.0, 50.0), (8.01, 50.01), (8.01, 50.0), (8.0, 50.01)])  # self-intersecting


def _cfg(tmp_path, input_path):
    out = tmp_path / "out"
    return {
        "initial_zoom": 2,
        "vector_tiles": {"enabled": True, "min_zoom": 12, "max_zoom": 13, "min_render_zoom": 12, "extent": 4096,
                         "layer_name": "unconnected_industrial", "simplify_tolerance_deg": 0.0, "zip_tiles": True},
        "paths": {"industrial_unconnected_input": str(input_path),
                  "industrial_unconnected_output": str(out / "prepared.gpkg"),
                  "industrial_unconnected_tiles_dir": str(out / "tiles"),
                  "industrial_unconnected_tiles_zip_filepath": str(out / "tiles.zip"),
                  "industrial_unconnected_html_filepath": str(out / "map" / "map.html")},
    }


def _input(tmp_path):
    path = tmp_path / "industrial.gpkg"
    gpd.GeoDataFrame({"name": ["big", "small", "bowtie"]},
                     geometry=[box(8.1, 50.1, 8.15, 50.13), box(8.3, 50.1, 8.301, 50.101), BOWTIE],
                     crs=4326).to_crs(3035).to_file(path)
    return path


def test_input_is_reprojected_repaired_and_given_ids(tmp_path):
    gdf = im._load_unconnected_industrial_input(_cfg(tmp_path, _input(tmp_path)))
    assert gdf.crs.to_epsg() == 4326 and gdf.geometry.is_valid.all()
    assert gdf["source_id"].tolist() == [0, 1, 2] and gdf["name"].tolist() == ["big", "small", "bowtie"]
    assert gdf["area_proxy_m2"].iloc[0] > gdf["area_proxy_m2"].iloc[1] > 0


def test_missing_or_empty_input_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        im._load_unconnected_industrial_input(_cfg(tmp_path, tmp_path / "nope.gpkg"))
    empty = tmp_path / "empty.gpkg"
    gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326).to_file(empty)
    with pytest.raises(ValueError):
        im._load_unconnected_industrial_input(_cfg(tmp_path, empty))


def test_larger_polygons_appear_at_lower_zoom():
    gdf = gpd.GeoDataFrame({"area_proxy_m2": [1e2, 1e4, 1e6]}, geometry=[box(0, 0, 1, 1)] * 3)
    assert im._assign_polygon_min_zoom(gdf, 10, 16)["tile_min_zoom"].tolist() == [16, 13, 10]
    same = gpd.GeoDataFrame({"area_proxy_m2": [5.0, 5.0]}, geometry=[box(0, 0, 1, 1)] * 2)
    assert im._assign_polygon_min_zoom(same, 10, 16)["tile_min_zoom"].tolist() == [10, 10]


def test_polygon_parts_drop_lines_and_flatten_collections():
    from shapely.geometry import GeometryCollection, LineString
    poly = box(0, 0, 1, 1)
    assert im._polygon_parts(GeometryCollection([poly, LineString([(0, 0), (1, 1)])])) == [poly]
    assert im._polygon_parts(LineString([(0, 0), (1, 1)])) == [] and im._polygon_parts(None) == []


def test_main_writes_decodable_tiles_zip_and_html(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, _input(tmp_path))
    monkeypatch.setattr(im, "load_config", lambda **kw: cfg)
    monkeypatch.setattr(sys, "argv", ["prog"])
    im.main()
    out = tmp_path / "out"
    tiles = sorted((out / "tiles").rglob("*.pbf"))
    assert tiles and {p.relative_to(out / "tiles").parts[0] for p in tiles} <= {"12", "13"}
    ids = set()
    for p in tiles:
        layer = mapbox_vector_tile.decode(p.read_bytes())["unconnected_industrial"]
        ids |= {f["properties"]["source_id"] for f in layer["features"]}
    assert ids == {0, 1, 2}
    with zipfile.ZipFile(out / "tiles.zip") as zf:
        assert len(zf.namelist()) == len(tiles)
    html = (out / "map" / "map.html").read_text()
    assert "../tiles/{z}/{x}/{y}.pbf" in html and "__VECTOR_TILE" not in html
    assert len(gpd.read_file(out / "prepared.gpkg")) == 3


def test_main_requires_vector_tiles(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, _input(tmp_path))
    cfg["vector_tiles"]["enabled"] = False
    monkeypatch.setattr(im, "load_config", lambda **kw: cfg)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(ValueError, match="vector_tiles.enabled"):
        im.main()
