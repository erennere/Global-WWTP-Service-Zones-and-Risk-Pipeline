"""NEW_04_EXPORTGEOTIFF: one georeferenced image per grid cell."""
import sys

import geopandas as gpd
import pytest
from shapely.geometry import box

from wwtp.annotation_scripts import NEW_04_EXPORTGEOTIFF as n4

pytestmark = pytest.mark.unit


def test_new04_exports_one_georeferenced_patch_per_cell(tmp_path, monkeypatch):
    data = tmp_path / "data"
    grid_dir = data / "annotations" / "GRID"
    grid_dir.mkdir(parents=True)
    gpd.GeoDataFrame({"idx": [7, 8]}, geometry=[box(0, 0, 100, 100), box(100, 0, 300, 100)],
                     crs=3857).to_file(grid_dir / "grids_points.gpkg")
    layer = data / "plants.geojson"
    gpd.GeoDataFrame(geometry=[box(10, 10, 50, 50)], crs=3857).to_crs(4326).to_file(layer)
    cfg = {"export": {"id_field": "idx", "output_width_px": 64, "output_height_px": 32, "dpi": 16,
                      "image_format": "png"},
           "paths": {"corrected_all_filepath": str(data / "points.gpkg"), "annotations_grid_dir": str(grid_dir),
                     "export_layer_filepath": str(layer), "export_patch_dir": str(data / "PATCH")}}
    monkeypatch.setattr(n4, "load_config", lambda **kw: cfg)
    monkeypatch.setattr(sys, "argv", ["prog"])
    n4.main()
    assert sorted(p.name for p in (data / "PATCH").iterdir()) == ["7.png", "7.wld", "8.png", "8.wld"]
    assert [float(v) for v in (data / "PATCH" / "8.wld").read_text().split()] == [200 / 64, 0, 0, -100 / 32, 100, 100]


def test_new04_rejects_unknown_id_field_and_format(tmp_path):
    grid = gpd.GeoDataFrame({"idx": [1]}, geometry=[box(0, 0, 1, 1)])
    with pytest.raises(KeyError):
        n4.export_patches(grid, [], str(tmp_path), "id", 8, 8, 8, "png")
    with pytest.raises(ValueError):
        n4.export_patches(grid, [], str(tmp_path), "idx", 8, 8, 8, "jpg")
