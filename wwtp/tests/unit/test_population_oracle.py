"""Zonal population sums against the raster itself.

- polygons that tile the raster add up to its total (nodata excluded)
- a polygon equal to one cell gets that cell's value; half a cell gets half
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from wwtp.add_pop import intersect_single_file

pytestmark = pytest.mark.unit

X0, Y0, RES, N = 8.0, 50.1, 0.01, 10
NODATA = -99999.0


@pytest.fixture()
def raster(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 50, size=(N, N)).astype("float32")
    data[:2, :2] = NODATA  # a lake
    path = tmp_path / "XX_pop_2020_CN_100m.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=N,
        width=N,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(X0, Y0, RES, RES),
        nodata=NODATA,
    ) as dst:
        dst.write(data, 1)
    return path, data


def _cell(i, j):
    """Bounds of raster cell (row i, column j)."""
    return box(X0 + j * RES, Y0 - (i + 1) * RES, X0 + (j + 1) * RES, Y0 - i * RES)


def test_tiles_add_up_to_the_raster_total(raster):
    path, data = raster
    half = N // 2
    tiles = [
        box(X0 + a * half * RES, Y0 - (b + 1) * half * RES, X0 + (a + 1) * half * RES, Y0 - b * half * RES)
        for a in (0, 1)
        for b in (0, 1)
    ]
    out = intersect_single_file(gpd.GeoDataFrame({"ISO_2": ["XX"] * 4}, geometry=tiles, crs=4326), [str(path)])
    valid = data[data != NODATA]
    assert out["2020_zonal_sum"].sum() == pytest.approx(float(valid.sum()), rel=1e-5)


def test_one_cell_and_half_a_cell(raster):
    path, data = raster
    cell = _cell(4, 6)
    minx, miny, maxx, maxy = cell.bounds
    half = box(minx, miny, (minx + maxx) / 2, maxy)
    out = intersect_single_file(gpd.GeoDataFrame({"ISO_2": ["XX", "XX"]}, geometry=[cell, half], crs=4326), [str(path)])
    assert out["2020_zonal_sum"].iloc[0] == pytest.approx(float(data[4, 6]), rel=1e-5)
    assert out["2020_zonal_sum"].iloc[1] == pytest.approx(float(data[4, 6]) / 2, rel=1e-4)


def test_nodata_counts_as_nobody(raster):
    path, _ = raster
    lake = box(X0, Y0 - 2 * RES, X0 + 2 * RES, Y0)
    out = intersect_single_file(gpd.GeoDataFrame({"ISO_2": ["XX"]}, geometry=[lake], crs=4326), [str(path)])
    assert out["2020_zonal_sum"].iloc[0] == pytest.approx(0.0, abs=1e-9)
