from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from exactextract import exact_extract as real_exact_extract
from rasterio.io import MemoryFile
from shapely.geometry import box

from src import add_pop


pytestmark = pytest.mark.integration


def test_intersect_single_file_with_memory_raster_returns_non_negative_population(monkeypatch, tiny_population_array):
    tif_path = "synthetic_population_2020_tile.tif"
    raster = tiny_population_array
    memfile = MemoryFile()
    dataset = memfile.open(
        driver="GTiff",
        height=raster["array"].shape[0],
        width=raster["array"].shape[1],
        count=1,
        dtype=str(raster["array"].dtype),
        transform=raster["transform"],
        crs=raster["crs"],
    )
    dataset.write(raster["array"], 1)
    dataset.close()

    polygons = gpd.GeoDataFrame(
        {
            "poly_id": [1, 2],
            "geometry": [box(-0.10, -0.10, 0.10, 0.10), box(10.0, 10.0, 11.0, 11.0)],
        },
        geometry="geometry",
        crs=raster["crs"],
    )

    original_exists = add_pop.os.path.exists
    monkeypatch.setattr(
        add_pop.os.path,
        "exists",
        lambda path: True if path == tif_path else original_exists(path),
    )
    monkeypatch.setattr(add_pop.rasterio, "open", lambda path: memfile.open())

    def memory_exact_extract(rast, vec, ops, output):
        src = memfile.open()
        try:
            return real_exact_extract(src, vec, ops, output=output)
        finally:
            src.close()

    monkeypatch.setattr(add_pop, "exact_extract", memory_exact_extract)

    try:
        result = add_pop.intersect_single_file(polygons.copy(), [tif_path], all_years=True)
    finally:
        memfile.close()

    assert pd.api.types.is_numeric_dtype(result["2020_zonal_sum"])
    assert result["2020_zonal_sum"].ge(0).fillna(True).all()
    outside_value = result.loc[result["poly_id"] == 2, "2020_zonal_sum"].iloc[0]
    assert pd.isna(outside_value) or outside_value == 0