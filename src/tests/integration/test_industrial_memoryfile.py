from __future__ import annotations

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from src.industrial_analysis import download_and_vectorize


pytestmark = pytest.mark.integration


def test_vectorize_raster_file_filters_polygons_below_min_cells(monkeypatch):
    raster = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype="uint8",
    )
    memfile = MemoryFile()
    dataset = memfile.open(
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="uint8",
        transform=from_origin(0, 30, 10, 10),
        crs="EPSG:3857",
    )
    dataset.write(raster, 1)
    dataset.close()

    monkeypatch.setattr(download_and_vectorize.rasterio, "open", lambda path: memfile.open())

    try:
        result = download_and_vectorize.vectorize_raster_file("synthetic.tif", crs="EPSG:3857", min_cells=2)
    finally:
        memfile.close()

    assert len(result) == 1
    assert result.geometry.iloc[0].area >= 200.0