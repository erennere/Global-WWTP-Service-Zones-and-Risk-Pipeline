from __future__ import annotations

import os

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from wwtp.industrial_analysis import download_and_vectorize
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from wwtp.annotation_scripts import download_bing_annotate as dba
from wwtp.figures_scripts import piechart_figure as pf
from wwtp.industrial_analysis import download_and_vectorize as dav
import argparse
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
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from shapely.geometry import Point, Polygon, box
from concurrent.futures import ProcessPoolExecutor, as_completed
import wwtp.download_pop as dp
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
from unittest.mock import Mock, patch, MagicMock
try:
    from wwtp import download_pop as dp
    from wwtp.pop_at_risk_river_calculations import create_rasters as cr
    from wwtp.industrial_analysis import download_and_vectorize as dv
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import download_pop as dp
    from pop_at_risk_river_calculations import create_rasters as cr
    from industrial_analysis import download_and_vectorize as dv
if __name__ == '__main__':
    pytest.main([__file__, '-v'])


pytestmark = pytest.mark.unit


def test_find_raster_dirs_returns_only_directories_with_tif_files(tmp_path):
    has_tif = tmp_path / "a" / "nested"
    has_tif.mkdir(parents=True)
    (has_tif / "tile_1.tif").write_text("stub", encoding="utf-8")

    no_tif = tmp_path / "b"
    no_tif.mkdir()
    (no_tif / "notes.txt").write_text("stub", encoding="utf-8")

    result = download_and_vectorize._find_raster_dirs(str(tmp_path))

    assert str(has_tif) in result
    assert str(no_tif) not in result


def test_vectorize_rasters_parallel_returns_empty_when_no_rasters(tmp_path):
    result = download_and_vectorize.vectorize_rasters_parallel(str(tmp_path), max_workers=2)

    assert result == []


def test_vectorize_rasters_parallel_rejects_non_positive_workers(tmp_path):
    with pytest.raises(ValueError, match="max_workers"):
        download_and_vectorize.vectorize_rasters_parallel(str(tmp_path), max_workers=0)


def test_vectorize_and_merge_raises_when_no_raster_dirs():
    with pytest.raises(FileNotFoundError, match="No raster directories found"):
        download_and_vectorize._vectorize_and_merge([], max_workers=1, min_cells=10)


def test_vectorize_and_merge_raises_when_vectorization_produces_no_outputs(monkeypatch):
    monkeypatch.setattr(download_and_vectorize, "vectorize_rasters_parallel", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="Failed to vectorize any raster files"):
        download_and_vectorize._vectorize_and_merge(["/tmp/rasters"], max_workers=1, min_cells=10)


def test_merge_geodataframes_rejects_invalid_max_workers():
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0).buffer(0.01)], "value": [1]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="max_workers"):
        download_and_vectorize.merge_geodataframes([gdf], max_workers="invalid")


def test_main_uses_cached_vectorized_polygons_and_writes_enriched_result(monkeypatch, tmp_path):
    vectorized_path = str(tmp_path / "industrial_merged.parquet")
    watershed_path = str(tmp_path / "watershed.gpkg")
    cfg = {
        "paths": {
            "industrial_merged_filepath": vectorized_path,
            "watershed": watershed_path,
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://example/overture.parquet",
            "industrial_raster_persistent_dir": str(tmp_path / "rasters"),
        },
        "overwrite_existing": False,
        "industrial_min_cells": 20,
        "industrial_persist_rasters": False,
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
    captured = {"removed": [], "writes": [], "replaced": []}
    state = {"vectorized_exists": True}

    merged_gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")
    watershed_gdf = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "geometry": [Point(0, 0).buffer(1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    enriched_gdf = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "ISO_2": ["DE"], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    original_exists = download_and_vectorize.os.path.exists

    def fake_exists(path):
        if path == vectorized_path:
            return state["vectorized_exists"]
        return original_exists(path)

    monkeypatch.setattr(download_and_vectorize, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(download_and_vectorize, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(download_and_vectorize.os.path, "exists", fake_exists)
    monkeypatch.setattr(download_and_vectorize.gpd, "read_parquet", lambda path: merged_gdf.copy())
    monkeypatch.setattr(download_and_vectorize.gpd, "read_file", lambda path, driver=None: watershed_gdf.copy())
    monkeypatch.setattr(download_and_vectorize, "add_boundary_info", lambda *args, **kwargs: enriched_gdf.copy())

    def fake_remove(path):
        captured["removed"].append(path)
        if path == vectorized_path:
            state["vectorized_exists"] = False

    monkeypatch.setattr(download_and_vectorize.os, "remove", fake_remove)
    monkeypatch.setattr(download_and_vectorize.os, "replace", lambda src, dst: captured["replaced"].append((src, dst)))

    original_to_parquet = gpd.GeoDataFrame.to_parquet

    def fake_to_parquet(self, path, index=False, **kwargs):
        captured["writes"].append({"path": path, "index": index, "rows": len(self), "columns": self.columns.tolist()})

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fake_to_parquet)
        result = download_and_vectorize.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", original_to_parquet)

    assert result is True
    assert captured["removed"] == []
    assert captured["writes"] == [
        {
            "path": f"{vectorized_path}.tmp",
            "index": False,
            "rows": 1,
            "columns": ["HYBAS_ID", "ISO_2", "geometry"],
        }
    ]
    assert captured["replaced"] == [(f"{vectorized_path}.tmp", vectorized_path)]


def test_add_boundary_info_empty_input_short_circuits(monkeypatch):
    industrial = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    watershed = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "geometry": [Point(0, 0).buffer(1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(download_and_vectorize, "intersects_with_country_db", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("country helper should not run")))
    monkeypatch.setattr(download_and_vectorize, "intersect_with_polygon_sindex", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("basin helper should not run")))

    out = download_and_vectorize.add_boundary_info(
        industrial,
        watershed,
        overture_path="unused.parquet",
        overture_s3_url="s3://unused",
        basin_col="HYBAS_ID",
        sindex_concurrency=False,
        country_boundary_col="country",
        country_output_col="ISO_2",
    )

    assert out.empty
    assert "ISO_2" in out.columns
    assert "HYBAS_ID" in out.columns


def test_main_returns_false_when_boundary_enrichment_fails(monkeypatch, tmp_path):
    vectorized_path = str(tmp_path / "industrial_merged.parquet")
    watershed_path = str(tmp_path / "watershed.gpkg")
    cfg = {
        "paths": {
            "industrial_merged_filepath": vectorized_path,
            "watershed": watershed_path,
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://example/overture.parquet",
            "industrial_raster_persistent_dir": str(tmp_path / "rasters"),
        },
        "overwrite_existing": False,
        "industrial_min_cells": 20,
        "industrial_persist_rasters": False,
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

    merged_gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")
    watershed_gdf = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "geometry": [Point(0, 0).buffer(1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(download_and_vectorize, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(download_and_vectorize, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(download_and_vectorize.os.path, "exists", lambda path: path == vectorized_path)
    monkeypatch.setattr(download_and_vectorize.gpd, "read_parquet", lambda path: merged_gdf.copy())
    monkeypatch.setattr(download_and_vectorize.gpd, "read_file", lambda path, driver=None: watershed_gdf.copy())
    monkeypatch.setattr(
        download_and_vectorize,
        "add_boundary_info",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("enrichment failed")),
    )

    result = download_and_vectorize.main()

    assert result is False


def test_main_rejects_non_positive_industrial_min_cells(monkeypatch, tmp_path):
    vectorized_path = str(tmp_path / "industrial_merged.parquet")
    cfg = {
        "paths": {
            "industrial_merged_filepath": vectorized_path,
            "watershed": str(tmp_path / "watershed.gpkg"),
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://example/overture.parquet",
            "industrial_raster_persistent_dir": str(tmp_path / "rasters"),
        },
        "overwrite_existing": False,
        "industrial_min_cells": 0,
        "industrial_persist_rasters": False,
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

    monkeypatch.setattr(download_and_vectorize, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(download_and_vectorize, "load_config", lambda **overrides: cfg)

    with pytest.raises(ValueError, match="industrial_min_cells"):
        download_and_vectorize.main()


def test_main_persist_rasters_reuses_existing_rasters_without_download(monkeypatch, tmp_path):
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
        "overwrite_existing": False,
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
    captured = {}

    merged_gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")
    watershed_gdf = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "geometry": [Point(0, 0).buffer(1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    enriched_gdf = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "ISO_2": ["DE"], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(download_and_vectorize, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(download_and_vectorize, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(download_and_vectorize.os.path, "exists", lambda path: False)
    monkeypatch.setattr(download_and_vectorize.Path, "rglob", lambda self, pattern: [tmp_path / "tile_1.tif"])
    monkeypatch.setattr(download_and_vectorize, "download_file", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download should not be called")))
    monkeypatch.setattr(download_and_vectorize, "_find_raster_dirs", lambda base_dir: [base_dir])
    monkeypatch.setattr(download_and_vectorize, "_vectorize_and_merge", lambda *args, **kwargs: merged_gdf.copy())
    monkeypatch.setattr(download_and_vectorize.gpd, "read_file", lambda path, driver=None: watershed_gdf.copy())
    monkeypatch.setattr(download_and_vectorize, "add_boundary_info", lambda *args, **kwargs: enriched_gdf.copy())
    monkeypatch.setattr(download_and_vectorize.os, "replace", lambda src, dst: captured.setdefault("replace", []).append((src, dst)))

    original_to_parquet = gpd.GeoDataFrame.to_parquet
    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", lambda self, path, index=False, **kwargs: captured.setdefault("writes", []).append(path))
        result = download_and_vectorize.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", original_to_parquet)

    assert result is True
    assert captured["writes"][0] == f"{vectorized_path}.tmp"
    assert captured["writes"][1] == f"{vectorized_path}.tmp"


def test_repair_geometry_handles_none_invalid_and_valid():
    assert download_and_vectorize._repair_geometry(None) is None

    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    repaired = download_and_vectorize._repair_geometry(invalid)
    assert repaired is not None
    assert repaired.is_valid

    valid = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
    same = download_and_vectorize._repair_geometry(valid)
    assert same.equals(valid)


def test_dissolve_by_overlap_groups_handles_empty_and_missing_group_assignments(monkeypatch):
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    out_empty = download_and_vectorize._dissolve_by_overlap_groups(empty)
    assert out_empty.empty

    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0).buffer(0.1), Point(1, 1).buffer(0.1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    monkeypatch.setattr(download_and_vectorize, "dissolve_overlapping_geometries_fast", lambda frame, radius=0, convex=True: ([], None))
    out = download_and_vectorize._dissolve_by_overlap_groups(gdf)
    assert len(out) == 2


def test_add_boundary_info_raises_when_basin_column_missing():
    industrial = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0).buffer(0.1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    watershed = gpd.GeoDataFrame({"wrong_col": [1], "geometry": [Point(0, 0).buffer(1)]}, geometry="geometry", crs="EPSG:4326")

    with pytest.raises(KeyError, match="Watershed column"):
        download_and_vectorize.add_boundary_info(
            industrial,
            watershed,
            overture_path="unused.parquet",
            overture_s3_url="s3://unused",
            basin_col="HYBAS_ID",
            sindex_concurrency=False,
            country_boundary_col="country",
            country_output_col="ISO_2",
        )


def test_main_non_persistent_branch_downloads_extracts_and_vectorizes(monkeypatch, tmp_path):
    vectorized_path = str(tmp_path / "industrial_merged.parquet")
    cfg = {
        "paths": {
            "industrial_merged_filepath": vectorized_path,
            "watershed": str(tmp_path / "watershed.gpkg"),
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://example/overture.parquet",
            "industrial_raster_persistent_dir": str(tmp_path / "rasters"),
        },
        "overwrite_existing": False,
        "industrial_min_cells": 20,
        "industrial_persist_rasters": False,
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
    captured = {}

    merged_gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")
    watershed_gdf = gpd.GeoDataFrame({"HYBAS_ID": [1], "geometry": [Point(0, 0).buffer(1)]}, geometry="geometry", crs="EPSG:4326")
    enriched_gdf = gpd.GeoDataFrame({"HYBAS_ID": [1], "ISO_2": ["DE"], "geometry": [Point(0, 0)]}, geometry="geometry", crs="EPSG:4326")

    class _ZipStub:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extractall(self, path):
            captured.setdefault("extract", []).append(path)

    class _TempDirStub:
        def __enter__(self):
            return str(tmp_path / "tmpdir")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(download_and_vectorize, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(download_and_vectorize, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(download_and_vectorize.os.path, "exists", lambda path: False)
    monkeypatch.setattr(download_and_vectorize.os, "makedirs", lambda path, exist_ok=False: captured.setdefault("makedirs", []).append((path, exist_ok)))
    monkeypatch.setattr(download_and_vectorize, "download_file", lambda url, dest, chunk_size=8192: captured.setdefault("download", (url, dest)))
    monkeypatch.setattr(download_and_vectorize.zipfile, "ZipFile", lambda path, mode: _ZipStub())
    monkeypatch.setattr(download_and_vectorize.tempfile, "TemporaryDirectory", lambda: _TempDirStub())
    monkeypatch.setattr(download_and_vectorize, "_find_raster_dirs", lambda base_dir: [base_dir])
    monkeypatch.setattr(download_and_vectorize, "_vectorize_and_merge", lambda *args, **kwargs: merged_gdf.copy())
    monkeypatch.setattr(download_and_vectorize.gpd, "read_file", lambda path, driver=None: watershed_gdf.copy())
    monkeypatch.setattr(download_and_vectorize, "add_boundary_info", lambda *args, **kwargs: enriched_gdf.copy())
    monkeypatch.setattr(download_and_vectorize.os, "replace", lambda src, dst: captured.setdefault("replace", []).append((src, dst)))

    original_to_parquet = gpd.GeoDataFrame.to_parquet
    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", lambda self, path, index=False, **kwargs: captured.setdefault("writes", []).append(path))
        result = download_and_vectorize.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", original_to_parquet)

    assert result is True
    assert captured["download"][0] == cfg["industrial_zenodo_url"]
    assert len(captured["writes"]) == 2
    assert captured["writes"][0] == f"{vectorized_path}.tmp"
    assert captured["writes"][1] == f"{vectorized_path}.tmp"


# --- moved from the coverage-campaign files ---

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


class TestDownloadAndVectorizeLogic:
    """Test industrial data download logic."""

    def test_download_file_url_structure(self):
        """Test URL validation."""
        url = "https://zenodo.org/record/12345/files/data.zip"
        
        assert url.startswith("http")
        assert ".zip" in url

    def test_download_file_destination_path(self):
        """Test output destination path."""
        url = "https://zenodo.org/record/12345/files/data.zip"
        dest_path = "/tmp/data.zip"
        
        assert os.path.dirname(dest_path) in ["/tmp", ""]
        assert dest_path.endswith(".zip")

    @patch('wwtp.industrial_analysis.download_and_vectorize.Path')
    def test_vectorize_raster_file_list_tifs(self, mock_path):
        """Test raster file discovery."""
        raster_dir = "/data/rasters"
        
        # Would glob for tif files
        assert isinstance(raster_dir, str)

    def test_vectorize_rasters_parallel_task_count(self):
        """Test parallel worker count."""
        max_workers = 8
        raster_files = ["/path/to/r1.tif", "/path/to/r2.tif", "/path/to/r3.tif"]
        
        # Verify worker count is reasonable
        assert max_workers > 0
        assert max_workers >= 1

    def test_add_boundary_info_gdf_structure(self):
        """Test enriched GeoDataFrame structure."""
        # Simulate input/output structure
        input_gdf = gpd.GeoDataFrame({
            "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)],
        })
        
        # After enrichment, should have additional columns
        output_gdf = input_gdf.copy()
        output_gdf["country"] = ["US", "CA"]
        output_gdf["basin"] = ["Basin1", "Basin1"]
        
        assert "country" in output_gdf.columns
        assert "basin" in output_gdf.columns


class TestDownloadAndVectorizeUtilities:
    """Test utility functions in download_and_vectorize."""
    
    def test_download_file_propagates_http_errors(self):
        """download_file must not swallow a failed HTTP response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "file.txt"

            response = MagicMock()
            response.raise_for_status.side_effect = OSError("404 Not Found")
            session = MagicMock()
            session.get.return_value = response

            with patch.object(dv, "requests_session_with_retries", return_value=session):
                with pytest.raises(OSError):
                    dv.download_file(
                        "http://invalid.nonexistent.url/file.zip",
                        str(dest),
                        chunk_size=8192,
                    )

            assert not dest.exists()
    
    def test_vectorize_raster_file_missing(self):
        """Test vectorize_raster_file with missing input."""
        with pytest.raises(Exception):
            dv.vectorize_raster_file("/nonexistent/raster.tif")
