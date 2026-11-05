"""Tests for create_rasters (moved from the coverage-campaign files)."""
import argparse
import os
import runpy
import sys
import types
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box
from src import download_pop as dp
from src.annotation_scripts import NEW_01_GENERATEGRIDS as n1
from src.annotation_scripts import NEW_02_EXTRACTOSMDATAFULL_GEOJSON as n2
from src.annotation_scripts import NEW_03_WASTEWATERJOIN_GEOJSON as ww3
from src.annotation_scripts import download_bing_annotate as dba
from src.figures_scripts import piechart_figure as pf
from src import starter
from src.pop_at_risk_river_calculations import create_rasters as cr
from src.pop_at_risk_river_calculations import find_intersection_river as fir
from src.pop_at_risk_river_calculations import impact_polygons_pop as ipp
from src.industrial_analysis import download_and_vectorize as dv
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from shapely.geometry import Point
try:
    from src import download_pop as dp
    from src.pop_at_risk_river_calculations import create_rasters as cr
    from src.industrial_analysis import download_and_vectorize as dv
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import download_pop as dp
    from pop_at_risk_river_calculations import create_rasters as cr
    from industrial_analysis import download_and_vectorize as dv
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
import io
from unittest.mock import mock_open, patch
from shapely.geometry import Point, Polygon, box
from src.pop_at_risk_river_calculations import create_rasters

pytestmark = pytest.mark.unit


def test_create_rasters_main_smoke(monkeypatch, tmp_path):
    args = argparse.Namespace(job_index=0, total_jobs=1)
    monkeypatch.setattr(cr, "parse_args", lambda: args)
    monkeypatch.setattr(cr, "parse_config_overrides", lambda args=None: {})

    output_tif_dir = tmp_path / "out_tifs"
    watershed_path = tmp_path / "watershed.geojson"
    overture_path = tmp_path / "overture.parquet"

    cfg = {
        "annotations": {"max_workers": 2, "random_seed": 11},
        "min_pixels": 3,
        "served_all_touched": False,
        "zoom_level": 8,
        "figures": {"approach": "1"},
        "paths": {
            "pop_tif_dir": str(tmp_path / "pop_tifs"),
            "WWTP_tif_dir": str(output_tif_dir),
            "non_served_outpath": str(tmp_path / "non_served.gpkg"),
            "csv_output_filepath": str(tmp_path / "stats.gpkg"),
            "watershed": str(watershed_path),
            "watershed_with_countries": str(tmp_path / "watershed_with_countries.gpkg"),
            "overture": str(overture_path),
            "overture_s3_url": "s3://dummy/overture",
        },
        "country_output_column": "iso3",
        "country_boundary_column": "country_name",
        "basin_column_name": "HYBAS_ID",
    }
    monkeypatch.setattr(cr, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(cr, "create_pop_output_paths", lambda cfg_: {"voronoi": {"1": str(tmp_path / "voronoi.gpkg")}})

    voronoi = gpd.GeoDataFrame(
        {"iso3": ["DEU"], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    watershed = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "country_name": ["Germany"], "geometry": [box(0, 0, 2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    def _read_file(path, crs=None):
        if str(path).endswith("voronoi.gpkg"):
            return voronoi.copy()
        return watershed.copy()

    monkeypatch.setattr(cr.gpd, "read_file", _read_file)
    monkeypatch.setattr(cr, "find_newest_country_tif_files", lambda countries, tif_dir: {"DEU": str(tmp_path / "deu.tif")})
    monkeypatch.setattr(cr.os, "chdir", lambda path: None)
    monkeypatch.setattr(cr.os.path, "exists", lambda p: False)
    monkeypatch.setattr(cr.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(cr, "download_overture_maps", lambda s3, out: None)
    monkeypatch.setattr(
        cr,
        "intersects_with_country_db",
        lambda gdf, overture, polygon_country_col, output_country_col: gdf.assign(**{output_country_col: ["DEU"]}),
    )
    monkeypatch.setattr(cr, "ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, *args, **kwargs: None)

    seen = {}
    monkeypatch.setattr(
        cr,
        "orchestrate_intersections",
        lambda tif_dict, gdf, watershed_gdf, output_tif_dir, csv_output_filepath, non_served_outpath, max_workers, **kwargs: seen.update(
            {
                "tif_count": len(tif_dict),
                "gdf_rows": len(gdf),
                "watershed_rows": len(watershed_gdf),
                "max_workers": max_workers,
            }
        ),
    )

    cr.main()
    assert seen["tif_count"] == 1
    assert seen["gdf_rows"] == 1
    assert seen["watershed_rows"] == 1
    assert seen["max_workers"] == 2


class TestCreateRastersUtilities:
    """Test utility functions in create_rasters."""
    
    def test_extract_worldpop_with_minimal_data(self):
        """Test extract_worldpop_universal with minimal test data."""
        # Create minimal test data
        hybas_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(0, 0).buffer(1)], 'HYBAS_ID': [1]},
            crs='EPSG:4326'
        )
        exclude_gdf = gpd.GeoDataFrame(
            columns=['geometry'],
            crs='EPSG:4326'
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal valid raster
            import rasterio
            from rasterio.transform import from_bounds
            
            raster_path = Path(tmpdir) / "test.tif"
            
            # Create test raster
            data = np.ones((10, 10), dtype=np.float32) * 100
            transform = from_bounds(0, 0, 1, 1, 10, 10)
            
            with rasterio.open(
                str(raster_path), 'w',
                driver='GTiff',
                height=10, width=10,
                count=1, dtype=np.float32,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(data, 1)
            
            result = cr.extract_worldpop_universal(
                str(raster_path),
                hybas_gdf,
                exclude_gdf,
                min_pixels=1
            )

            assert isinstance(result, gpd.GeoDataFrame)


def test_orchestrate_country_intersection_wraps_sign_and_extract(monkeypatch):
    tiny = gpd.GeoDataFrame(
        {"HYBAS_ID": [101], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(create_rasters, "polygon_raster_sign_from_gdf", lambda *args, **kwargs: ("out.tif", 10, -5))
    monkeypatch.setattr(create_rasters, "extract_worldpop_universal", lambda *args, **kwargs: tiny.copy())

    out = create_rasters.orchestrate_country_intersection("in.tif", tiny, tiny, "out.tif")

    assert out[0] == "out.tif"
    assert out[1] == 10 and out[2] == -5
    assert isinstance(out[3], gpd.GeoDataFrame)


def test_orchestrate_intersections_writes_stats_and_nonserved(monkeypatch, tmp_path):
    tif_dict = {"DE": "de.tif", "FR": "fr.tif"}
    gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "FR"],
            "HYBAS_ID": [101, 202],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    watershed = gdf.copy()

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

    def fake_orchestrate_country_intersection(tif_filepath, sub_gdf, sub_ws, out, **kwargs):
        island = gpd.GeoDataFrame(
            {"tile": ["1/1/1"], "geometry": [box(0, 0, 1, 1)]},
            geometry="geometry",
            crs="EPSG:3857",
        )
        return out, 100, -25, island

    writes = []

    def fake_to_csv(self, path, index=False, mode=None, header=None, **kwargs):
        writes.append({"path": path, "mode": mode, "header": header, "rows": len(self)})

    monkeypatch.setattr(create_rasters, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(create_rasters, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(create_rasters, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(create_rasters.random, "shuffle", lambda seq: None)
    monkeypatch.setattr(create_rasters, "orchestrate_country_intersection", fake_orchestrate_country_intersection)
    monkeypatch.setattr(create_rasters, "ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

    csv_path = str(tmp_path / "stats.csv")
    non_served = str(tmp_path / "non_served.gpkg")

    monkeypatch.setattr(create_rasters.os.path, "exists", lambda path: False)

    result = create_rasters.orchestrate_intersections(
        tif_dict,
        gdf,
        watershed,
        output_dir=str(tmp_path / "out"),
        csv_output_filepath=csv_path,
        non_served_outpath=non_served,
        max_workers=1,
        country_col="ISO_2",
        basin_col="HYBAS_ID",
    )

    assert result == {"DE": True, "FR": True}
    assert len(writes) >= 2


def test_orchestrate_intersections_skips_processed_countries_and_appends_outputs(monkeypatch, tmp_path):
    tif_dict = {"DE": "de.tif", "FR": "fr.tif"}
    gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "FR"],
            "HYBAS_ID": [101, 202],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    watershed = gdf.copy()

    class _Future:
        def __init__(self, fn, args, kwargs):
            self._fn = fn
            self._args = args
            self._kwargs = kwargs

        def result(self):
            return self._fn(*self._args, **self._kwargs)

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _Future(fn, args, kwargs)

    def fake_orchestrate_country_intersection(tif_filepath, sub_gdf, sub_ws, out, **kwargs):
        island = gpd.GeoDataFrame(
            {"tile": ["1/1/1"], "geometry": [box(0, 0, 1, 1)]},
            geometry="geometry",
        )
        return out, 100, -25, island

    csv_path = str(tmp_path / "stats.csv")
    non_served = str(tmp_path / "non_served.gpkg")
    ensured = []
    writes = []

    def fake_to_csv(self, path, index=False, mode=None, header=None, **kwargs):
        writes.append({"path": path, "mode": mode, "header": header, "rows": len(self)})

    def fake_exists(path):
        return path in {csv_path, non_served.replace(".gpkg", ".csv")}

    monkeypatch.setattr(create_rasters, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(create_rasters, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(create_rasters, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(create_rasters.random, "shuffle", lambda seq: None)
    monkeypatch.setattr(create_rasters, "orchestrate_country_intersection", fake_orchestrate_country_intersection)
    monkeypatch.setattr(create_rasters, "ensure_output_dir_for_file", lambda path: ensured.append(path))
    monkeypatch.setattr(create_rasters.os.path, "exists", fake_exists)
    monkeypatch.setattr(pd, "read_csv", lambda path: pd.DataFrame({"country": ["DE"]}))
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

    result = create_rasters.orchestrate_intersections(
        tif_dict,
        gdf,
        watershed,
        output_dir=str(tmp_path / "out"),
        csv_output_filepath=csv_path,
        non_served_outpath=non_served,
        max_workers=1,
        country_col="ISO_2",
        basin_col="HYBAS_ID",
    )

    assert result == {"FR": True}
    assert ensured == [csv_path, non_served.replace(".gpkg", ".csv")]
    assert all(write["mode"] == "a" and write["header"] is False for write in writes)


def test_orchestrate_intersections_handles_none_results_and_future_errors(monkeypatch, tmp_path):
    tif_dict = {"DE": "de.tif", "FR": "fr.tif"}
    gdf = gpd.GeoDataFrame(
        {
            "ISO_2": ["DE", "FR"],
            "HYBAS_ID": [101, 202],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    class _Future:
        def __init__(self, fn, args, kwargs):
            self._fn = fn
            self._args = args
            self._kwargs = kwargs

        def result(self):
            return self._fn(*self._args, **self._kwargs)

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _Future(fn, args, kwargs)

    def fake_orchestrate_country_intersection(tif_filepath, sub_gdf, sub_ws, out, **kwargs):
        if tif_filepath == "de.tif":
            return out, None, None, None
        raise RuntimeError("boom")

    monkeypatch.setattr(create_rasters, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(create_rasters, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(create_rasters, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(create_rasters.random, "shuffle", lambda seq: None)
    monkeypatch.setattr(create_rasters, "orchestrate_country_intersection", fake_orchestrate_country_intersection)
    monkeypatch.setattr(create_rasters, "ensure_output_dir_for_file", lambda path: None)
    monkeypatch.setattr(create_rasters.os.path, "exists", lambda path: False)

    result = create_rasters.orchestrate_intersections(
        tif_dict,
        gdf,
        gdf.copy(),
        output_dir=str(tmp_path / "out"),
        csv_output_filepath=str(tmp_path / "stats.csv"),
        non_served_outpath=str(tmp_path / "non_served.gpkg"),
        max_workers=1,
        country_col="ISO_2",
        basin_col="HYBAS_ID",
    )

    assert result == {"FR": False}


def test_orchestrate_intersections_rejects_non_positive_workers(monkeypatch, tmp_path):
    tif_dict = {"DE": "de.tif"}
    gdf = gpd.GeoDataFrame(
        {"ISO_2": ["DE"], "HYBAS_ID": [101], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="max_workers"):
        create_rasters.orchestrate_intersections(
            tif_dict,
            gdf,
            gdf.copy(),
            output_dir=str(tmp_path / "out"),
            csv_output_filepath=str(tmp_path / "stats.csv"),
            non_served_outpath=str(tmp_path / "non_served.gpkg"),
            max_workers=0,
            country_col="ISO_2",
            basin_col="HYBAS_ID",
        )


def test_create_rasters_main_watershed_requires_crs_and_read_file_no_crs_kwarg(monkeypatch, tmp_path):
    args = type("Args", (), {"job_index": 0, "total_jobs": 1})()
    monkeypatch.setattr(create_rasters, "parse_args", lambda: args)
    monkeypatch.setattr(create_rasters, "parse_config_overrides", lambda args=None: {})

    cfg = {
        "annotations": {"max_workers": 1, "random_seed": 7},
        "min_pixels": 3,
        "zoom_level": 8,
        "figures": {"approach": "1"},
        "paths": {
            "pop_tif_dir": str(tmp_path / "pop_tifs"),
            "WWTP_tif_dir": str(tmp_path / "out_tifs"),
            "non_served_outpath": str(tmp_path / "non_served.gpkg"),
            "csv_output_filepath": str(tmp_path / "stats.gpkg"),
            "watershed": str(tmp_path / "watershed.gpkg"),
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://dummy/overture",
        },
        "country_output_column": "iso3",
        "country_boundary_column": "country_name",
        "basin_column_name": "HYBAS_ID",
    }
    monkeypatch.setattr(create_rasters, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(
        create_rasters,
        "create_pop_output_paths",
        lambda cfg_: {"voronoi": {"1": str(tmp_path / "voronoi.gpkg")}},
    )

    voronoi = gpd.GeoDataFrame(
        {"iso3": ["DEU"], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    watershed_no_crs = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "country_name": ["Germany"], "geometry": [box(0, 0, 2, 2)]},
        geometry="geometry",
    )

    calls = {"watershed_kwargs": None}

    def _read_file(path, **kwargs):
        if str(path).endswith("voronoi.gpkg"):
            return voronoi.copy()
        calls["watershed_kwargs"] = dict(kwargs)
        return watershed_no_crs.copy()

    monkeypatch.setattr(create_rasters.gpd, "read_file", _read_file)
    monkeypatch.setattr(create_rasters.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(create_rasters.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(create_rasters.os, "chdir", lambda _p: None)
    monkeypatch.setattr(create_rasters, "find_newest_country_tif_files", lambda countries, tif_dir: {"DEU": str(tmp_path / "deu.tif")})

    with pytest.raises(ValueError, match="Watershed dataset must include CRS metadata"):
        create_rasters.main()

    assert calls["watershed_kwargs"] == {}


def test_create_rasters_script_entrypoint_runs_main_guard(monkeypatch):
    from pathlib import Path
    import sys

    import src.starter as starter_mod

    module_path = Path(__file__).resolve().parents[3] / "src" / "pop_at_risk_river_calculations" / "create_rasters.py"

    monkeypatch.setattr(starter_mod, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(starter_mod, "load_config", lambda **kwargs: {})
    monkeypatch.setattr(sys, "argv", [str(module_path)])

    with pytest.raises(KeyError, match="annotations"):
        runpy.run_path(str(module_path), run_name="__main__")
