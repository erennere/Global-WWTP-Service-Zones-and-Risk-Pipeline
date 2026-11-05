from __future__ import annotations

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box, mapping

from src import geo_utils
from src.pop_at_risk_river_calculations import create_rasters, find_intersection_river


pytestmark = pytest.mark.unit


class _MiniRow(dict):
    def __init__(self, geometry, **data):
        super().__init__(data)
        self.geometry = geometry


class _MiniSubset:
    def __init__(self, rows):
        self.geometry = [row.geometry for row in rows]


class _MiniILoc:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, index):
        if isinstance(index, list):
            return _MiniSubset([self._rows[idx] for idx in index])
        return self._rows[index]


class _MiniSindex:
    def __init__(self, matches):
        self._matches = list(matches)
        self._position = 0

    def intersection(self, bounds):
        match = self._matches[self._position]
        self._position += 1
        return match


class _MiniGdf:
    def __init__(self, rows, sindex_matches):
        self.iloc = _MiniILoc(rows)
        self._sindex = _MiniSindex(sindex_matches)

    def to_crs(self, crs):
        return self

    @property
    def sindex(self):
        return self._sindex


class _FakeRasterSrc:
    def __init__(self, arrays):
        self.crs = "EPSG:4326"
        self.res = (0.1, 0.1)
        self.transform = from_origin(0, 1, 1, 1)
        self._windows = [rasterio.windows.Window(idx, 0, 1, 1) for idx in range(len(arrays))]
        self._arrays = arrays

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def block_windows(self, band):
        return [((0, idx), window) for idx, window in enumerate(self._windows)]

    def read(self, band, window=None):
        return self._arrays[self._windows.index(window)].copy()


def test_graph_helpers_find_common_intersections():
    rivers = pd.DataFrame({"HYRIV_ID": [1, 2, 3, 4], "NEXT_DOWN": [3, 3, 4, None]})
    graph = find_intersection_river.build_graph(rivers)

    assert graph[1] == 3
    assert find_intersection_river.find_intersection_id(1, 2, graph) == 3
    assert find_intersection_river.find_common_intersection([1, 2], graph) == 3
    assert find_intersection_river.find_common_intersection([], graph) is None


def test_optimize_river_lookup_and_assign_main_riv():
    polygons = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 202],
            "geometry": [box(0, 0, 1, 1), box(10, 10, 11, 11)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 101, 202],
            "HYRIV_ID": [11, 12, 21],
            "MAIN_RIV": [1001, 1001, 2001],
            "geometry": [LineString([(0.1, 0.1), (0.9, 0.9)]), LineString([(0.2, 0.2), (0.8, 0.8)]), LineString([(10.1, 10.1), (10.9, 10.9)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    out = find_intersection_river.optimize_river_lookup(polygons.copy(), rivers.copy(), x_distance=1000.0, utm_epsg=3857)

    assert len(out["river_list"].iloc[0]) >= 1
    assert len(out["river_list"].iloc[1]) >= 1

    with_main = find_intersection_river.assign_main_riv(out.copy(), rivers)
    assert with_main["MAIN_RIV"].notna().all()


def test_assign_river_juncture_and_orchestrate_assignment(monkeypatch):
    polygons = gpd.GeoDataFrame(
        {
            "river_list": [[1, 2], []],
            "MAIN_RIV": [1001, 1002],
            "geometry": [box(0, 0, 1, 1), box(2, 2, 3, 3)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {
            "HYRIV_ID": [1, 2, 3],
            "NEXT_DOWN": [3, 3, None],
            "MAIN_RIV": [1001, 1001, 1001],
            "geometry": [LineString([(0, 0), (1, 1)])] * 3,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    single = find_intersection_river.assign_river_juncture(polygons.iloc[[0]].copy(), rivers)
    assert single["NXT_DIS"].iloc[0] == 3

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

    monkeypatch.setattr(find_intersection_river, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(find_intersection_river, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(find_intersection_river, "tqdm", lambda iterable, **kwargs: iterable)

    out = find_intersection_river.orchestrate_river_assignment(polygons.copy(), rivers.copy(), max_workers=1)
    assert "NXT_DIS" in out.columns


def test_sanitize_polygon_geom_and_sharding_helpers():
    assert create_rasters._sanitize_polygon_geom(None) is None

    gc = GeometryCollection([
        Point(0, 0),
        Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]),
    ])
    cleaned = create_rasters._sanitize_polygon_geom(gc)
    assert cleaned is not None

    gc_without_polygons = GeometryCollection([Point(5, 5), LineString([(0, 0), (1, 1)])])
    assert create_rasters._sanitize_polygon_geom(gc_without_polygons) is None

    tif_dict = {"DE": "a.tif", "FR": "b.tif", "IT": "c.tif"}
    shard = create_rasters.shard_tif_dict(tif_dict, job_index=0, total_jobs=2, seed=123)
    assert len(shard) >= 1

    with pytest.raises(ValueError):
        create_rasters.shard_tif_dict(tif_dict, job_index=-1, total_jobs=2, seed=1)
    with pytest.raises(ValueError):
        create_rasters.shard_tif_dict(tif_dict, job_index=0, total_jobs=0, seed=1)


def test_sanitize_polygon_geom_handles_make_valid_fallback_and_nonpolygon_results(monkeypatch):
    class _BrokenGeom:
        is_empty = False
        # the shared repair_geometry short-circuits on already-valid input, so the
        # stub has to declare itself invalid to reach the make_valid/buffer path
        is_valid = False

        def buffer(self, distance):
            return None

    monkeypatch.setattr(geo_utils, "make_valid", lambda geom: (_ for _ in ()).throw(RuntimeError("boom")))
    assert create_rasters._sanitize_polygon_geom(_BrokenGeom()) is None

    monkeypatch.setattr(geo_utils, "make_valid", lambda geom: geom)
    monkeypatch.setattr(create_rasters, "unary_union", lambda geoms: Polygon())

    gc = GeometryCollection([Polygon([(0, 0), (1, 0), (0, 1), (0, 0)])])
    assert create_rasters._sanitize_polygon_geom(gc) is None
    assert create_rasters._sanitize_polygon_geom(LineString([(0, 0), (1, 1)])) is None




def test_geotiff_exists_valid_and_polygon_raster_sign(tmp_path):
    raster_path = tmp_path / "src.tif"
    out_path = tmp_path / "signed.tif"
    transform = from_origin(0, 2, 1, 1)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="int32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.array([[1, 2], [3, 4]], dtype=np.int32), 1)

    assert create_rasters.geotiff_exists_and_valid(str(raster_path)) is True
    assert create_rasters.geotiff_exists_and_valid(str(tmp_path / "missing.tif")) is False

    polygons = gpd.GeoDataFrame(
        {"geometry": [box(0, 1, 1, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    filepath, pos, neg = create_rasters.polygon_raster_sign_from_gdf(str(raster_path), polygons, str(out_path))
    assert filepath == str(out_path)
    assert pos is not None and neg is not None
    assert out_path.exists()


def test_polygon_raster_sign_handles_nodata_and_windows_without_polygons(tmp_path):
    raster_path = tmp_path / "src_nodata.tif"
    out_path = tmp_path / "signed_nodata.tif"
    transform = from_origin(0, 2, 1, 1)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(np.array([[-9999, 2], [3, 4]], dtype=np.float32), 1)

    polygons = gpd.GeoDataFrame(
        {"geometry": [box(10, 10, 11, 11)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    filepath, pos, neg = create_rasters.polygon_raster_sign_from_gdf(str(raster_path), polygons, str(out_path))

    assert filepath == str(out_path)
    assert pos == 0
    assert neg == -9


def test_polygon_raster_sign_returns_none_stats_on_error(monkeypatch):
    monkeypatch.setattr(create_rasters.rasterio, "open", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad raster")))

    filepath, pos, neg = create_rasters.polygon_raster_sign_from_gdf(
        "broken.tif",
        gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326"),
        "out.tif",
    )

    assert filepath == "out.tif"
    assert pos is None and neg is None


def test_geotiff_exists_returns_false_on_open_error(monkeypatch):
    monkeypatch.setattr(create_rasters.os.path, "exists", lambda path: True)
    monkeypatch.setattr(create_rasters.rasterio, "open", lambda path: (_ for _ in ()).throw(RuntimeError("bad tiff")))

    assert create_rasters.geotiff_exists_and_valid("broken.tif") is False


def test_parse_args_reads_optional_positional(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--job-index", "1", "--total-jobs", "3", "--level", "7"],
    )

    args = create_rasters.parse_args()

    assert args.job_index == 1
    assert args.total_jobs == 3
    assert args.level == "7"


def test_create_rasters_main_requires_watershed_crs(monkeypatch, tmp_path):
    cfg = {
        "annotations": {"max_workers": 1, "random_seed": 1},
        "min_pixels": 9,
        "served_all_touched": False,
        "zoom_level": 8,
        "figures": {"approach": "1"},
        "country_output_column": "ISO_2",
        "country_boundary_column": "country",
        "basin_column_name": "HYBAS_ID",
        "paths": {
            "pop_tif_dir": str(tmp_path / "tifs"),
            "WWTP_tif_dir": str(tmp_path / "out_tifs"),
            "non_served_outpath": str(tmp_path / "non_served.gpkg"),
            "csv_output_filepath": str(tmp_path / "stats.gpkg"),
            "watershed": str(tmp_path / "watershed.geojson"),
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://example/overture.parquet",
        },
    }

    no_crs_watershed = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
    )
    voronoi_gdf = gpd.GeoDataFrame(
        {"ISO_2": ["DE"], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(create_rasters, "parse_args", lambda: type("_Args", (), {"job_index": 0, "total_jobs": 1})())
    monkeypatch.setattr(create_rasters.os, "chdir", lambda path: None)
    monkeypatch.setattr(create_rasters, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(create_rasters, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(create_rasters, "create_pop_output_paths", lambda _cfg: {"voronoi": {"1": str(tmp_path / "voronoi.gpkg")}})
    monkeypatch.setattr(
        create_rasters.gpd,
        "read_file",
        lambda path: no_crs_watershed.copy() if path == cfg["paths"]["watershed"] else voronoi_gdf.copy(),
    )
    monkeypatch.setattr(create_rasters, "find_newest_country_tif_files", lambda countries, tif_dir: {"DE": "de.tif"})

    with pytest.raises(ValueError, match="must include CRS"):
        create_rasters.main()


def test_create_rasters_main_enriches_watershed_when_country_column_missing(monkeypatch, tmp_path):
    cfg = {
        "annotations": {"max_workers": 2, "random_seed": 7},
        "min_pixels": 9,
        "served_all_touched": False,
        "zoom_level": 8,
        "figures": {"approach": "1"},
        "country_output_column": "ISO_2",
        "country_boundary_column": "country",
        "basin_column_name": "HYBAS_ID",
        "paths": {
            "pop_tif_dir": str(tmp_path / "tifs"),
            "WWTP_tif_dir": str(tmp_path / "out_tifs"),
            "non_served_outpath": str(tmp_path / "non_served.gpkg"),
            "csv_output_filepath": str(tmp_path / "stats.gpkg"),
            "watershed": str(tmp_path / "watershed.geojson"),
            "watershed_with_countries": str(tmp_path / "watershed_with_countries.gpkg"),
            "overture": str(tmp_path / "overture.parquet"),
            "overture_s3_url": "s3://example/overture.parquet",
        },
    }
    captured = {}
    os.makedirs(cfg["paths"]["WWTP_tif_dir"], exist_ok=True)

    watershed = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "country": ["DE"], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    voronoi = gpd.GeoDataFrame(
        {"ISO_2": ["DE"], "geometry": [Point(0.2, 0.2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    enriched = gpd.GeoDataFrame(
        {"HYBAS_ID": [1], "country": ["DE"], "ISO_2": ["DE"], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(create_rasters, "parse_args", lambda: type("_Args", (), {"job_index": 0, "total_jobs": 1})())
    monkeypatch.setattr(create_rasters.os, "chdir", lambda path: None)
    monkeypatch.setattr(create_rasters, "parse_config_overrides", lambda args=None: {})
    monkeypatch.setattr(create_rasters, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(create_rasters, "create_pop_output_paths", lambda _cfg: {"voronoi": {"1": str(tmp_path / "voronoi.gpkg")}})
    monkeypatch.setattr(
        create_rasters.gpd,
        "read_file",
        lambda path: watershed.copy() if path == cfg["paths"]["watershed"] else voronoi.copy(),
    )
    monkeypatch.setattr(create_rasters, "find_newest_country_tif_files", lambda countries, tif_dir: {"DE": "de.tif"})
    monkeypatch.setattr(create_rasters.os.path, "exists", lambda path: False if path == cfg["paths"]["overture"] else True)
    monkeypatch.setattr(create_rasters, "download_overture_maps", lambda s3, out: captured.setdefault("download", (s3, out)))
    monkeypatch.setattr(create_rasters, "intersects_with_country_db", lambda *args, **kwargs: enriched.copy())
    monkeypatch.setattr(create_rasters, "ensure_output_dir_for_file", lambda path: captured.setdefault("ensured", path))
    monkeypatch.setattr(
        create_rasters,
        "orchestrate_intersections",
        lambda *args, **kwargs: captured.setdefault("orchestrated", {"countries": list(args[0].keys()), "workers": args[6]}),
    )

    original_to_file = gpd.GeoDataFrame.to_file
    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda self, path, driver=None, index=None, **kwargs: captured.setdefault("write", (path, driver, index)))
        create_rasters.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["download"] == (cfg["paths"]["overture_s3_url"], cfg["paths"]["overture"])
    assert captured["write"][0].endswith("watershed_with_countries.gpkg")
    assert captured["write"][1] == "GPKG"
    assert captured["write"][2] is False
    assert captured["orchestrated"] == {"countries": ["DE"], "workers": 2}


def test_extract_worldpop_universal_handles_skip_merge_and_invalid_chunk_paths(monkeypatch):
    hybas = _MiniGdf(
        [
            _MiniRow(box(0, 0, 1, 1), HYBAS_ID=1, NEXT_DOWN=10, NEXT_SINK=11, MAIN_BAS=12),
            _MiniRow(box(0, 0, 1, 1), HYBAS_ID=2, NEXT_DOWN=20, NEXT_SINK=21, MAIN_BAS=22),
        ],
        sindex_matches=[[0, 1], [1], [1]],
    )
    exclude = _MiniGdf([_MiniRow(box(0, 0, 1, 1))], sindex_matches=[[0], [], []])
    src = _FakeRasterSrc([np.array([[1]], dtype=np.int32) for _ in range(3)])
    mask_values = [
        np.array([[False]], dtype=bool),
        np.array([[True]], dtype=bool),
        np.array([[False]], dtype=bool),
        np.array([[False]], dtype=bool),
        np.array([[False]], dtype=bool),
    ]
    shape_values = [
        [(mapping(box(0.1, 0.1, 0.2, 0.2)), 1)],
        [(mapping(box(0.1, 0.1, 0.9, 0.9)), 1)],
        [(mapping(box(0.1, 0.1, 0.9, 0.9)), 1)] * 199,
    ]
    sanitize_values = [None, box(0.1, 0.1, 0.5, 0.5), None]

    monkeypatch.setattr(create_rasters.rasterio, "open", lambda path: src)
    monkeypatch.setattr(create_rasters, "geometry_mask", lambda *args, **kwargs: mask_values.pop(0))
    monkeypatch.setattr(create_rasters, "shapes", lambda *args, **kwargs: iter(shape_values.pop(0)))
    monkeypatch.setattr(create_rasters, "finding_tiles", lambda island, zoom_level=8: ["1/1/1", "1/1/2"])
    monkeypatch.setattr(create_rasters, "find_bbox", lambda tile: box(0, 0, 1, 1))
    monkeypatch.setattr(create_rasters, "_sanitize_polygon_geom", lambda geom: sanitize_values.pop(0))
    monkeypatch.setattr(create_rasters, "exact_extract", lambda *args, **kwargs: pytest.fail("exact_extract should not run"))
    monkeypatch.setattr(create_rasters.gc, "collect", lambda: None)

    result = create_rasters.extract_worldpop_universal(
        "country.tif",
        hybas,
        exclude,
        min_pixels=2,
        zoom_level=4,
        basin_col="HYBAS_ID",
    )

    assert result.empty


def test_extract_worldpop_universal_returns_none_when_no_final_rows(monkeypatch):
    src = _FakeRasterSrc([np.array([[1]], dtype=np.int32)])

    monkeypatch.setattr(create_rasters.rasterio, "open", lambda path: src)
    monkeypatch.setattr(create_rasters.gc, "collect", lambda: None)

    result = create_rasters.extract_worldpop_universal(
        "country.tif",
        _MiniGdf([], sindex_matches=[[]]),
        _MiniGdf([], sindex_matches=[[]]),
    )

    assert result is None


def test_extract_worldpop_universal_handles_empty_exact_extract_results(monkeypatch):
    src = _FakeRasterSrc([np.array([[1]], dtype=np.int32)])
    hybas = _MiniGdf(
        [_MiniRow(box(0, 0, 1, 1), HYBAS_ID=1, NEXT_DOWN=10, NEXT_SINK=11, MAIN_BAS=12)],
        sindex_matches=[[0]],
    )

    monkeypatch.setattr(create_rasters.rasterio, "open", lambda path: src)
    monkeypatch.setattr(create_rasters, "geometry_mask", lambda *args, **kwargs: np.array([[False]], dtype=bool))
    monkeypatch.setattr(create_rasters, "shapes", lambda *args, **kwargs: iter([(mapping(box(0.1, 0.1, 0.9, 0.9)), 1)]))
    monkeypatch.setattr(create_rasters, "finding_tiles", lambda island, zoom_level=8: ["1/1/1"])
    monkeypatch.setattr(create_rasters, "find_bbox", lambda tile: box(0, 0, 1, 1))
    monkeypatch.setattr(create_rasters, "_sanitize_polygon_geom", lambda geom: box(0.1, 0.1, 0.9, 0.9))
    monkeypatch.setattr(create_rasters, "exact_extract", lambda *args, **kwargs: pd.DataFrame(columns=["sum", "count"]))
    monkeypatch.setattr(create_rasters.gc, "collect", lambda: None)

    result = create_rasters.extract_worldpop_universal(
        "country.tif",
        hybas,
        _MiniGdf([], sindex_matches=[[]]),
        min_pixels=2,
        zoom_level=4,
        basin_col="HYBAS_ID",
    )

    assert result.empty


def test_extract_worldpop_universal_returns_none_on_critical_failure(monkeypatch):
    monkeypatch.setattr(create_rasters.rasterio, "open", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    assert create_rasters.extract_worldpop_universal("country.tif", None, None) is None
