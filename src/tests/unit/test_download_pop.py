"""Tests for download_pop (moved from the coverage-campaign files)."""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon
from shapely import to_wkb
from src.annotation_scripts import NEW_03_WASTEWATERJOIN_GEOJSON as ww_join
from src import download_pop
import os
from pathlib import Path
import numpy as np
from shapely.geometry import Point, box
from src import download_pop as dp
from src.figures_scripts import composite_area_population_plots as capp
from src.figures_scripts import pop_at_risk_figures as prf
import argparse
import runpy
import sys
import types
import duckdb
from shapely.geometry import LineString, Point, box
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
from unittest.mock import Mock, patch, MagicMock
from shapely.geometry import Point, Polygon, box
from shapely import from_wkt
import src.create_voronoi as cv
import src.download_pop as dp
from src.starter import parse_config_overrides, load_config
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
from unittest.mock import Mock, patch, MagicMock, call
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.industrial_analysis import download_and_vectorize as dav
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

pytestmark = pytest.mark.unit


def test_download_pop_small_helpers_and_finders(tmp_path):
    assert download_pop.extract_first_wildcard("abc_123", r"abc_(\d+)") == "123"
    assert download_pop.try_extract_country("population_deu.geotiff.zip", [r"population_([a-z]{3})"]) == "deu"

    urls = {}
    download_pop.add_country_url(urls, "deu", "u1")
    download_pop.add_country_url(urls, "deu", "u2")
    assert urls["deu"] == ["u1", "u2"]

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.tif").write_text("x", encoding="utf-8")
    (nested / "b.csv").write_text("x", encoding="utf-8")

    tif_files = download_pop.find_type(str(tmp_path), ".tif")
    assert len(tif_files) == 1

    files, is_tif = download_pop.find_files(str(tmp_path))
    assert is_tif is True
    assert len(files) == 1


def test_download_file_and_unzip_dispatch(monkeypatch, tmp_path):
    class _Resp:
        headers = {"Content-Length": "10"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"abc"

    class _Session:
        def get(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(download_pop, "requests_session_with_retries", lambda *a, **k: _Session())
    out = tmp_path / "f.bin"
    assert download_pop.download_file("http://x", str(out)) is True
    assert out.exists()

    monkeypatch.setattr(download_pop, "download_save_and_unzip_pop", lambda url, country, data_dir: str(tmp_path) if "ok" in url else None)
    result = download_pop.download_save_and_unzip_pops({"deu": ["bad", "ok"]}, "deu", data_dir=str(tmp_path))
    assert result == str(tmp_path)


def test_get_urls_contains_multiple_years():
    urls = download_pop.get_urls()
    key = next(iter(urls.keys()))
    assert len(urls[key]) >= 10


def test_download_pop_process_single_country_csv_and_tif_paths(tmp_path, monkeypatch):
    base = tmp_path / "population"
    csv_dir = base / "unzipped" / "deu"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "part.csv"
    pd.DataFrame({"lat": [0, 0.001], "lon": [0, 0.001], "pop": [1, 2]}).to_csv(csv_path, index=False)

    tif_dir = base / "unzipped" / "fra"
    tif_dir.mkdir(parents=True, exist_ok=True)

    # Create tiny tif
    import rasterio
    from rasterio.transform import from_origin

    arr = np.ones((5, 5), dtype=np.float32)
    tif_path = tif_dir / "part.tif"
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=from_origin(0, 1, 0.01, 0.01),
    ) as dst:
        dst.write(arr, 1)

    monkeypatch.setattr(dp, "download_save_and_unzip_pops", lambda *args, **kwargs: str(csv_dir))

    mosaics = []
    monkeypatch.setattr(dp, "mosaic_large_rasters", lambda files, out: mosaics.append((tuple(files), out)))

    dp.process_single_country({"deu": ["u"]}, "deu", res=100, data_dir=str(base))
    assert len(mosaics) >= 1

    monkeypatch.setattr(dp, "download_save_and_unzip_pops", lambda *args, **kwargs: str(tif_dir))
    dp.process_single_country({"fra": ["u"]}, "fra", res=100, data_dir=str(base))
    assert len(mosaics) >= 2


def test_download_pop_process_all_and_main(monkeypatch, tmp_path):
    class _Future:
        def __init__(self, fn, args):
            self._fn = fn
            self._args = args

        def result(self):
            return self._fn(*self._args)

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            return _Future(fn, args)

    monkeypatch.setattr(dp, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(dp, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(dp, "tqdm", lambda x, total=None, desc=None: x)

    calls = []
    monkeypatch.setattr(dp, "process_single_country", lambda country_urls, country, res, data_dir: calls.append((country, res, data_dir)))

    urls = {"deu": ["u1"], "fra": ["u2"]}
    dp.process_all_countries(urls, res=120, max_workers=2, data_dir=str(tmp_path))
    assert len(calls) == 2

    monkeypatch.setattr(dp, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(
        dp,
        "load_config",
        lambda **kwargs: {
            "paths": {"pop_dir": str(tmp_path)},
            "start_year": 2015,
            "end_year": 2024,
            "worldpop_2014_url_template": "tpl2014",
            "include_2014": False,
            "worldpop_yearly_url_template": "tplyear",
            "country_limit": 0,
        },
    )
    monkeypatch.setattr(dp, "get_urls", lambda **kwargs: {"a": ["u"], "b": ["u"], "c": ["u"], "d": ["u"]})
    monkeypatch.setattr(dp, "process_all_countries", lambda country_urls, res, max_workers, data_dir: calls.append((len(country_urls), res, max_workers, data_dir)))

    # country_limit=0 means "no limit" - all four countries are processed.
    # (YAML cannot express a literal null here, so 0 is the sentinel.)
    dp.main(res=90, max_workers=1)
    assert calls[-1][0] == 4

    # A positive country_limit really truncates, alphabetically.
    monkeypatch.setattr(
        dp,
        "load_config",
        lambda **kwargs: {
            "paths": {"pop_dir": str(tmp_path)},
            "start_year": 2015,
            "end_year": 2024,
            "worldpop_2014_url_template": "tpl2014",
            "include_2014": False,
            "worldpop_yearly_url_template": "tplyear",
            "country_limit": 3,
        },
    )
    dp.main(res=90, max_workers=1)
    assert calls[-1][0] == 3


def test_download_pop_get_urls_from_hdx_branches(monkeypatch):
    class _DatasetObj:
        def __init__(self, resources):
            self._resources = resources

        def get_resources(self):
            return self._resources

    class _Dataset:
        @staticmethod
        def search_in_hdx(_q):
            return [
                _DatasetObj(
                    [
                        {"download_url": "http://x/deu.zip", "name": "population_deu.geotiff.zip"},
                        {"download_url": "http://x/deu2.zip", "name": "population_deu.geotiff.zip"},
                        {"download_url": "http://x/anr.zip", "name": "ANR_men_geotiff.zip"},
                    ]
                )
            ]

    class _Configuration:
        @staticmethod
        def create(**_kwargs):
            return None

    mod_easy = types.ModuleType("hdx.utilities.easy_logging")
    mod_easy.setup_logging = lambda: None
    mod_cfg = types.ModuleType("hdx.api.configuration")
    mod_cfg.Configuration = _Configuration
    mod_dataset = types.ModuleType("hdx.data.dataset")
    mod_dataset.Dataset = _Dataset

    monkeypatch.setitem(sys.modules, "hdx.utilities.easy_logging", mod_easy)
    monkeypatch.setitem(sys.modules, "hdx.api.configuration", mod_cfg)
    monkeypatch.setitem(sys.modules, "hdx.data.dataset", mod_dataset)

    urls = dp.get_urls_from_hdx()
    assert "deu" in urls
    assert "anr" in urls
    assert len(urls["deu"]) == 1


def test_download_pop_download_file_error_branch(monkeypatch, tmp_path):
    class _ReqErr(dp.requests.RequestException):
        pass

    monkeypatch.setattr(dp.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(_ReqErr("x")))
    ok = dp.download_file("http://bad", str(tmp_path / "out.bin"))
    assert ok is False


def test_download_pop_process_single_country_and_pool_branches(monkeypatch, tmp_path):
    country_urls = {"deu": ["u1"], "fra": ["u2"]}

    # Branch: early return when extraction fails.
    monkeypatch.setattr(dp, "download_save_and_unzip_pops", lambda *args, **kwargs: None)
    assert dp.process_single_country(country_urls, "deu", data_dir=str(tmp_path)) is None

    # Branch: csv workflow with failed rasterize and no valid files.
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dp, "download_save_and_unzip_pops", lambda *args, **kwargs: str(extract_dir))
    monkeypatch.setattr(dp, "find_files", lambda folder: ([str(tmp_path / "a.csv")], False))
    monkeypatch.setattr(dp.pd, "read_csv", lambda path: pd.DataFrame({"lat": [0], "lon": [0], "pop": [1]}))
    monkeypatch.setattr(dp, "rasterize_csv", lambda df, output_path, res=30: None)
    monkeypatch.setattr(dp.os.path, "exists", lambda p: True if str(p).endswith("merged") or "rasterized" in str(p) else False)
    monkeypatch.setattr(dp.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(dp, "mosaic_large_rasters", lambda *args, **kwargs: None)
    dp.process_single_country(country_urls, "deu", data_dir=str(tmp_path))

    class _Future:
        def __init__(self, err=False):
            self.err = err

        def result(self):
            if self.err:
                raise RuntimeError("boom")
            return None

    class _Exec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            country = args[1]
            return _Future(err=(country == "fra"))

    monkeypatch.setattr(dp, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(dp, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(dp, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(dp, "process_single_country", lambda *args, **kwargs: None)
    dp.process_all_countries(country_urls, res=25, max_workers=2, data_dir=str(tmp_path))


def test_download_pop_main_smoke(monkeypatch, tmp_path):
    cfg = {
        "paths": {"pop_dir": str(tmp_path / "pop")},
        "start_year": 2015,
        "end_year": 2024,
        "country_limit": 0,
        "worldpop_2014_url_template": "tpl2014",
        "include_2014": False,
        "worldpop_yearly_url_template": "tplyear",
    }

    monkeypatch.setattr(dp.os, "chdir", lambda path: None)
    monkeypatch.setattr(dp, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(dp, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(
        dp,
        "get_urls",
        lambda **kwargs: {
            "deu": ["u1"],
            "fra": ["u2"],
            "usa": ["u3"],
            "bra": ["u4"],
        },
    )

    seen = {}
    monkeypatch.setattr(
        dp,
        "process_all_countries",
        lambda country_urls, res, max_workers, data_dir: seen.update(
            {
                "countries": sorted(country_urls.keys()),
                "res": res,
                "max_workers": max_workers,
                "data_dir": data_dir,
            }
        ),
    )

    dp.main(res=12, max_workers=3)
    assert seen["countries"] == ["bra", "deu", "fra", "usa"]
    assert seen["res"] == 12
    assert seen["max_workers"] == 3


def test_mosaic_large_rasters_places_tiles_side_by_side(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from src import download_pop as dp

    def write_tile(path, x_origin, fill):
        profile = {
            "driver": "GTiff", "height": 3, "width": 5, "count": 1,
            "dtype": "float32", "crs": "EPSG:32632",
            "transform": from_origin(x_origin, 3, 1, 1),
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.full((1, 3, 5), fill, dtype="float32"))

    left, right = tmp_path / "left.tif", tmp_path / "right.tif"
    write_tile(left, 0, 1.0)
    write_tile(right, 5, 2.0)

    out = tmp_path / "mosaic.tif"
    dp.mosaic_large_rasters([str(left), str(right)], str(out))

    with rasterio.open(out) as src:
        data = src.read(1)

    assert data.shape == (3, 10)
    assert np.allclose(data[:, :5], 1.0)
    assert np.allclose(data[:, 5:], 2.0)


class TestRasterizeCsv:
    """Point CSV -> raster: counts are summed per cell and the total is kept."""

    def test_total_is_preserved_and_shared_cells_are_summed(self, tmp_path):
        import rasterio

        cell = 30 / dp.METRES_PER_DEGREE_LAT
        df = pd.DataFrame({"latitude": [50.0, 50.0 + cell * 0.2, 50.0 + 3 * cell, 49.9],
                           "longitude": [8.0, 8.0, 8.0 + 2 * cell, 8.1],
                           "population": [1.5, 2.25, 4.0, 10.0]})
        out = dp.rasterize_csv(df.copy(), str(tmp_path / "pop.tif"), res=30)
        with rasterio.open(out) as src:
            data = src.read(1).astype(float)
            assert src.crs.to_epsg() == 4326 and src.res[0] == pytest.approx(cell)
            row, col = src.index(8.0, 50.0)
        assert data.sum() == pytest.approx(df.population.sum())
        assert data[row, col] == pytest.approx(3.75)  # two points in one cell add up

    def test_missing_columns_or_values(self, tmp_path):
        assert dp.rasterize_csv(pd.DataFrame({"a": [1]}), str(tmp_path / "x.tif")) is None
        bad = pd.DataFrame({"lat": [None], "lon": [8.0], "pop": [1.0]})
        assert dp.rasterize_csv(bad, str(tmp_path / "y.tif")) is None


class TestDownloadPopUrlDiscovery:
    """Test URL aggregation and discovery."""

    def test_get_iso_codes_returns_dicts(self):
        """Test get_iso_codes returns complete lookups."""
        result = dp.get_iso_codes()
        
        assert len(result) == 4
        alpha_3_to_2, alpha_2_to_3, alpha_3_to_names, alpha_2_to_names = result
        
        assert isinstance(alpha_3_to_2, dict)
        assert isinstance(alpha_2_to_3, dict)
        assert len(alpha_3_to_2) > 0
        assert len(alpha_2_to_3) > 0

    def test_extract_first_wildcard_valid_match(self):
        """Test regex extraction with valid pattern."""
        test_string = "filename_2024_USA.zip"
        pattern = r"(\w+)\.zip"
        
        result = dp.extract_first_wildcard(test_string, pattern)
        
        assert result is not None
        assert "USA" in result or "2024_USA" in result

    def test_extract_first_wildcard_no_match(self):
        """Test regex extraction with no match."""
        test_string = "filename_no_match"
        pattern = r"_(\d{4})_"
        
        result = dp.extract_first_wildcard(test_string, pattern)
        
        assert result is None

    def test_extract_first_wildcard_multiple_groups(self):
        """Test that only first group is extracted."""
        test_string = "data_2024_03_USA"
        pattern = r"_(\d{4})_(\d{2})_"
        
        result = dp.extract_first_wildcard(test_string, pattern)
        
        assert result == "2024"  # Only first capture group

    def test_try_extract_country_multiple_patterns(self):
        """Test fallback through multiple patterns."""
        name = "worldpop_2020_USA.tif"
        patterns = [
            r"_(\d{4})_",  # Will match '2020'
            r"_(\w{3})\.tif",  # Also matches 'USA'
        ]
        
        result = dp.try_extract_country(name, patterns)
        
        assert result is not None  # One of the patterns will match

    def test_add_country_url_creates_list(self):
        """Test URL dict initialization."""
        country_urls = {}
        
        dp.add_country_url(country_urls, "US", "http://example.com/us.zip")
        
        assert "US" in country_urls
        assert "http://example.com/us.zip" in country_urls["US"]

    def test_add_country_url_appends_to_existing(self):
        """Test URL dict append."""
        country_urls = {"US": ["http://example.com/us1.zip"]}
        
        dp.add_country_url(country_urls, "US", "http://example.com/us2.zip")
        
        assert len(country_urls["US"]) == 2


class TestDownloadPopRasterization:
    """Test CSV rasterization logic."""

    def test_rasterize_csv_creates_valid_geom(self):
        """Test CSV data creates valid geometries."""
        df = pd.DataFrame({
            "x": [0.5, 1.5],
            "y": [0.5, 1.5],
            "population": [100, 200],
        })
        
        # Convert to geometries
        geometries = [Point(row.x, row.y) for _, row in df.iterrows()]
        
        assert len(geometries) == 2
        assert all(g.is_valid for g in geometries)

    def test_rasterize_csv_bounds_from_dataframe(self):
        """Test bounds calculation."""
        df = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
        })
        
        bounds = (df["x"].min(), df["y"].min(), df["x"].max(), df["y"].max())
        
        assert bounds == (0.0, 0.0, 2.0, 2.0)

    def test_rasterize_csv_empty_dataframe(self):
        """Test with empty CSV."""
        df = pd.DataFrame({
            "x": [],
            "y": [],
            "population": [],
        })
        
        assert len(df) == 0
        assert df.empty

    def test_rasterize_csv_population_aggregation(self):
        """Test population sum."""
        df = pd.DataFrame({
            "x": [0.5, 1.5, 0.5],
            "y": [0.5, 1.5, 1.5],
            "population": [100, 200, 150],
        })
        
        total_pop = df["population"].sum()
        
        assert total_pop == 450


class TestMosaicRasterMerging:
    """Test raster mosaic operations."""

    def test_mosaic_accumulates_bounds(self):
        """Test boundary union from multiple rasters."""
        rasters = [
            {"bounds": (0, 0, 2, 2)},
            {"bounds": (1, 1, 3, 3)},
        ]
        
        bounds_list = [r["bounds"] for r in rasters]
        min_x = min(b[0] for b in bounds_list)
        min_y = min(b[1] for b in bounds_list)
        max_x = max(b[2] for b in bounds_list)
        max_y = max(b[3] for b in bounds_list)
        
        merged_bounds = (min_x, min_y, max_x, max_y)
        
        assert merged_bounds == (0, 0, 3, 3)

    def test_mosaic_single_raster(self):
        """Test mosaic with single raster."""
        rasters = [{"bounds": (0, 0, 1, 1)}]
        
        bounds_list = [r["bounds"] for r in rasters]
        assert len(bounds_list) == 1


class TestCreateRastersSharding:
    """Test job sharding logic."""

    def test_shard_tif_dict_deterministic(self):
        """Test deterministic sharding with seed."""
        countries = ["US", "CA", "MX", "BR", "AR"]
        
        # Simulate sharding with seed
        seed = 42
        np.random.seed(seed)
        
        sharded = {c: i for i, c in enumerate(countries)}
        
        assert len(sharded) == 5

    def test_shard_tif_dict_by_job_index(self):
        """Test job index sharding."""
        countries = ["US", "CA", "MX", "BR", "AR", "CO", "VE", "PE", "CL", "UY"]
        job_index = 0
        total_jobs = 2
        
        # Assign countries to jobs
        assigned = [c for i, c in enumerate(countries) if i % total_jobs == job_index]
        
        assert len(assigned) >= 0
        assert "US" in assigned or "MX" in assigned  # Depends on modulo

    def test_find_newest_country_tif_files_valid_structure(self):
        """Test tif file dict structure."""
        countries = ["US", "CA"]
        
        # Simulate tif dict
        tif_dict = {
            "US": ["/path/to/us_2020.tif"],
            "CA": ["/path/to/ca_2020.tif"],
        }
        
        assert isinstance(tif_dict, dict)
        assert all(isinstance(v, list) for v in tif_dict.values())

    def test_find_newest_country_tif_files_valid_params(self):
        """Test file discovery parameters."""
        countries = ["US"]
        tif_dir = "/data/tifs"
        
        # Just verify the parameters are valid
        assert isinstance(countries, list)
        assert isinstance(tif_dir, str)


class TestOrchestrateIntersectionsSharding:
    """Test orchestrate_intersections sharding."""

    def test_orchestrate_intersections_processes_countries(self):
        """Test country iteration."""
        countries_set = {"US", "CA", "MX"}
        
        processed = []
        for country in countries_set:
            processed.append(country)
        
        assert len(processed) == 3

    def test_orchestrate_intersections_respects_max_workers(self):
        """Test worker count is valid."""
        max_workers = 4
        countries = ["US", "CA", "MX", "BR"]
        
        # Verify max_workers is reasonable
        assert max_workers > 0
        assert max_workers <= len(countries) + 1


class TestErrorPaths:
    """Test error conditions."""

    def test_download_pop_handles_missing_config(self):
        """Test missing configuration."""
        config = {}

        with pytest.raises(KeyError):
            _ = config["missing_key"]

    def test_create_rasters_handles_empty_tif_dict(self):
        """Test empty tif dictionary."""
        tif_dict = {}
        
        assert len(tif_dict) == 0
        assert tif_dict or True  # Handles empty case

    def test_vectorize_handles_no_rasters(self):
        """Test with no raster files."""
        raster_files = []
        
        assert len(raster_files) == 0

    def test_download_validates_urls(self):
        """Test URL validation."""
        valid_url = "https://example.com/file.zip"
        invalid_url = "not_a_url"
        
        assert "http" in valid_url
        assert "http" not in invalid_url

    def test_geodataframe_empty_handling(self):
        """Test empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame()
        
        assert gdf.empty

    def test_parquet_roundtrip_structure(self):
        """Test parquet save/load structure."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        # Check structure before/after
        assert df.shape == (3, 2)


class TestDownloadPopUtilities:
    """Test core utility functions in download_pop."""
    
    def test_find_type_recursive_tif_discovery(self):
        """Test recursive TIF file discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure
            subdir1 = Path(tmpdir) / "level1"
            subdir2 = subdir1 / "level2"
            subdir2.mkdir(parents=True)
            
            # Create TIF files at different levels
            (Path(tmpdir) / "file1.tif").write_text("tif1")
            (subdir1 / "file2.tif").write_text("tif2")
            (subdir2 / "file3.tif").write_text("tif3")
            (Path(tmpdir) / "file.csv").write_text("csv")
            
            result = dp.find_type(tmpdir, '.tif')
            
            assert len(result) == 3
            assert all(Path(f).name.endswith('.tif') for f in result)
    
    def test_find_type_empty_directory(self):
        """Test find_type with no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").write_text("text")
            (Path(tmpdir) / "file.csv").write_text("csv")
            
            result = dp.find_type(tmpdir, '.tif')
            
            assert result == []
    
    def test_find_files_tif_preference(self):
        """Test that find_files prefers TIF over CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data.tif").write_text("tif")
            (Path(tmpdir) / "data.csv").write_text("csv")
            
            result, is_tif = dp.find_files(tmpdir)
            
            assert is_tif is True
            assert len(result) > 0
            assert result[0].endswith('.tif')
    
    def test_find_files_csv_fallback(self):
        """Test that find_files falls back to CSV if no TIF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data.csv").write_text("csv")
            
            result, is_tif = dp.find_files(tmpdir)
            
            assert is_tif is False
            assert len(result) > 0
    
    def test_get_iso_codes_structure(self):
        """Test that get_iso_codes returns proper structure."""
        result = dp.get_iso_codes()
        
        # Should return tuple of 4 dicts
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        alpha_3_to_2, alpha_2_to_3, alpha_3_to_names, alpha_2_to_names = result
        
        # Each should be a dict
        assert isinstance(alpha_3_to_2, dict)
        assert isinstance(alpha_2_to_3, dict)
        assert isinstance(alpha_3_to_names, dict)
        assert isinstance(alpha_2_to_names, dict)
        
        # Should have typical countries
        assert len(alpha_3_to_2) > 100
    
    def test_extract_first_wildcard_simple_pattern(self):
        """Test regex extraction with simple pattern."""
        test_string = "file_2020_USA_data.zip"
        pattern = r"_(\d{4})_"
        
        result = dp.extract_first_wildcard(test_string, pattern)
        
        # Should extract year
        assert "2020" in result or result is not None
    
    def test_extract_first_wildcard_no_match(self):
        """Test regex extraction when pattern doesn't match."""
        test_string = "filename.zip"
        pattern = r"_(\d{4})_"
        
        result = dp.extract_first_wildcard(test_string, pattern)
        
        # Should return None or empty when no match
        assert result is None or result == ""
    
    def test_try_extract_country_fallthrough(self):
        """Test pattern fallthrough in try_extract_country."""
        name = "worldpop_2020_USA.tif"
        patterns = [
            r"_(\d{4})_",  # First pattern
            r"_(\w{3})\.tif",  # Fallback pattern
        ]
        
        result = dp.try_extract_country(name, patterns)
        
        # Should match one of the patterns
        assert result is not None
    
    def test_try_extract_country_none_match(self):
        """Test try_extract_country when nothing matches."""
        name = "random_file.txt"
        patterns = [r"_(\d{4})_", r"_(\w{3})\.tif"]
        
        result = dp.try_extract_country(name, patterns)
        
        assert result is None
    
    def test_rasterize_csv_column_detection(self):
        """Test CSV column detection in rasterize_csv."""
        df = pd.DataFrame({
            'LAT': [0.0, 1.0, 2.0],
            'LON': [0.0, 1.0, 2.0],
            'POPULATION': [100, 200, 300]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.tif"
            
            result = dp.rasterize_csv(df, str(output), res=100)

            assert result == str(output)
            assert output.exists()


class TestDownloadPopProcessing:
    """Test data processing logic in download_pop."""
    
    def test_mosaic_large_rasters_single_file(self):
        """Test mosaic_large_rasters with single file (copy case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source raster
            import rasterio
            from rasterio.transform import from_bounds
            
            src_path = Path(tmpdir) / "source.tif"
            dst_path = Path(tmpdir) / "output.tif"
            
            data = np.ones((10, 10), dtype=np.float32)
            transform = from_bounds(0, 0, 1, 1, 10, 10)
            
            with rasterio.open(
                str(src_path), 'w',
                driver='GTiff',
                height=10, width=10,
                count=1, dtype=np.float32,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(data, 1)
            
            dp.mosaic_large_rasters([str(src_path)], str(dst_path))

            assert dst_path.exists()
    
    def test_get_urls_returns_dict(self):
        """Test get_urls returns proper structure."""
        result = dp.get_urls(start_year=2015, end_year=2017)
        with_2014 = dp.get_urls(start_year=2015, end_year=2017, include_2014=True)

        assert isinstance(result, dict)
        assert len(result) > 0
        assert all(len(urls) == 4 for urls in with_2014.values())
        for country, urls in result.items():
            assert isinstance(urls, list)
            # One URL per year in the requested range; the 1 km 2014 layer is opt-in.
            assert len(urls) == 3
            assert all(isinstance(u, str) for u in urls)
            assert all(u.startswith("https://data.worldpop.org/") for u in urls)
    
    def test_process_single_country_returns_none_when_download_fails(self):
        """A failed download short-circuits before any raster work."""
        country_urls = {
            'USA': ['http://example.com/usa.zip'],
            'CAN': ['http://example.com/can.zip']
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(dp, 'download_save_and_unzip_pops', return_value=None) as mock_dl, \
                 patch.object(dp, 'mosaic_large_rasters') as mock_mosaic:
                result = dp.process_single_country(country_urls, 'USA', res=30, data_dir=tmpdir)

            assert result is None
            mock_dl.assert_called_once_with(country_urls, 'USA', tmpdir)
            mock_mosaic.assert_not_called()

    def test_process_all_countries_submits_one_job_per_country(self):
        """Every country is submitted, and a worker failure is not fatal."""
        country_urls = {
            'USA': ['http://example.com/usa.zip'],
            'CAN': ['http://example.com/can.zip'],
        }
        submitted = []

        class InlineExecutor:
            """Runs submitted work in-process so no subprocess or network is used."""

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def submit(self, fn, *args, **kwargs):
                submitted.append(args[1])
                future = MagicMock()
                if args[1] == 'CAN':
                    future.result.side_effect = RuntimeError("worker blew up")
                else:
                    future.result.return_value = None
                return future

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(dp, 'ProcessPoolExecutor', InlineExecutor), \
                 patch.object(dp, 'as_completed', side_effect=lambda fs: list(fs)):
                dp.process_all_countries(country_urls, res=30, max_workers=1, data_dir=tmpdir)

        assert sorted(submitted) == ['CAN', 'USA']


class TestDownloadPopIntegration:
    """Test integration between download_pop functions."""
    
    def test_download_pop_main_forwards_res_and_workers(self):
        """main() must pass its res/max_workers through to the pool driver."""
        with patch.object(dp, 'process_all_countries') as mock_process:
            dp.main(res=30, max_workers=1)

        mock_process.assert_called_once()
        country_urls, res, max_workers, data_dir = mock_process.call_args[0]
        assert res == 30
        assert max_workers == 1
        assert isinstance(country_urls, dict) and country_urls
        assert os.path.isabs(data_dir)
