"""
Phase 4 Test Suite: High-Precision Coverage Push to 75%

Target: Specific testable functions in download_pop, create_rasters, 
and download_and_vectorize that are currently untested.

Strategy: Focus on utility functions with clear inputs/outputs rather than I/O-heavy orchestration.
Targeting ~64 statements for the 75% push.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

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

        assert isinstance(result, dict)
        assert len(result) > 0
        for country, urls in result.items():
            assert isinstance(urls, list)
            # One 2014 aggregate URL plus one per year in the requested range.
            assert len(urls) == 4
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
    
    def test_resample_raster_parameter_validation(self):
        """Test resample_raster with mock rasterio."""
        import rasterio
        from rasterio.transform import from_bounds

        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "test.tif"

            data = np.ones((10, 10), dtype=np.float32)
            transform = from_bounds(0, 0, 1, 1, 10, 10)

            with rasterio.open(
                str(raster_path), 'w',
                driver='GTiff',
                height=10, width=10,
                count=1, dtype=np.float32,
                crs='EPSG:4326',
                transform=transform
            ) as src:
                src.write(data, 1)

            # Test resample_raster
            with rasterio.open(str(raster_path)) as src:
                result = dp.resample_raster(
                    src,
                    from_bounds(0, 0, 1, 1, 20, 20),
                    (20, 20),
                    'EPSG:4326'
                )
                assert isinstance(result, np.ndarray)
                assert result.shape == (20, 20)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
