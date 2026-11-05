# Test Suite

The test suite uses tiny synthetic geometries, in-memory rasters, and mocked I/O so development stays fast and independent of the repository's large geospatial inputs.

## Running the suite

Install test dependencies from the package root:

```bash
cd src
python -m pip install -e .[test]
cd ..
pytest --tb=short -q
```

Run only unit tests:

```bash
pytest -m unit --tb=short -q
```

Run only integration tests:

```bash
pytest -m integration --tb=short -q
```

## Shared fixtures

- `tiny_points_gdf`: six synthetic WWTP points with IDs, basin IDs, country codes, served-population values, and detection metadata.
- `tiny_watershed_gdf`: two simple basin polygons with `HYBAS_ID`.
- `tiny_country_gdf`: country-boundary polygons aligned to the watershed fixture.
- `tiny_population_array`: small synthetic population raster data plus affine transform and CRS.
- `mock_cfg`: a `load_config()`-shaped config dictionary with all paths redirected into temporary directories.
- `mock_rivershed_gdf`: simple river lines with basin IDs and discharge values.

## I/O policy

- Unit tests patch GeoPackage reads and writes, CSV reads, DuckDB queries, and raster opens.
- Integration tests may use real geospatial operations on synthetic geometries and `rasterio.MemoryFile` rasters.
- Network calls and remote data access must stay mocked.

## Real data requirements

The current suite does not require any files from `data/` and does not download external datasets.