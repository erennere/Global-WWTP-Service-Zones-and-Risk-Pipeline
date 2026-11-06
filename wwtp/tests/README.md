# Test Suite

The test suite uses tiny synthetic geometries, in-memory rasters, and mocked I/O so development stays fast and independent of the repository's large geospatial inputs.

## Running the suite

Install test dependencies from the package root:

```bash
cd wwtp
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

## Layout

One file per pipeline module (`test_<module>.py`), plus:

| layer | files | checks |
|---|---|---|
| oracle | `test_*_oracle.py` | results against answers computed another way: exact Euclidean Voronoi, the Apollonius circle, raster totals, 499/501 m at 0/50/70 deg, plume decay, flux at confluences |
| property | `hypothesis` cases inside the oracle files | invariants over random inputs (translation, duplicates, scalar = vectorized) |
| regression | `test_regressions.py` | one test per audit finding (B1-B9, R1-R15) and the service-area rule |
| config | `test_config.py` | every section resolves; bad values fail at load |
| integration | `integration/test_mock_world_chain.py` | all 13 stages through their CLIs on the mock world (`fixtures/mock_world.py`), with hand-off invariants (about 30 s) |

Coverage is measured on pipeline code only (`.coveragerc`: no tests, no `wwtp/maps`), with branches, and fails below the floor:

```bash
pytest --cov --cov-report=term-missing
```

The before/after harness (`harness/mock_before_after.py`) runs two git revisions on the same mock world and tabulates their outputs.

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