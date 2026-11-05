from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from src import create_voronoi


pytestmark = pytest.mark.unit


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
SITES_PATH = SAMPLE_DIR / "backbone_sites.geojson"
WATERSHEDS_PATH = SAMPLE_DIR / "backbone_watersheds.geojson"
COUNTRIES_PATH = SAMPLE_DIR / "backbone_countries.geojson"

BUFFER_KWARGS = {
    "buffer": 10000,
    "dynamic_buffering": True,
    "max_buffer": 50000,
    "radius_exponent": 0.286,
    "radius_coefficient_km": 1.025,
    "radius_unsized_km": 9.85,
}


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _ImmediateExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))


def _load_sample_layers():
    sites = gpd.read_file(SITES_PATH)
    watersheds = gpd.read_file(WATERSHEDS_PATH)
    countries = gpd.read_file(COUNTRIES_PATH)
    return sites, watersheds, countries


def _prep_group_with_weights(group_gdf):
    prepared = create_voronoi.calculate_area(group_gdf.copy(), only_round=False)
    prepared = create_voronoi.create_weights(prepared, sigma=3, percent_threshold=10, method="linear")
    return prepared


def test_sample_files_exist_and_readable():
    assert SITES_PATH.exists()
    assert WATERSHEDS_PATH.exists()
    assert COUNTRIES_PATH.exists()

    sites, watersheds, countries = _load_sample_layers()
    assert len(sites) == 6
    assert len(watersheds) == 2
    assert len(countries) == 2


def test_weighted_voronoi_runs_on_sample_group():
    sites, watersheds, countries = _load_sample_layers()

    group = sites[sites["HYBAS_ID"] == 9001].copy().reset_index(drop=True)
    group = _prep_group_with_weights(group)

    utm = create_voronoi.estimate_utm_crs(group)
    group_utm = group.to_crs(utm)
    clip_utm = watersheds[watersheds["HYBAS_ID"] == 9001].to_crs(utm)
    country_utm = countries[countries["country"] == "DE"].to_crs(utm)

    region_df, point_df = create_voronoi.weighted_voronoi(
        group_utm,
        col="HYBAS_ID",
        country_clip=country_utm,
        scale_weights=True,
        clipping=clip_utm,
        n_points=80,
        distance_fn=create_voronoi.default_distance_multiplicative,
        scipy_true=False,
        cv2_true=False,
        centroid_points=True,
        buffering=True,
        threshold=150,
        calculate_buffer_fn=create_voronoi.calculate_buffer,
        buffer_fn_kwargs=BUFFER_KWARGS,
        site_id_col="WASTE_ID",
    )

    assert region_df is not None
    assert point_df is not None
    assert len(point_df) > 0
    assert "geometry" in region_df.columns
    assert region_df.crs.to_epsg() == 4326


def test_orchestrate_voronoi_runs_on_sample_data(monkeypatch):
    sites, watersheds, countries = _load_sample_layers()

    monkeypatch.setattr(create_voronoi, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(create_voronoi, "as_completed", lambda futures: futures)

    region_df, point_df = create_voronoi.orchestrate_voronoi_weights(
        sites,
        "HYBAS_ID",
        countries,
        workers=2,
        scale_weights=True,
        clipping=watersheds,
        n_points=70,
        distance_fn=create_voronoi.default_distance_multiplicative,
        scipy_true=False,
        cv2_true=False,
        centroid_points=True,
        buffering=True,
        threshold=150,
        sigma=3,
        percent_threshold=10,
        area_fn=create_voronoi.calculate_area,
        area_fn_kwargs={"only_round": False},
        method="linear",
        output_path=None,
        overwrite=False,
        flush_size=1,
        calculate_buffer_fn=create_voronoi.calculate_buffer,
        buffer_fn_kwargs=BUFFER_KWARGS,
        site_country_col="ISO_2",
        country_boundary_col="country",
        site_id_col="WASTE_ID",
    )

    assert region_df is not None
    assert point_df is not None
    assert len(point_df) > 0
    assert "geometry" in region_df.columns
    assert region_df.crs.to_epsg() == 4326


def test_intersection_and_dissolve_backbone_steps():
    sites, watersheds, _ = _load_sample_layers()

    joined = create_voronoi.intersect_with_polygons_parallelized(
        sites.copy(),
        watersheds.copy(),
        cols=["HYBAS_ID", "country"],
        use_duckdb=False,
        max_workers=2,
    )
    assert joined is not None
    assert len(joined) == len(sites)

    # Build overlapping polygons to exercise dissolve path with some_id metadata.
    overlap = gpd.GeoDataFrame(
        {
            "some_id": ["A", "B", "C"],
            "geometry": [
                watersheds.geometry.iloc[0].buffer(0.005),
                watersheds.geometry.iloc[0].buffer(0.008),
                watersheds.geometry.iloc[1].buffer(0.006),
            ],
        },
        geometry="geometry",
        crs=watersheds.crs,
    )

    dissolve_result = create_voronoi.dissolve_overlapping_geometries_fast(overlap, radius=0.01, convex=False)

    if isinstance(dissolve_result, tuple):
        _, dissolved_gdf = dissolve_result
    else:
        dissolved_gdf = dissolve_result

    assert dissolved_gdf is not None
    assert len(dissolved_gdf) > 0


def test_calculate_buffer_with_sample_weights():
    sites, _, _ = _load_sample_layers()
    prepared = _prep_group_with_weights(sites.copy())

    buffers = create_voronoi.calculate_buffer(prepared, prepared["weights"].to_numpy(), **BUFFER_KWARGS)

    assert isinstance(buffers, np.ndarray)
    assert len(buffers) == len(prepared)
    assert np.isfinite(buffers).all()
    assert (buffers > 0).all() and (buffers <= BUFFER_KWARGS["max_buffer"]).all()
