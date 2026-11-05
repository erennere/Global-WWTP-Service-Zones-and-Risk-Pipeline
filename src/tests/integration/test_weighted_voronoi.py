from __future__ import annotations

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from src import create_voronoi


pytestmark = pytest.mark.integration


def _tiny_voronoi_inputs():
    sites = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1, 2],
            "HYBAS_ID": [101, 101],
            "weights": [0.4, 0.6],
            "geometry": [Point(0, 0), Point(2000, 0)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    clip = gpd.GeoDataFrame(
        {"geometry": [box(-1000, -1000, 3000, 1000)]},
        geometry="geometry",
        crs="EPSG:3857",
    )
    country = gpd.GeoDataFrame(
        {"country": ["DE"], "geometry": [box(-1000, -1000, 3000, 1000)]},
        geometry="geometry",
        crs="EPSG:3857",
    )
    return sites, clip, country


@pytest.mark.parametrize(
    ("distance_fn", "scale_weights", "weights"),
    [
        (create_voronoi.default_distance_multiplicative, True, [0.4, 0.6]),
        (create_voronoi.default_distance_additive, False, [0.5, 0.5]),
    ],
)
def test_weighted_voronoi_produces_valid_non_overlapping_regions(distance_fn, scale_weights, weights):
    sites, clip, country = _tiny_voronoi_inputs()
    sites["weights"] = weights

    region_df, point_df = create_voronoi.weighted_voronoi(
        sites,
        "HYBAS_ID",
        country,
        scale_weights=scale_weights,
        clipping=clip,
        n_points=200,
        distance_fn=distance_fn,
        threshold=1,
        calculate_buffer_fn=lambda df, weights, **kwargs: np.full(len(df), 1500.0),
        site_id_col="WASTE_ID",
    )

    assert len(region_df) == 2
    assert len(point_df) == 2
    assert region_df.geometry.notna().all()
    assert region_df.geometry.is_valid.all()
    overlap_area = region_df.geometry.iloc[0].intersection(region_df.geometry.iloc[1]).area
    assert overlap_area == pytest.approx(0.0)


def test_weighted_voronoi_clips_regions_to_supplied_boundary():
    sites, clip, country = _tiny_voronoi_inputs()

    region_df, _ = create_voronoi.weighted_voronoi(
        sites,
        "HYBAS_ID",
        country,
        scale_weights=False,
        clipping=clip,
        n_points=200,
        distance_fn=create_voronoi.default_distance_multiplicative,
        threshold=1,
        calculate_buffer_fn=lambda df, weights, **kwargs: np.full(len(df), 1500.0),
        site_id_col="WASTE_ID",
    )

    clip_geom = clip.to_crs(4326).iloc[0].geometry
    union_geom = region_df.geometry.union_all()
    assert union_geom.difference(clip_geom).area == pytest.approx(0.0, abs=1e-9)