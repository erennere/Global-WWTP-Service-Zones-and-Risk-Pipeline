"""Metric distances where degrees would mislead: a 499 m pair matches at 500 m, a 501 m pair does not,
at 0, 50 and 70 degrees, north-south and east-west."""

import geopandas as gpd
import pytest
from pyproj import Geod
from shapely.geometry import Point

from wwtp.data_merge.merge_seg_results import assign_to_nearest
from wwtp.geo_utils import buffer_meters, cluster_point_indices, nearest_distances_m

pytestmark = pytest.mark.unit

LATITUDES = (0.0, 50.0, 70.0)


GEOD = Geod(ellps="WGS84")


def _offset(lon, lat, metres, direction):
    """Point ``metres`` away on the WGS84 ellipsoid, due north or due east."""
    x, y, _ = GEOD.fwd(lon, lat, 0.0 if direction == "north" else 90.0, metres)
    return x, y


def _pair(lat, metres, direction):
    a = (8.0, lat)
    b = _offset(*a, metres, direction)
    return (
        gpd.GeoDataFrame({"id": [1]}, geometry=[Point(a)], crs=4326),
        gpd.GeoDataFrame({"ref": [2]}, geometry=[Point(b)], crs=4326),
    )


@pytest.mark.parametrize("lat", LATITUDES)
@pytest.mark.parametrize("direction", ["north", "east"])
def test_great_circle_distance_is_metres(lat, direction):
    src, tgt = _pair(lat, 500.0, direction)
    d, idx = nearest_distances_m(src, tgt)
    assert d[0] == pytest.approx(500.0, abs=0.001) and idx[0] == 0


@pytest.mark.parametrize("lat", LATITUDES)
@pytest.mark.parametrize("direction", ["north", "east"])
def test_metric_buffer_contains_499_m_but_not_501_m(lat, direction):
    centre = gpd.GeoDataFrame(geometry=[Point(8.0, lat)], crs=4326)
    ring = buffer_meters(centre, 500.0, resolution=64).iloc[0]
    assert ring.contains(Point(_offset(8.0, lat, 499.0, direction)))
    assert not ring.contains(Point(_offset(8.0, lat, 501.0, direction)))


@pytest.mark.parametrize("lat", LATITUDES)
@pytest.mark.parametrize("direction", ["north", "east"])
def test_matching_and_clustering_in_utm_split_at_500_m(lat, direction):
    for metres, expect in ((499.0, True), (501.0, False)):
        src, tgt = _pair(lat, metres, direction)
        utm = src.estimate_utm_crs()
        matched = assign_to_nearest(src.to_crs(utm), tgt.to_crs(utm), threshold=500)
        assert bool(matched["ref"].notna().iloc[0]) is expect, (lat, direction, metres)
        both = gpd.GeoSeries(list(src.geometry) + list(tgt.geometry), crs=4326).to_crs(utm)
        assert (len(cluster_point_indices(both.values, 500)) == 1) is expect, (lat, direction, metres)


@pytest.mark.parametrize("lat", LATITUDES)
@pytest.mark.parametrize("direction", ["north", "east"])
def test_unmatched_targets_split_at_500_m(lat, direction):
    from wwtp.data_merge.final_data_merge import find_unmatched_targets

    for metres, matched in ((499.0, True), (501.0, False)):
        src, tgt = _pair(lat, metres, direction)
        assert find_unmatched_targets(src, tgt, 500).empty is matched, (lat, direction, metres)
