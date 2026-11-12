"""Service areas against answers computed another way.

- equal weights: the grid solver reproduces the exact Euclidean Voronoi diagram
- two weighted sites: the boundary is the analytic Apollonius circle
- partition: cells never overlap, cover the clip area and contain their own site
- invariance: translating every site translates every cell
- duplicates: merging records of one plant never increases POP_SERVED
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from shapely.geometry import Point, box

from wwtp import service_areas as sa

pytestmark = pytest.mark.unit

UTM = 32632
STEP = 100.0  # grid step (m); the solver's resolution
FAR = dict(
    buffer=9000, dynamic_buffering=False, max_buffer=1_000_000
)  # no circle (buffering off); 9 km sets the grid extent


def _sites(xy, **cols):
    data = {"WASTE_ID": list(range(1, len(xy) + 1)), "weights": [1.0] * len(xy)}
    data.update(cols)
    return gpd.GeoDataFrame(data, geometry=[Point(x, y) for x, y in xy], crs=UTM)


def _solve(sites, clip, scale_weights=False, size_kwargs=None, size_weights=False):
    kwargs = dict(FAR, **(size_kwargs or {}))
    clipping = gpd.GeoDataFrame(geometry=[clip], crs=UTM)
    regions, _ = sa.weighted_voronoi(
        sites,
        "WASTE_ID",
        None,
        scale_weights=scale_weights,
        clipping=clipping,
        n_points=STEP,
        buffer_fn_kwargs=kwargs,
        threshold=1.0,
        size_weights=size_weights,
    )
    return regions.to_crs(UTM).set_index("WASTE_ID").geometry


def _iou(a, b):
    return a.intersection(b).area / a.union(b).area


def test_equal_weights_reproduce_the_euclidean_voronoi_diagram():
    xy = [(2_000, 3_000), (7_500, 2_000), (5_000, 8_000), (9_000, 9_000), (1_500, 8_500)]
    clip = box(0, 0, 10_000, 10_000)
    cells = _solve(_sites(xy), clip)
    exact = shapely.voronoi_polygons(shapely.multipoints([Point(p) for p in xy]), extend_to=clip)
    for i, p in enumerate(xy, start=1):
        truth = next(c for c in shapely.get_parts(exact) if c.contains(Point(p))).intersection(clip)
        assert _iou(cells[i], truth) > 0.98, i


def test_two_weighted_sites_meet_on_the_apollonius_circle():
    """d1 / w1 = d2 / w2 with w1 = 3, w2 = 1: the small site's cell is the circle
    with centre (x2 + (x2 - x1) / (k^2 - 1)) and radius k * D / (k^2 - 1), k = 3."""
    x1, x2, k = 0.0, 8_000.0, 3.0
    sites = _sites([(x1, 0), (x2, 0)], weights=[3.0, 1.0])
    clip = box(-20_000, -20_000, 30_000, 20_000)
    cells = _solve(sites, clip, scale_weights=True)
    D = x2 - x1
    centre, radius = x2 + D / (k**2 - 1), k * D / (k**2 - 1)
    circle = Point(centre, 0).buffer(radius, 256)
    assert _iou(cells[2], circle) > 0.95
    assert cells[1].contains(Point(centre - radius - 2 * STEP, 0))  # just outside the circle, toward site 1


def test_cells_partition_the_clip_area_and_contain_their_site():
    rng = np.random.default_rng(3)
    xy = rng.uniform(500, 9_500, size=(12, 2))
    clip = box(0, 0, 10_000, 10_000)
    cells = _solve(_sites([tuple(p) for p in xy]), clip)
    assert len(cells) == 12
    # no gaps, no double counting, nothing lost along the clip edge (was -2 %)
    assert sum(c.area for c in cells) == pytest.approx(clip.area, rel=1e-6)
    union = shapely.union_all(cells.values)
    assert union.area == pytest.approx(clip.area, rel=1e-6)
    for i, p in enumerate(xy, start=1):
        assert cells[i].buffer(STEP).contains(Point(p))


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(dx=st.integers(-50, 50), dy=st.integers(-50, 50))
def test_translating_every_site_translates_every_cell(dx, dy):
    """Shift by whole grid steps so the raster solver sees the identical problem."""
    xy = [(2_000, 3_000), (7_500, 2_000), (5_000, 8_000)]
    clip = box(0, 0, 10_000, 10_000)
    base = _solve(_sites(xy), clip)
    sx, sy = dx * STEP, dy * STEP
    moved = _solve(_sites([(x + sx, y + sy) for x, y in xy]), shapely.affinity.translate(clip, sx, sy))
    for i in base.index:
        assert _iou(shapely.affinity.translate(base[i], sx, sy), moved[i]) > 0.99


@settings(max_examples=40, deadline=None)
@given(
    pops=st.lists(st.one_of(st.none(), st.integers(0, 10_000_000)), min_size=1, max_size=5),
    offsets=st.lists(st.floats(0, 400), min_size=5, max_size=5),
)
def test_merging_duplicate_records_never_increases_pop_served(pops, offsets):
    xy = [(offsets[i], 0.0) for i in range(len(pops))]  # all within the 500 m threshold
    df = _sites(xy, POP_SERVED=[np.nan if p is None else float(p) for p in pops])
    merged = sa.cluster_points(df, threshold=500)
    assert len(merged) == 1
    known = [p for p in pops if p is not None]
    if known:
        assert merged["POP_SERVED"].iloc[0] == max(known)
    else:
        assert pd.isna(merged["POP_SERVED"].iloc[0])


def test_size_weights_follow_people_not_footprint():
    """Two plants with equal footprint: the one with 100x the reported people gets the larger cell."""
    size = dict(
        size_measure="reported",
        footprint_to_people={
            "intercept": 1.5,
            "slope": 0.7,
            "unsized_people": 2500,
            "country_offsets": {},
            "unsized_people_by_country": {},
        },
        reported_size_columns=["POP_SERVED"],
        reported_size_skip={},
        country_column="ISO_2",
        size_scale={},
    )
    sites = _sites(
        [(0, 0), (10_000, 0)], POP_SERVED=[1_000_000, 10_000], footprint_m2=[20_000, 20_000], ISO_2=["ZZ", "ZZ"]
    )
    clip = box(-15_000, -15_000, 25_000, 15_000)
    cells = _solve(sites, clip, scale_weights=True, size_kwargs=size, size_weights=True)
    assert cells[1].area > 20 * cells[2].area


@pytest.mark.parametrize("factor", [2, 3])
def test_scaling_every_site_and_the_grid_scales_every_cell(factor):
    """Equal weights are scale-free: scale sites, clip area and grid step together."""
    xy = [(2_000, 3_000), (7_500, 2_000), (5_000, 8_000), (9_000, 9_000)]
    clip = box(0, 0, 10_000, 10_000)
    base = _solve(_sites(xy), clip)
    sites = _sites([(x * factor, y * factor) for x, y in xy])
    clipping = gpd.GeoDataFrame(geometry=[shapely.affinity.scale(clip, factor, factor, origin=(0, 0))], crs=UTM)
    kwargs = dict(FAR, buffer=FAR["buffer"] * factor)
    scaled, _ = sa.weighted_voronoi(
        sites, "WASTE_ID", None, clipping=clipping, n_points=STEP * factor, buffer_fn_kwargs=kwargs, threshold=1.0
    )
    scaled = scaled.to_crs(UTM).set_index("WASTE_ID").geometry
    for i in base.index:
        assert _iou(shapely.affinity.scale(base[i], factor, factor, origin=(0, 0)), scaled[i]) > 0.99
