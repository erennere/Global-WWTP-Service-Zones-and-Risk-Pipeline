"""The pruned site assignment and the windowed region tracing give the same result as the full ones."""

import numpy as np
import pytest
import shapely
from shapely.ops import unary_union

from wwtp import service_areas as sa
from wwtp.service_areas import solver

pytestmark = pytest.mark.unit


def _full(distance_fn):
    """The same distance function under another name: assign_sites_streaming then compares every site."""
    return lambda a, b, weight, factor: distance_fn(a, b, weight, factor)


def _grid(rng, step=100.0, size=120, keep=0.8):
    xs = 500_000 + step * np.arange(size)
    ys = 5_000_000 + step * np.arange(size)
    xv, yv = np.meshgrid(xs, ys)
    points = np.column_stack([xv.ravel(), yv.ravel()])
    return points[rng.random(len(points)) < keep]  # a ragged clip


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("distance_fn", [sa.default_distance_multiplicative, sa.default_distance_additive])
def test_pruned_assignment_equals_comparing_every_site(seed, distance_fn, monkeypatch):
    monkeypatch.setattr(solver, "TILE_POINTS", 256)  # many tiles on a small grid
    rng = np.random.default_rng(seed)
    grid = _grid(rng)
    n_sites = 60
    # clustered sites, a few outside the grid, weights spanning two orders of magnitude
    centres = rng.uniform([500_000, 5_000_000], [512_000, 5_012_000], size=(4, 2))
    sites = centres[rng.integers(0, 4, n_sites)] + rng.normal(0, 2_500, size=(n_sites, 2))
    sites[:3] += 20_000
    if distance_fn is sa.default_distance_multiplicative:
        weights, factor = np.sqrt(rng.lognormal(9, 1.5, n_sites)), 0
    else:
        weights, factor = rng.uniform(0, 0.2, n_sites), 1.0
    points = [tuple(p) for p in sites]

    pruned = sa.assign_sites_streaming(grid, points, weights, distance_fn, factor)
    full = sa.assign_sites_streaming(grid, points, weights, _full(distance_fn), factor)
    assert np.array_equal(pruned, full)


def test_ties_go_to_the_same_site():
    """Two sites with equal weights at mirrored positions: the grid points on the bisector are ties."""
    grid = _grid(np.random.default_rng(0), keep=1.0)
    points = [(500_000.0 + 3_000, 5_000_000.0 + 6_000), (500_000.0 + 9_000, 5_000_000.0 + 6_000)]
    weights = np.ones(2)
    pruned = sa.assign_sites_streaming(grid, points, weights, sa.default_distance_multiplicative, 0)
    full = sa.assign_sites_streaming(grid, points, weights, _full(sa.default_distance_multiplicative), 0)
    assert np.array_equal(pruned, full) and (pruned == 0).sum() > (pruned == 1).sum()


def test_a_site_without_coordinates_gets_nothing():
    grid = _grid(np.random.default_rng(1))
    points = [(None, None), (505_000.0, 5_005_000.0)]
    pruned = sa.assign_sites_streaming(grid, points, np.ones(2), sa.default_distance_multiplicative, 0)
    full = sa.assign_sites_streaming(grid, points, np.ones(2), _full(sa.default_distance_multiplicative), 0)
    assert np.array_equal(pruned, full) and set(pruned) == {1}


@pytest.mark.parametrize(
    "extract",
    [sa.extract_contours_rasterio, sa.extract_contours_scipy, sa.extract_contours_cv2],
    ids=lambda f: f.__name__,
)
def test_tracing_a_region_in_its_window_gives_the_whole_grid_outline(extract):
    rng = np.random.default_rng(3)
    n_rows, n_cols, step, x0, y0 = 90, 110, 100.0, 500_000.0, 5_000_000.0
    sites = rng.uniform([0, 0], [n_cols, n_rows], size=(12, 2))
    rows, cols = np.indices((n_rows, n_cols))
    grid = np.argmin(((cols[..., None] - sites[:, 0]) ** 2 + (rows[..., None] - sites[:, 1]) ** 2), axis=-1)
    grid[rng.random(grid.shape) < 0.05] = -1  # holes, as outside a clip
    flat = np.flatnonzero(grid.ravel() >= 0)
    assignments = grid.ravel()[flat]
    windows = solver.site_cell_windows(assignments, flat, n_cols, n_rows, len(sites))
    for i, window in enumerate(windows):
        whole = extract(grid == i, step, x0, y0)
        if window is None:
            assert not whole
            continue
        r0, r1, c0, c1 = window
        part = extract(grid[r0 : r1 + 1, c0 : c1 + 1] == i, step, x0 + c0 * step, y0 + r0 * step)
        a, b = unary_union(whole), unary_union(part)
        assert shapely.symmetric_difference(a, b).area <= 1e-6 * max(a.area, 1.0)
        assert abs(a.area - b.area) <= 1e-6 * max(a.area, 1.0)
