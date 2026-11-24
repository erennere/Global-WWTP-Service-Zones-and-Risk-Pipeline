"""create_rasters: country totals and non-served islands per basin, on small real rasters."""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import Point, Polygon, box

from wwtp.pop_at_risk_river_calculations import create_rasters as cr

pytestmark = pytest.mark.unit
RES = 0.001
X0, Y0 = 10.0, 50.5


def _raster(path, data, nodata=-99999, x0=X0, y0=Y0, **profile):
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(x0, y0, RES, RES), nodata=nodata, **profile,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return str(path)


def _blobs(seed, h=300, w=400):
    rng = np.random.default_rng(seed)
    blob = ndimage.binary_dilation(rng.random((h, w)) < 0.004, iterations=3) & (rng.random((h, w)) < 0.8)
    return np.where(blob, rng.gamma(2.0, 3.0, (h, w)), 0).astype(np.float32), rng


def _wkb(geoms):
    return shapely.to_wkb(np.asarray(list(geoms), dtype=object))


def _islands(path, basins, served, min_pixels=9, min_pop=0, copy_dir=None):
    cp = cr.country_pass(path, _wkb(served.geometry), False, copy_dir)
    tree = shapely.STRtree(basins.geometry.values)
    tasks = cr._basin_tasks(cp, basins, tree, ["HYBAS_ID"], np.asarray(served.geometry.values, dtype=object))
    rows = [r for w, g, a, s in tasks for r in cr.basin_islands(cp["read_path"], w, g, a, s, min_pixels, min_pop)]
    return cp, rows


def _reference(path, basins, served, min_pixels=9, min_pop=0):
    """The same rule on the whole raster at once."""
    with rasterio.open(path) as src:
        data, tr = src.read(1), src.transform
    data = np.where(data > 0, data, 0)
    shape = data.shape
    srv = rasterize(((g, 1) for g in served.geometry), out_shape=shape, transform=tr, fill=0, dtype=np.uint8) > 0
    out = []
    for h, g in zip(basins.HYBAS_ID, basins.geometry):
        m = (data > 0) & ~srv & (rasterize([(g, 1)], out_shape=shape, transform=tr, fill=0, dtype=np.uint8) > 0)
        lab, k = ndimage.label(m, structure=np.ones((3, 3), bool))
        pop = np.round(np.bincount(lab[m], weights=data[m], minlength=k + 1)).astype(int)
        px = np.bincount(lab[m], minlength=k + 1)
        out += [(int(h), int(pop[i]), int(px[i])) for i in range(1, k + 1) if px[i] >= min_pixels and pop[i] > min_pop]
    return sorted(out)


def _setup(tmp_path, seed=0):
    data, rng = _blobs(seed)
    data[:20, :30] = -99999  # nodata
    path = _raster(tmp_path / "xx.tif", data)
    # a staircase border on a 5-pixel grid, as HydroBASINS borders lie on its 15" grid (never through a pixel centre)
    steps = [box(X0 - 1, Y0 - 0.3 + k * 0.015, X0 + 0.2 + (k % 3) * 0.005, Y0 - 0.3 + (k + 1) * 0.015) for k in range(20)]
    left = shapely.union_all(steps + [box(X0 - 1, Y0 - 1, X0 + 0.2, Y0 - 0.3), box(X0 - 1, Y0, X0 + 0.2, Y0 + 1)])
    right = box(X0 - 1, Y0 - 1, X0 + 2, Y0 + 1).difference(left)
    basins = gpd.GeoDataFrame({"HYBAS_ID": [1, 2]}, geometry=[left, right], crs=4326)
    served = gpd.GeoDataFrame(
        {"ISO_2": ["XX"] * 12},
        geometry=[Point(X0 + rng.random() * 0.4, Y0 - rng.random() * 0.3).buffer(0.02) for _ in range(12)],
        crs=4326,
    )
    return path, data, basins, served


@pytest.mark.parametrize("seed", [0, 1])
def test_islands_are_the_8_connected_non_served_pixel_groups_of_each_basin(tmp_path, seed):
    path, data, basins, served = _setup(tmp_path, seed)
    cp, rows = _islands(path, basins, served)
    got = sorted((r["HYBAS_ID"], r["pop_sum"], r["pixel_count"]) for r in rows)
    assert got == _reference(path, basins, served) and len(got) > 50
    # each polygon covers exactly its pixels
    assert all(round(r["geometry"].area / RES**2) == r["pixel_count"] for r in rows)


def test_country_totals_ignore_nodata_and_split_at_the_service_areas(tmp_path):
    path, data, basins, served = _setup(tmp_path)
    cp = cr.country_pass(path, _wkb(served.geometry), False)
    with rasterio.open(path) as src:
        tr = src.transform
    people = np.where(data > 0, data, 0).astype(np.float64)
    srv = rasterize(((g, 1) for g in served.geometry), out_shape=data.shape, transform=tr, fill=0, dtype=np.uint8) > 0
    assert cp["served"] == pytest.approx(people[srv].sum(), rel=1e-9)
    assert cp["unserved"] == pytest.approx(people[~srv].sum(), rel=1e-9)
    assert cp["occupied"].shape == (1, 1) and cp["occupied"].all()


def test_diagonal_pixels_form_one_island_and_the_threshold_is_on_the_whole_island(tmp_path):
    data = np.zeros((40, 40), np.float32)
    for i in range(12):
        data[5 + i, 5 + i] = 100.0  # a diagonal line: one 8-connected island of 1,200 people, 12 pixels
    data[30:33, 30:33] = 50.0  # 9 pixels, 450 people
    path = _raster(tmp_path / "d.tif", data)
    basins = gpd.GeoDataFrame({"HYBAS_ID": [7]}, geometry=[box(X0 - 1, Y0 - 1, X0 + 1, Y0 + 1)], crs=4326)
    none = gpd.GeoDataFrame({"ISO_2": []}, geometry=[], crs=4326)
    _, rows = _islands(path, basins, none, min_pixels=9, min_pop=0)
    assert sorted((r["pop_sum"], r["pixel_count"]) for r in rows) == [(450, 9), (1200, 12)]
    _, rows = _islands(path, basins, none, min_pixels=9, min_pop=1000)
    assert [(r["pop_sum"], r["pixel_count"]) for r in rows] == [(1200, 12)]


def test_a_basin_reaching_into_the_country_counts_there_whatever_country_it_was_assigned(tmp_path):
    data = np.zeros((50, 50), np.float32)
    data[10:20, 10:20] = 20.0
    path = _raster(tmp_path / "b.tif", data)
    # a basin that lies mostly outside the raster (it would have been assigned to the neighbour)
    basins = gpd.GeoDataFrame({"HYBAS_ID": [3]}, geometry=[box(X0 - 5, Y0 - 5, X0 + 0.03, Y0 + 5)], crs=4326)
    none = gpd.GeoDataFrame({"ISO_2": []}, geometry=[], crs=4326)
    _, rows = _islands(path, basins, none)
    assert [(r["HYBAS_ID"], r["pop_sum"], r["pixel_count"]) for r in rows] == [(3, 2000, 100)]


def test_a_wide_striped_raster_is_read_through_a_tiled_copy_with_the_same_result(tmp_path, monkeypatch):
    path, data, basins, served = _setup(tmp_path)
    plain = sorted((r["HYBAS_ID"], r["pop_sum"], r["pixel_count"]) for r in _islands(path, basins, served)[1])
    monkeypatch.setattr(cr, "COPY_MIN_WIDTH", 100)
    copies = tmp_path / "copies"
    copies.mkdir()
    cp, rows = _islands(path, basins, served, copy_dir=str(copies))
    assert cp["read_path"].startswith(str(copies))
    with rasterio.open(cp["read_path"]) as src:
        assert src.block_shapes[0] == (cr.BLOCK, cr.BLOCK)
    assert sorted((r["HYBAS_ID"], r["pop_sum"], r["pixel_count"]) for r in rows) == plain


def _two_countries(tmp_path):
    a, _ = _blobs(3, 200, 250)
    b, _ = _blobs(4, 150, 200)
    tifs = {"AA": _raster(tmp_path / "aa.tif", a), "BB": _raster(tmp_path / "bb.tif", b, x0=X0 + 0.3)}
    basins = gpd.GeoDataFrame(
        {"HYBAS_ID": [1, 2], "NEXT_DOWN": [0, 1], "MAIN_BAS": [1, 1]},
        geometry=[box(X0 - 1, Y0 - 1, X0 + 0.32, Y0 + 1), box(X0 + 0.32, Y0 - 1, X0 + 2, Y0 + 1)],
        crs=4326,
    )
    served = gpd.GeoDataFrame(
        {"ISO_2": ["AA", "AA", "BB"]},
        geometry=[Point(X0 + 0.1, Y0 - 0.1).buffer(0.05), Point(X0 + 0.2, Y0 - 0.05).buffer(0.03),
                  Point(X0 + 0.4, Y0 - 0.1).buffer(0.05)],
        crs=4326,
    )
    return tifs, basins, served


def test_countries_are_written_once_complete_and_a_rerun_continues(tmp_path):
    tifs, basins, served = _two_countries(tmp_path)
    stats, islands = str(tmp_path / "out" / "stats.csv"), str(tmp_path / "out" / "islands.csv")
    res = cr.process_countries(tifs, served, basins, stats, islands, max_workers=2, min_pixels=9, min_pop=50)
    assert res == {"AA": True, "BB": True}
    s = pd.read_csv(stats).set_index("country")
    i = pd.read_csv(islands)
    assert set(s.index) == {"AA", "BB"} and set(i.country) == {"AA", "BB"}
    assert (i.pop_sum > 50).all() and (i.pixel_count >= 9).all()
    assert {"HYBAS_ID", "NEXT_DOWN", "MAIN_BAS", "geometry"} <= set(i.columns)
    assert shapely.from_wkt(i.geometry.iloc[0]).geom_type in ("Polygon", "MultiPolygon")
    # islands of BB lie in basin 2 only where BB's raster reaches into it, and both basins appear
    assert set(i[i.country == "AA"].HYBAS_ID) <= {1, 2} and 2 in set(i.HYBAS_ID)
    for c, path in tifs.items():
        with rasterio.open(path) as src:
            d = src.read(1)
        assert s.loc[c, "population_total"] == pytest.approx(d[d > 0].sum(dtype=np.float64), rel=1e-6)
    # a rerun skips both; island rows of a country without stats (an interrupted run) are removed
    extra = i.iloc[:1].assign(country="CC")
    pd.concat([i, extra]).to_csv(islands, index=False)
    assert cr.process_countries(tifs, served, basins, stats, islands, max_workers=2) == {}
    assert set(pd.read_csv(islands).country) == {"AA", "BB"} and len(pd.read_csv(stats)) == 2


def test_a_failed_country_stops_the_run_after_the_others_are_written(tmp_path):
    tifs, basins, served = _two_countries(tmp_path)
    tifs["ZZ"] = str(tmp_path / "missing.tif")
    stats, islands = str(tmp_path / "stats.csv"), str(tmp_path / "islands.csv")
    with pytest.raises(RuntimeError, match="ZZ"):
        cr.process_countries(tifs, served, basins, stats, islands, max_workers=2)
    assert set(pd.read_csv(stats).country) == {"AA", "BB"}


def test_shards_are_balanced_by_raster_size(tmp_path):
    sizes = {"RU": 900, "CN": 800, "US": 700, "DE": 100, "LU": 10, "NL": 20}
    tifs = {}
    for c, n in sizes.items():
        p = tmp_path / f"{c}.tif"
        p.write_bytes(b"x" * n)
        tifs[c] = str(p)
    shards = [cr.shard_tif_dict(tifs, i, 3) for i in range(3)]
    assert sorted(c for s in shards for c in s) == sorted(sizes)
    assert [sorted(s) for s in shards] == [["RU"], ["CN", "NL"], ["DE", "LU", "US"]]
    with pytest.raises(ValueError):
        cr.shard_tif_dict(tifs, -1, 2)
    with pytest.raises(ValueError):
        cr.shard_tif_dict(tifs, 0, 0)


def test_main_check_run_reads_the_configured_year(tmp_path, monkeypatch):
    tifs, basins, served = _two_countries(tmp_path)
    layer, ws = tmp_path / "areas.gpkg", tmp_path / "basins.gpkg"
    served.to_file(layer)
    basins.to_file(ws)
    calls = {}

    def for_year(countries, tif_dir, year):
        calls["year"] = year
        return {c: p for c, p in tifs.items() if c in set(countries)}

    cfg = {
        "population_year": 2026, "threshold_value": 50, "min_pixels": 9, "served_all_touched": False,
        "annotations": {"max_workers": 2}, "figures": {"approach": "1"}, "country_output_column": "ISO_2",
        "basin_column_name": "HYBAS_ID",
        "paths": {"pop_tif_dir": str(tmp_path), "watershed": str(ws),
                  "non_served_outpath": str(tmp_path / "ns.gpkg"), "csv_output_filepath": str(tmp_path / "st.csv")},
    }
    monkeypatch.setattr(cr, "load_config", lambda **kw: cfg)
    monkeypatch.setattr(cr.provenance, "write_run_manifest", lambda *a, **k: None)
    monkeypatch.setattr(cr, "create_pop_output_paths", lambda c: {"voronoi": {"1": str(layer)}})
    monkeypatch.setattr(cr, "find_country_tif_files_for_year", for_year)
    out = tmp_path / "check"
    cr.main(["--countries", "BB", "--out-dir", str(out)])
    assert calls["year"] == 2026
    assert pd.read_csv(out / "st.csv").country.tolist() == ["BB"]
    assert set(pd.read_csv(out / "ns.csv").country) == {"BB"}


def test_the_raster_of_the_configured_year_is_picked(tmp_path):
    from wwtp.add_pop import find_country_tif_files_for_year

    folder = tmp_path / "deu"
    folder.mkdir()
    for y in (2024, 2026, 2030):
        (folder / f"deu_pop_{y}_CN_100m_R2024B_v1.tif").write_bytes(b"")
    got = find_country_tif_files_for_year(["DE", "FR"], str(tmp_path), 2026)
    assert list(got) == ["DE"] and os.path.basename(got["DE"]) == "deu_pop_2026_CN_100m_R2024B_v1.tif"


@pytest.mark.real_atomic_output
def test_shards_write_their_own_csvs_and_merge_into_one(tmp_path):
    final = str(tmp_path / "stats.csv")
    assert cr.shard_path(final, 0, 1) == final
    paths = [cr.shard_path(final, i, 3) for i in range(3)]
    assert paths[2].endswith("stats_shard2of3.csv")
    pd.DataFrame({"country": ["DE"], "population_served": [1.0]}).to_csv(paths[0], index=False)
    open(paths[1], "w").close()  # a shard whose countries had no rows
    with pytest.raises(FileNotFoundError, match="shards not finished"):
        cr.merge_shards([final], 3)
    pd.DataFrame({"country": ["NA"], "population_served": [2.0]}).to_csv(paths[2], index=False)
    cr.merge_shards([final], 3)
    assert pd.read_csv(final, keep_default_na=False)["country"].tolist() == ["DE", "NA"]  # Namibia survives


def test_an_island_gets_the_tile_of_its_representative_point():
    from wwtp.pop_at_risk_river_calculations.find_unserved_pop import _point_tile

    island = box(10.0, 50.0, 13.0, 50.2)  # spans several zoom-8 tiles
    x, y, z = map(int, _point_tile(island, 8).split("-"))
    import mercantile

    pt = island.representative_point()
    assert (x, y, z) == tuple(mercantile.tile(pt.x, pt.y, 8))
