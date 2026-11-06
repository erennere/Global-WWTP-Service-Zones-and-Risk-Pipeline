"""The mock world contains the situations the before/after run relies on."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio

from wwtp.geo_utils import nearest_distances_m
from wwtp.tests.fixtures.mock_world import COUNTRIES, build_mock_world

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_mock_world(tmp_path_factory.mktemp("mock") / "data", seed=0)


def test_every_layer_is_written_and_readable(world):
    for key, path in world.items():
        if key == "pop_tif_dir":
            continue
        if path.endswith(".parquet"):
            assert len(pd.read_parquet(path))
        elif path.endswith(".gpkg"):
            assert len(gpd.read_file(path))


def test_same_seed_gives_the_same_world(world, tmp_path):
    again = build_mock_world(tmp_path / "data", seed=0)
    a = gpd.read_file(world["annotated_all_filepath"])
    b = gpd.read_file(again["annotated_all_filepath"])
    assert a.geometry.equals(b.geometry)


def test_sites_cover_the_audit_situations(world):
    sites = gpd.read_file(world["annotated_all_filepath"])
    basins = gpd.read_file(world["watershed"])
    assert sites["WASTE_ID"].isna().any() and sites["QUAL_POP"].isna().any()  # new plants (R8, R12)
    assert set(sites["category_number"]) >= {1, 2, 4, 6}  # industrial and mixed use (R10)
    distances, _ = nearest_distances_m(sites, sites, k=2)
    assert (distances[:, 1] < 500).sum() >= 2  # duplicates inside the clustering threshold (B6)
    per_basin = gpd.sjoin(sites, basins, predicate="within").groupby("HYBAS_ID").size()
    assert (per_basin == 1).any()  # a single-plant basin (B9)
    crosses = basins.geometry.bounds.pipe(lambda b: (b.minx < 8.0) & (b.maxx > 8.0))
    assert crosses.any()  # a basin across the border (R4)


def test_population_rasters_have_nodata_and_two_years(world):
    for iso3, *_ in COUNTRIES.values():
        totals = []
        for year in (2020, 2024):
            path = f"{world['pop_tif_dir']}/{iso3}/{iso3}_pop_{year}_CN_100m_R2024B_v1.tif"
            with rasterio.open(path) as src:
                data = src.read(1, masked=True)
                assert data.mask.any()  # the lake (R9)
                totals.append(float(data.sum()))
        assert totals[1] > totals[0] > 0


def test_river_network_is_a_valid_tree_with_confluences(world):
    rivers = gpd.read_file(world["rivershed"])
    ids = set(rivers["HYRIV_ID"])
    assert set(rivers["NEXT_DOWN"]) - {0} <= ids
    inflows = rivers[rivers["NEXT_DOWN"] != 0].groupby("NEXT_DOWN").size()
    assert (inflows >= 2).any()  # confluences (R6)
    outlets = rivers.loc[rivers["NEXT_DOWN"] == 0, "HYRIV_ID"]
    assert set(rivers["MAIN_RIV"]) == set(outlets)
    downstream = rivers.set_index("HYRIV_ID")["DIS_AV_CMS"]
    joined = rivers[rivers["NEXT_DOWN"] != 0]
    assert np.all(downstream.loc[joined["NEXT_DOWN"]].to_numpy() >= joined["DIS_AV_CMS"].to_numpy() - 1e-9)
