"""Regression tests for the result-changing bugs found in the November 2026 audit.

Each test is named after its finding ID in the audit doc and checks the
answer against something computed independently (exact Euclidean Voronoi,
a known raster total, a known distance), so it fails on the old code and
passes on the fix.
"""
from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely import to_wkb
from shapely.geometry import Point, box

from src import create_voronoi as cv
from src import geo_utils
from src import pipelines
from src.data_merge import final_data_merge
from src.industrial_analysis.find_unconnected_industrial_areas import find_unconnected_areas
from src.pop_at_risk_river_calculations.create_rasters import polygon_raster_sign_from_gdf
from src.pop_at_risk_river_calculations.find_unserved_pop import add_buffers_to_WWTP


pytestmark = pytest.mark.unit

DUCKDB_SPATIAL = True
try:  # the R1 country-join path and R4 need the DuckDB spatial extension
    import duckdb

    duckdb.connect().execute("LOAD spatial")
except Exception:  # pragma: no cover - depends on the local DuckDB install
    try:
        duckdb.connect().execute("INSTALL spatial; LOAD spatial")
    except Exception:
        DUCKDB_SPATIAL = False

needs_spatial = pytest.mark.skipif(not DUCKDB_SPATIAL, reason="DuckDB spatial extension not available")

# One degree of longitude at 50 deg N, in metres (sphere)
KM_LON_50N = 1 / (111.195 * np.cos(np.radians(50.0)))


# --------------------------------------------------------------------------- #
# B1  normalize_plane must not stretch one axis                               #
# --------------------------------------------------------------------------- #
def test_B1_equal_weights_reproduce_euclidean_voronoi_on_a_non_square_grid():
    sites = np.array([[2000.0, 500.0], [6000.0, 1500.0]])
    xs, ys = np.meshgrid(np.arange(0, 10001, 50.0), np.arange(0, 2001, 50.0))
    grid = np.column_stack([xs.ravel(), ys.ravel()])

    got = cv.assign_sites_streaming(grid, sites, np.ones(2), cv.default_distance_multiplicative, 0)

    d = np.sqrt(((grid[:, None, :] - sites[None]) ** 2).sum(-1))
    expected = d.argmin(axis=1)
    decided = np.abs(d[:, 0] - d[:, 1]) > 1e-6  # ignore exact ties on the bisector
    assert np.array_equal(got[decided], expected[decided])


def test_B1_normalisation_is_isotropic():
    a = np.array([[0.0, 0.0], [10.0, 2.0]])
    a_norm, _ = cv.normalize_plane(a, (0.0, 0.0))
    # x and y are divided by the same scalar (the larger extent, 10)
    assert a_norm[1] == pytest.approx([1.0, 0.2])


# --------------------------------------------------------------------------- #
# B2  additive distance must rank points inside a site's disc first           #
# --------------------------------------------------------------------------- #
def test_B2_additive_distance_is_monotonic_in_true_distance():
    xs = np.linspace(0.0, 100.0, 401)
    a = np.column_stack([xs, np.zeros_like(xs)])
    d = cv.default_distance_additive(a, (0.0, 0.0), weight=0.5, factor=0)
    assert np.all(np.diff(d) >= 0)
    # the site itself is closer than any point just outside its disc
    assert d[0] < d[np.searchsorted(xs, 50.25)]


# --------------------------------------------------------------------------- #
# B3  grid spacing and cell placement                                         #
# --------------------------------------------------------------------------- #
def test_B3_grid_spacing_equals_the_step_used_to_scale_contours():
    r = cv.create_ranges(0.0, 250.0, 100.0)
    assert np.allclose(np.diff(r), 100.0)
    assert r[0] == 0.0 and r[-1] >= 250.0


def test_B3_rasterio_cells_are_centred_on_their_grid_node():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True  # node (200, 200) with step 100 and origin (0, 0)
    (poly,) = cv.extract_contours_rasterio(mask, 100, 0.0, 0.0)
    assert poly.bounds == pytest.approx((150.0, 150.0, 250.0, 250.0))


# --------------------------------------------------------------------------- #
# B4  overlaps are cut from the smaller polygon                               #
# --------------------------------------------------------------------------- #
def test_B4_overlap_is_removed_from_the_smaller_polygon():
    big, small = box(0, 0, 10, 10), box(8, 0, 12, 4)
    out = cv.resolve_polygon_overlaps(gpd.GeoDataFrame(geometry=[big, small]))
    assert out[0].area == pytest.approx(100.0)
    assert out[1].area == pytest.approx(8.0)
    assert out[0].intersection(out[1]).area == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# B5  merge thresholds are metres, whatever the CRS                           #
# --------------------------------------------------------------------------- #
def test_B5_unmatched_targets_use_metres_not_degrees():
    merged = gpd.GeoDataFrame(geometry=[Point(8.68, 49.40)], crs=4326)  # Heidelberg
    targets = gpd.GeoDataFrame(
        {"name": ["Hamburg", "next door"]},
        geometry=[Point(9.99, 53.55), Point(8.68, 49.4027)],  # ~470 km and ~300 m away
        crs="OGC:CRS84",
    )
    out = final_data_merge.find_unmatched_targets(merged, targets, 500)
    assert out["name"].tolist() == ["Hamburg"]


# --------------------------------------------------------------------------- #
# B6  duplicate records of one plant must not add up their population         #
# --------------------------------------------------------------------------- #
def test_B6_create_voronoi_cluster_points_never_sums_population():
    df = gpd.GeoDataFrame(
        {"POP_SERVED": [50_000.0, 50_000.0], "weights": [1.0, 1.0]},
        geometry=[Point(0, 0), Point(100, 0)],
        crs=32632,
    )
    out = cv.cluster_points(df, 500)
    assert out["POP_SERVED"].tolist() == [50_000.0]
    assert out["n_merged_records"].tolist() == [2]


def test_B6_final_merge_cluster_points_never_sums_population():
    df = gpd.GeoDataFrame(
        {
            "meter_geometry": [Point(0, 0).wkt, Point(100, 0).wkt],
            "POP_SERVED": [50_000, 48_000],
        },
        geometry=[Point(8, 50), Point(8.001, 50)],
        crs=4326,
    )
    out = final_data_merge.cluster_points(df, 500)
    assert out["POP_SERVED"].tolist() == [50_000]


# --------------------------------------------------------------------------- #
# B7  fractional WorldPop counts must survive the signed raster               #
# --------------------------------------------------------------------------- #
def test_B7_signed_raster_keeps_fractional_population(tmp_path):
    raster = tmp_path / "pop.tif"
    with rasterio.open(
        raster, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(0, 1, 0.01, 0.01), nodata=-99999,
    ) as dst:
        dst.write(np.full((100, 100), 0.6, dtype="float32"), 1)  # 6,000 people
    served = gpd.GeoDataFrame(geometry=[box(0, 0, 0.5, 1)], crs=4326)

    _, pos, neg = polygon_raster_sign_from_gdf(str(raster), served, str(tmp_path / "out.tif"))

    assert pos == pytest.approx(3000.0, rel=1e-4)
    assert -neg == pytest.approx(3000.0, rel=1e-4)


# --------------------------------------------------------------------------- #
# B8  an industrial area spanning two service areas is connected              #
# --------------------------------------------------------------------------- #
def test_B8_straddling_industrial_area_is_connected():
    cells = gpd.GeoDataFrame(geometry=[box(0, 0, 0.01, 0.01), box(0.01, 0, 0.02, 0.01)], crs=4326)
    industrial = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(0.008, 0.004, 0.012, 0.006), box(0.05, 0.05, 0.06, 0.06)],
        crs=4326,
    )
    assert find_unconnected_areas(industrial, cells)["id"].tolist() == [2]


# --------------------------------------------------------------------------- #
# B9  one buffer formula for every site; area ceiling applies                 #
# --------------------------------------------------------------------------- #
BUFFER_KW = dict(
    buffer=9000, dynamic_buffering=True, min_buffer=2000, max_buffer=50000, area_ceiling=True,
    k_min=0.32, k_max=0.92, detection_confidence_threshold=3, k_value=0.8,
)


def _plants(xs, total_area, mean_2_nnd=np.nan):
    return gpd.GeoDataFrame(
        {
            "total_area": [total_area] * len(xs),
            "num_detection_circle": [0] * len(xs),
            "num_detection_rect": [0] * len(xs),
            "mean_2_nnd": [mean_2_nnd] * len(xs),
        },
        geometry=[Point(x, 0) for x in xs],
        crs=32632,
    )


def test_B9_area_ceiling_caps_small_plants():
    out = cv.calculate_buffer(_plants([0, 40_000], total_area=50_000), None, **BUFFER_KW)
    assert out.tolist() == [8000.0, 8000.0]  # < 100,000 m2 detected -> 8 km cap


def test_B9_area_ceiling_can_be_switched_off():
    kw = dict(BUFFER_KW, area_ceiling=False)
    out = cv.calculate_buffer(_plants([0, 40_000], total_area=50_000), None, **kw)
    assert out.min() > 8000.0


def test_B9_single_site_uses_its_neighbour_distance_like_any_other_site():
    alone = cv.calculate_buffer(_plants([0], total_area=500_000, mean_2_nnd=20_000), None, **BUFFER_KW)
    pair = cv.calculate_buffer(_plants([0, 20_000], total_area=500_000), None, **BUFFER_KW)
    assert alone[0] == pytest.approx(pair[0])


# --------------------------------------------------------------------------- #
# R1  non-served polygons keep their geometry and are written once            #
# --------------------------------------------------------------------------- #
def _write_plants(path, lons, countries):
    gpd.GeoDataFrame(
        {"ISO_2": countries},
        geometry=[Point(8.0 + lon_km * KM_LON_50N, 50.0) for lon_km in lons],
        crs=4326,
    ).to_file(path)


def test_R1_unserved_polygons_keep_their_geometry_and_population(tmp_path):
    island = box(8.0, 50.0, 8.0 + KM_LON_50N, 50.0 + 1 / 111.195)  # ~1 x 1 km
    unserved = gpd.GeoDataFrame({"pop_sum": [5000]}, geometry=[island], crs=4326)
    plants = tmp_path / "plants.gpkg"
    _write_plants(plants, [20, -20], ["DE", "FR"])  # two countries' plants, 20 km away

    out = add_buffers_to_WWTP(unserved, str(plants), wwtp_buffer=50_000, zoom_level=8,
                              max_workers=1, is_parallel=False, wwtp_country_col="ISO_2")

    assert len(out) == 1
    assert out.geometry.iloc[0].equals(island)
    assert out["pop_sum"].sum() == 5000


def test_R1_distance_test_is_one_buffer_in_metres(tmp_path):
    island = box(8.0, 50.0, 8.0 + 0.1 * KM_LON_50N, 50.001)
    unserved = gpd.GeoDataFrame({"pop_sum": [10]}, geometry=[island], crs=4326)
    near, far = tmp_path / "near.gpkg", tmp_path / "far.gpkg"
    _write_plants(near, [45], ["DE"])  # inside 50 km
    _write_plants(far, [60], ["DE"])   # outside 50 km, inside the old 2 x 50 km

    kwargs = dict(wwtp_buffer=50_000, zoom_level=8, max_workers=1, is_parallel=False, wwtp_country_col="ISO_2")
    assert len(add_buffers_to_WWTP(unserved, str(near), **kwargs)) == 1
    assert len(add_buffers_to_WWTP(unserved, str(far), **kwargs)) == 0


# --------------------------------------------------------------------------- #
# R2  metric helpers                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lat", [0.0, 50.0, 70.0])
def test_R2_buffer_meters_is_metres_at_every_latitude(lat):
    gdf = gpd.GeoDataFrame(geometry=[Point(10.0, lat)], crs=4326)
    buf = gpd.GeoSeries(geo_utils.buffer_meters(gdf, 1000.0), crs=4326)
    area = buf.to_crs(f"+proj=laea +lat_0={lat} +lon_0=10").area.iloc[0]
    assert area == pytest.approx(np.pi * 1000.0 ** 2, rel=0.01)


def test_R2_mean_2_nnd_is_in_metres():
    sites = gpd.GeoDataFrame(
        geometry=[Point(8.0, 60.0), Point(8.0 + 2 * KM_LON_50N * np.cos(np.radians(50)) / np.cos(np.radians(60)), 60.0),
                  Point(8.0, 60.0 + 2 / 111.195)],
        crs=4326,
    )
    out = pipelines._compute_mean_2_nnd_m(sites.copy())
    # every site has its two neighbours at ~2 km and ~2.8 km; EPSG:3857 would report ~2x
    assert out["mean_2_nnd"].between(2000, 2900).all()


# --------------------------------------------------------------------------- #
# R4  one country per feature                                                 #
# --------------------------------------------------------------------------- #
@needs_spatial
def test_R4_country_join_returns_one_row_per_feature(tmp_path):
    countries = pd.DataFrame({
        "country": ["DE", "DE", "FR"],
        "class": ["land", "maritime", "land"],
        "geometry": [to_wkb(box(5, 47, 15, 55)), to_wkb(box(4.9, 46.9, 15.1, 55.1)), to_wkb(box(-5, 42, 5, 51))],
    })
    path = tmp_path / "countries.parquet"
    countries.to_parquet(path)

    features = gpd.GeoDataFrame(
        {"WASTE_ID": [1, 2, 3]},
        geometry=[Point(10, 50), box(4, 48, 9, 49), Point(-30, 0)],  # inside DE; mostly DE; in the sea
        crs=4326,
    )
    out = cv.intersects_with_country_db(features.copy(), str(path), polygon_country_col="country", output_country_col="ISO_2")

    assert out["WASTE_ID"].tolist() == [1, 2, 3]
    assert out["ISO_2"].tolist()[:2] == ["DE", "DE"]
    assert pd.isna(out["ISO_2"].iloc[2])
    assert not {"LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"} & set(out.columns)


# --------------------------------------------------------------------------- #
# Config: every section of the shipped config.yaml must resolve               #
# --------------------------------------------------------------------------- #
def test_every_config_section_resolves():
    import yaml
    from src.starter import load_config

    config_path = os.path.join(os.path.dirname(cv.__file__), "config.yaml")
    with open(config_path, encoding="utf-8") as fh:
        sections = list(yaml.safe_load(fh))
    failures = {}
    for section in sections:
        try:
            load_config(section, config=config_path)
        except Exception as err:  # collect them all, then fail once
            failures[section] = f"{type(err).__name__}: {err}"
    assert failures == {}


def test_borrowed_thresholds_are_explicit():
    """Keys that used to inherit another stage's meaning are set in their own section."""
    import yaml

    config_path = os.path.join(os.path.dirname(cv.__file__), "config.yaml")
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for section in ("create_voronoi", "eu_comparison", "compare_pop_sweep_hw_eu", "sweep_ver_ranking"):
        assert raw[section]["threshold"] is not None, section
    for section in ("create_rasters", "find_pop_in_danger_pop"):
        assert raw[section]["annotations"]["max_workers"] is not None, section


# --------------------------------------------------------------------------- #
# R5  area is measured after clipping; surviving sites are matched by site id  #
# --------------------------------------------------------------------------- #
def test_R5_area_and_points_describe_the_clipped_regions():
    utm = "EPSG:32632"
    sites = gpd.GeoDataFrame(
        {
            "WASTE_ID": [1, 2, 3],
            "HYBAS_ID": [101, 101, 101],  # the group column: shared by every site
            "weights": [1.0, 1.0, 1.0],
            "POP_SERVED": [1000, 1000, 1000],
            "geometry": [Point(500_000, 5_500_000), Point(510_000, 5_500_000), Point(520_000, 5_500_000)],
        },
        geometry="geometry",
        crs=utm,
    )
    # The country boundary only covers the first site's cell
    country = gpd.GeoDataFrame({"country": ["XX"]}, geometry=[box(490_000, 5_490_000, 504_000, 5_510_000)], crs=utm)

    regions, points = cv.weighted_voronoi(
        sites,
        "HYBAS_ID",
        country,
        n_points=60,
        threshold=100,
        calculate_buffer_fn=lambda df, weights, **kwargs: np.full(len(df), 20_000.0),
        site_id_col="WASTE_ID",
    )

    assert sorted(regions["WASTE_ID"]) == [1]
    assert sorted(points["WASTE_ID"]) == [1]  # was [1, 2, 3]: matched on HYBAS_ID
    clipped_area = regions.to_crs(utm).geometry.area.to_numpy()
    np.testing.assert_allclose(regions["area"].to_numpy(), clipped_area, rtol=1e-3)
    assert regions["area"].iloc[0] <= 14_000 * 20_000 * 1.001


# --------------------------------------------------------------------------- #
# R13  the Overpass query asks for both waterway values                       #
# --------------------------------------------------------------------------- #
def test_R13_overpass_query_requests_streams_and_rivers(monkeypatch):
    from src.annotation_scripts import NEW_02_EXTRACTOSMDATAFULL_GEOJSON as osm

    sent = {}

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"elements": []}

    class _Session:
        def get(self, url, params, timeout):
            sent["q"] = params["data"]
            return _Response()

    monkeypatch.setattr(osm, "requests_session_with_retries", lambda **kwargs: _Session())
    osm.query_overpass("0,0,1,1", osm.queries, "https://example.invalid")

    assert '["waterway"="stream"]' in sent["q"]
    assert '["waterway"="river"]' in sent["q"]


def test_additive_weighting_is_not_in_the_default_sweep():
    from src.sensitivity_analysis_scripts.create_voronoi_parallel_sweep import generate_parameter_combinations

    default_funcs = {combo[4] for combo in generate_parameter_combinations()}
    assert default_funcs == {"mult", ""}
    assert {combo[4] for combo in generate_parameter_combinations("add")} == {"add"}


# --------------------------------------------------------------------------- #
# R8  reported-population filter and one-to-one reference matching            #
# --------------------------------------------------------------------------- #
def test_R8_reported_population_mask_reads_numbers_and_requires_the_column():
    from src.utils import reported_population_mask

    df = pd.DataFrame({"QUAL_POP": ["1.0", 1, 2.0, None, "2"]})
    assert reported_population_mask(df).tolist() == [True, True, False, False, False]
    with pytest.raises(KeyError, match="QUAL_POP"):
        reported_population_mask(pd.DataFrame({"POP_SERVED": [1]}))


def test_R8_one_reference_plant_is_matched_by_at_most_one_cell():
    from src.data_merge.merge_seg_results import assign_to_nearest

    utm = "EPSG:32632"
    cells = gpd.GeoDataFrame(
        {"cell": [1, 2, 3]},
        geometry=[Point(500_100, 5_500_000), Point(500_000, 5_500_050), Point(510_000, 5_500_000)],
        crs=utm,
    )
    reference = gpd.GeoDataFrame({"POP_SERVED_EU": [1000.0]}, geometry=[Point(500_000, 5_500_000)], crs=utm)

    many = assign_to_nearest(cells, reference, threshold=500)
    one = assign_to_nearest(cells, reference, threshold=500, one_to_one=True)

    assert many["POP_SERVED_EU"].notna().sum() == 2  # the old behaviour: counted twice
    matched = one[one["POP_SERVED_EU"].notna()]
    assert matched["cell"].tolist() == [2]  # the closest cell keeps the match
    assert sorted(one["cell"]) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# R6  confluences dilute by discharge; model constants come from config        #
# --------------------------------------------------------------------------- #
def test_R6_two_equal_tributaries_mix_to_their_own_concentration():
    from src.pop_at_risk_river_calculations.impact_polygons_pop import mix_downstream_concentration

    c = 8.0  # mg/l leaving each tributary; Q = 10 m3/s each, 20 m3/s below the confluence
    after_first = mix_downstream_concentration(0.0, c, 10.0, 20.0)
    after_both = mix_downstream_concentration(after_first, c, 10.0, 20.0)

    assert after_both == pytest.approx(c)  # was 2c: concentrations were added undiluted


def test_R6_dilution_never_concentrates_and_handles_missing_discharge():
    from src.pop_at_risk_river_calculations.impact_polygons_pop import mix_downstream_concentration

    assert mix_downstream_concentration(1.0, 4.0, 30.0, 20.0) == pytest.approx(5.0)  # Q_down < Q_up: no gain
    assert mix_downstream_concentration(0.0, 4.0, np.nan, 0.0, least_discharge_cms=0.5) == pytest.approx(4.0)


def test_R6_velocity_and_temperature_are_parameters():
    from src.pop_at_risk_river_calculations import impact_polygons_pop as ipp

    assert ipp.calculate_kt(0.0, water_temp_equator_c=20.0) == pytest.approx(0.23)
    from src.starter import load_config, get_runtime_params

    params = get_runtime_params(load_config(script_name="impact_polygons_pop"))
    assert params["velocity_m_s"] == 1.0 and params["water_temp_equator_c"] == 28.0


# --------------------------------------------------------------------------- #
# R9  population mosaics: nodata never counted, overlaps not summed, one year  #
# --------------------------------------------------------------------------- #
def _write_pop_tile(path, data, left, top, res=1.0, nodata=-99999.0):
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs="EPSG:4326", transform=from_origin(left, top, res, res), nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def test_R9_mosaic_ignores_nodata_and_does_not_sum_overlaps(tmp_path):
    from src.download_pop import mosaic_large_rasters

    nd = -99999.0
    left = np.array([[1.0, 2.0, nd], [3.0, 4.0, nd]])  # columns 0-2
    right = np.array([[5.0, 6.0], [7.0, 8.0]])  # columns 2-3: overlaps left's nodata column
    a, b, out = tmp_path / "a_2020.tif", tmp_path / "b_2020.tif", tmp_path / "out.tif"
    _write_pop_tile(a, left, 0, 2)
    _write_pop_tile(b, right, 2, 2)

    mosaic_large_rasters([str(a), str(b)], str(out))

    with rasterio.open(out) as src:
        values = src.read(1, masked=True)
    assert values.shape == (2, 4)
    assert float(values.sum()) == pytest.approx(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8)  # old code: -199998 added


def test_R9_mosaic_refuses_mixed_resolutions(tmp_path):
    from src.download_pop import mosaic_large_rasters

    _write_pop_tile(tmp_path / "a.tif", np.ones((2, 2)), 0, 2, res=1.0)
    _write_pop_tile(tmp_path / "b.tif", np.ones((2, 2)), 2, 2, res=0.5)
    with pytest.raises(ValueError, match="resolution"):
        mosaic_large_rasters([str(tmp_path / "a.tif"), str(tmp_path / "b.tif")], str(tmp_path / "o.tif"))


def test_R9_years_are_mosaicked_separately_and_2014_is_opt_in():
    from src.download_pop import get_urls, group_rasters_by_year

    groups = group_rasters_by_year(["x/deu_pop_2020_CN_100m.tif", "x/deu_pop_2021_CN_100m.tif", "x/deu_ppp_2014_1km.tif",
                                    "x/notes.tif"])
    assert sorted(groups, key=lambda k: (k is None, k or 0)) == [2014, 2020, 2021, None]
    urls = get_urls(start_year=2015, end_year=2016)
    assert not any("2014" in url for country_urls in urls.values() for url in country_urls)


# --------------------------------------------------------------------------- #
# R12  site IDs do not depend on row order                                     #
# --------------------------------------------------------------------------- #
def test_R12_site_ids_are_independent_of_row_order():
    gdf = gpd.GeoDataFrame(
        {"WASTE_ID": [10, 20, np.nan, np.nan]},
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)],
        crs="EPSG:4326",
    )
    gdf["WKT_WWTP"] = gdf.geometry.to_wkt()

    a = pipelines.assign_stable_site_ids(gdf, "WASTE_ID", "OLD_WASTE_ID")
    b = pipelines.assign_stable_site_ids(gdf.iloc[::-1], "WASTE_ID", "OLD_WASTE_ID")

    key = a["WKT_WWTP"]
    assert dict(zip(key, a["WASTE_ID"])) == dict(zip(b["WKT_WWTP"], b["WASTE_ID"]))
    assert dict(zip(key, a["SITE_UID"])) == dict(zip(b["WKT_WWTP"], b["SITE_UID"]))
    assert sorted(a["WASTE_ID"]) == [0, 1, 2, 3]
    assert (a["SITE_UID"] > 0).all()


# --------------------------------------------------------------------------- #
# R3  dissolve buffers are true squares in metres across a wide country        #
# --------------------------------------------------------------------------- #
def test_R3_buffer_squares_keep_their_size_far_from_the_country_centre():
    sites = gpd.GeoDataFrame(
        {"some_id": [0, 1, 2]},
        geometry=[Point(40.0, 60.0), Point(105.0, 60.0), Point(170.0, 60.0)],  # one country, 130 deg wide
        crs="EPSG:4326",
    )
    radius = 10_000.0

    groups, boxes = cv.dissolve_overlapping_geometries_fast(sites, radius)

    side_m = np.sqrt(boxes.to_crs(6933).area.to_numpy())  # equal-area CRS
    np.testing.assert_allclose(side_m, 2 * radius, rtol=0.01)  # one central UTM zone: off by > 2x at the edges
    assert sorted(map(sorted, groups)) == [[0], [1], [2]]
