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

from wwtp import create_voronoi as cv
from wwtp import geo_utils
from wwtp import pipelines
from wwtp.data_merge import final_data_merge
from wwtp.industrial_analysis.find_unconnected_industrial_areas import find_unconnected_areas
from wwtp.pop_at_risk_river_calculations.create_rasters import polygon_raster_sign_from_gdf
from wwtp.pop_at_risk_river_calculations.find_unserved_pop import add_buffers_to_WWTP
from wwtp import service_areas as sa


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

    got = sa.assign_sites_streaming(grid, sites, np.ones(2), sa.default_distance_multiplicative, 0)

    d = np.sqrt(((grid[:, None, :] - sites[None]) ** 2).sum(-1))
    expected = d.argmin(axis=1)
    decided = np.abs(d[:, 0] - d[:, 1]) > 1e-6  # ignore exact ties on the bisector
    assert np.array_equal(got[decided], expected[decided])


def test_B1_normalisation_is_isotropic():
    a = np.array([[0.0, 0.0], [10.0, 2.0]])
    a_norm, _ = sa.normalize_plane(a, (0.0, 0.0))
    # x and y are divided by the same scalar (the larger extent, 10)
    assert a_norm[1] == pytest.approx([1.0, 0.2])


# --------------------------------------------------------------------------- #
# B2  additive distance must rank points inside a site's disc first           #
# --------------------------------------------------------------------------- #
def test_B2_additive_distance_is_monotonic_in_true_distance():
    xs = np.linspace(0.0, 100.0, 401)
    a = np.column_stack([xs, np.zeros_like(xs)])
    d = sa.default_distance_additive(a, (0.0, 0.0), weight=0.5, factor=0)
    assert np.all(np.diff(d) >= 0)
    # the site itself is closer than any point just outside its disc
    assert d[0] < d[np.searchsorted(xs, 50.25)]


# --------------------------------------------------------------------------- #
# B3  grid spacing and cell placement                                         #
# --------------------------------------------------------------------------- #
def test_B3_grid_spacing_equals_the_step_used_to_scale_contours():
    r = sa.create_ranges(0.0, 250.0, 100.0)
    assert np.allclose(np.diff(r), 100.0)
    assert r[0] == 0.0 and r[-1] >= 250.0


def test_B3_rasterio_cells_are_centred_on_their_grid_node():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True  # node (200, 200) with step 100 and origin (0, 0)
    (poly,) = sa.extract_contours_rasterio(mask, 100, 0.0, 0.0)
    assert poly.bounds == pytest.approx((150.0, 150.0, 250.0, 250.0))


# --------------------------------------------------------------------------- #
# B4  overlaps are cut from the smaller polygon                               #
# --------------------------------------------------------------------------- #
def test_B4_overlap_is_removed_from_the_smaller_polygon():
    big, small = box(0, 0, 10, 10), box(8, 0, 12, 4)
    out = sa.resolve_polygon_overlaps(gpd.GeoDataFrame(geometry=[big, small]))
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
    out = sa.cluster_points(df, 500)
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
        raster,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 1, 0.01, 0.01),
        nodata=-99999,
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
# Service radius and Voronoi weight: from plant size in people, calibrated on #
# US census population in EPA sewersheds (replaces B9's NND x k rule, the     #
# area ceiling and the footprint-only radius)                                 #
# --------------------------------------------------------------------------- #
FOOTPRINT_TO_PEOPLE = {
    "intercept": 1.369,
    "slope": 0.738,
    "unsized_people": 2014.0,
    "country_offsets": {},
    "unsized_people_by_country": {},
}
RADIUS_KW = dict(
    buffer=9000,
    dynamic_buffering=True,
    max_buffer=50000,
    size_measure="reported",
    footprint_to_people=FOOTPRINT_TO_PEOPLE,
    reported_size_columns=["POP_SERVED", "EPA POP 2022"],
    reported_size_skip={"POP_SERVED": {"QUAL_POP": [4]}},
    radius_exponent=0.35,
    radius_at_10k_people_km=3.75,
    radius_by_connection=None,
    connected_share={},
    country_column="ISO_2",
    size_scale={},
)


def _plants(xs, footprint, lagoon=False, **columns):
    data = {"footprint_m2": [footprint] * len(xs), "lagoon_detected": [lagoon] * len(xs), "ISO_2": ["ZZ"] * len(xs)}
    data.update({k: v if isinstance(v, list) else [v] * len(xs) for k, v in columns.items()})
    return gpd.GeoDataFrame(data, geometry=[Point(x, 0) for x in xs], crs=32632)


def test_radius_grows_with_people_as_calibrated():
    small = sa.calculate_buffer(_plants([0], 1_000, POP_SERVED=10_000), None, **RADIUS_KW)[0]
    large = sa.calculate_buffer(_plants([0], 1_000, POP_SERVED=1_000_000), None, **RADIUS_KW)[0]
    assert small == pytest.approx(3750)
    assert large / small == pytest.approx(100**0.35)  # 100x people -> 5x radius


def test_reported_size_wins_and_footprint_fills_the_gaps():
    plants = _plants(
        [0, 1, 2, 3], 10_000, POP_SERVED=[50_000, None, 7_000, None], **{"EPA POP 2022": [None, 20_000, 1, None]}
    )
    people = sa.plant_size_people(plants, **RADIUS_KW)
    from_footprint = 10 ** (1.369 + 0.738 * 4)
    assert people == pytest.approx([50_000, 20_000, 7_000, from_footprint])
    only_footprint = sa.plant_size_people(plants, **{**RADIUS_KW, "size_measure": "footprint"})
    assert only_footprint == pytest.approx([from_footprint] * 4)


def test_hydrowaste_11km_estimates_are_not_used_as_size():
    plants = _plants([0, 1], 10_000, POP_SERVED=900_000, QUAL_POP=[4, 2])
    people = sa.plant_size_people(plants, **RADIUS_KW)
    assert people[0] == pytest.approx(10 ** (1.369 + 0.738 * 4)) and people[1] == 900_000


def test_radius_does_not_depend_on_neighbours():
    alone = sa.calculate_buffer(_plants([0], 5_000), None, **RADIUS_KW)
    crowded = sa.calculate_buffer(_plants([0, 300, 600, 900], 5_000), None, **RADIUS_KW)
    assert np.allclose(crowded, alone[0])


def test_lagoons_and_undetected_plants_get_the_size_free_people():
    lagoon = sa.calculate_buffer(_plants([0], 400_000, lagoon=True), None, **RADIUS_KW)[0]
    undetected = sa.calculate_buffer(_plants([0], 0), None, **RADIUS_KW)[0]
    assert lagoon == undetected == pytest.approx(3750 * (2014 / 1e4) ** 0.35)


def test_plant_size_has_country_offsets_unsized_values_and_a_national_scale():
    """Footprint estimate x 10 ** offset of the country; its own size-free people; then every size,
    reported or estimated, x the country's scale. Countries without a value keep the general ones."""
    plants = _plants([0, 1, 2, 3, 4, 5], 10_000, POP_SERVED=[None, None, None, None, 50_000, 50_000])
    plants["ISO_2"] = ["AA", "BB", "AA", "BB", "AA", "BB"]
    plants["footprint_m2"] = [10_000, 10_000, 0, 0, 10_000, 10_000]
    to_people = {**FOOTPRINT_TO_PEOPLE, "country_offsets": {"AA": 0.5}, "unsized_people_by_country": {"AA": 90_000.0}}
    fit = 10 ** (1.369 + 0.738 * 4)
    people = sa.plant_size_people(plants, **{**RADIUS_KW, "footprint_to_people": to_people})
    assert people == pytest.approx([fit * 10**0.5, fit, 90_000, 2014, 50_000, 50_000])
    scaled = sa.plant_size_people(plants, **{**RADIUS_KW, "footprint_to_people": to_people, "size_scale": {"BB": 0.5}})
    assert scaled == pytest.approx([fit * 10**0.5, fit / 2, 90_000, 1007, 50_000, 25_000])
    footprint = sa.plant_size_people(
        plants, **{**RADIUS_KW, "footprint_to_people": to_people, "size_measure": "footprint", "size_scale": {"AA": 2}}
    )
    assert footprint == pytest.approx([2 * fit * 10**0.5, fit, 180_000, 2014, 2 * fit * 10**0.5, fit])
    # a factor k on the sizes of a country is a factor k ** exponent on its radii
    radius = sa.calculate_buffer(plants, None, **{**RADIUS_KW, "footprint_to_people": to_people})
    doubled = sa.calculate_buffer(
        plants, None, **{**RADIUS_KW, "footprint_to_people": to_people, "size_scale": {"AA": 2.0}}
    )
    assert doubled / radius == pytest.approx([2**0.35, 1, 2**0.35, 1, 2**0.35, 1])


def test_radius_never_exceeds_max_buffer():
    assert sa.calculate_buffer(_plants([0], 1e9), None, **RADIUS_KW)[0] == 50000


def test_unknown_size_measure_is_rejected():
    with pytest.raises(ValueError, match="size_measure"):
        sa.plant_size_people(_plants([0], 1), **{**RADIUS_KW, "size_measure": "capacity"})


def test_size_weights_give_large_plants_proportionally_more_area():
    """Two plants 20 km apart, 100x and 1x people: the boundary sits 10x closer to
    the small one (multiplicatively weighted Voronoi with sqrt(people) weights)."""
    plants = _plants([0, 20_000], 1, POP_SERVED=[1_000_000, 10_000])
    plants["WASTE_ID"] = [1, 2]
    plants["weights"] = 0.5
    kw = {**RADIUS_KW, "max_buffer": 40_000, "radius_at_10k_people_km": 40.0}  # both circles reach past the other plant
    regions, _ = sa.weighted_voronoi(
        plants,
        "WASTE_ID",
        None,
        scale_weights=True,
        n_points=200,
        buffering=True,
        buffer_fn_kwargs=kw,
        size_weights=True,
    )
    regions = regions.to_crs(32632).set_index("WASTE_ID")
    # on the axis between the plants: the point at d/1000 = (20000-d)/100 -> d = 18182 m
    assert regions.geometry[1].contains(Point(17_500, 0)) and regions.geometry[2].contains(Point(19_000, 0))
    unweighted, _ = sa.weighted_voronoi(
        plants, "WASTE_ID", None, scale_weights=False, n_points=200, buffering=True, buffer_fn_kwargs=kw
    )
    assert unweighted.to_crs(32632).set_index("WASTE_ID").geometry[2].contains(Point(11_000, 0))


def test_radius_scale_follows_the_connected_share_of_the_country():
    """r10k = min(max_km, exp(intercept + slope logit(C))); countries without a share keep the fixed value."""
    plants = _plants([0, 10_000, 20_000, 30_000], 1, POP_SERVED=[10_000] * 4)
    plants["ISO_2"] = ["AA", "BB", "CC", "ZZ"]
    kw = {
        **RADIUS_KW,
        "radius_by_connection": {"intercept": 1.456, "slope": 0.166, "max_km": 5.5},
        "connected_share": {"AA": 0.0, "BB": 0.5, "CC": 0.97},
    }
    radius = sa.calculate_buffer(plants, None, **kw)
    # a plant of 10,000 people: its radius is r10k itself
    assert radius == pytest.approx([2000, 4289, 5500, 3750], rel=2e-3)  # clipped share, middle, cap, no share
    big = _plants([0], 1, POP_SERVED=[40_000])
    big["ISO_2"] = ["CC"]
    big["POP_SERVED"] = [40_000]
    assert sa.calculate_buffer(big, None, **kw)[0] == pytest.approx(5500 * 4**0.35, rel=1e-6)


def test_size_weights_need_the_multiplicative_distance():
    plants = _plants([0, 20_000], 1, POP_SERVED=[1_000_000, 10_000])
    plants["WASTE_ID"] = [1, 2]
    plants["weights"] = 0.5
    with pytest.raises(ValueError, match="multiplicative"):
        sa.weighted_voronoi(
            plants,
            "WASTE_ID",
            None,
            scale_weights=True,
            distance_fn=sa.default_distance_additive,
            buffer_fn_kwargs=RADIUS_KW,
            size_weights=True,
        )


def test_starter_reads_the_radius_for_the_configured_size_measure():
    from wwtp.starter import load_config

    cfg = load_config(script_name="create_voronoi")
    kw = cfg["calculate_buffer_kwargs"]
    assert cfg["weight_method"] == "size" and cfg["weight_type"] == "sz" and cfg["buffer_path_token"] == "rep"
    assert kw["size_measure"] == "reported" and kw["radius_exponent"] == pytest.approx(0.3)
    assert kw["radius_at_10k_people_km"] == pytest.approx(3.75)
    assert kw["radius_by_connection"]["max_km"] == pytest.approx(5.75) and kw["country_column"] == "ISO_2"
    assert 0 <= min(kw["connected_share"].values()) and max(kw["connected_share"].values()) <= 1
    assert kw["connected_share"]["NO"] > 0  # the codes are text: YAML must not read Norway as false
    other = load_config(script_name="create_voronoi", size_measure="footprint")
    assert other["calculate_buffer_kwargs"]["radius_exponent"] == pytest.approx(0.5)
    assert other["buffer_path_token"] == "fp"
    with pytest.raises(ValueError, match="size_measure"):
        load_config(script_name="create_voronoi", size_measure="capacity")
    with pytest.raises(ValueError, match="needs weight_func 'mult'"):
        load_config(script_name="create_voronoi", weight_func="add")


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

    out = add_buffers_to_WWTP(
        unserved,
        str(plants),
        wwtp_buffer=50_000,
        zoom_level=8,
        max_workers=1,
        is_parallel=False,
        wwtp_country_col="ISO_2",
    )

    assert len(out) == 1
    assert out.geometry.iloc[0].equals(island)
    assert out["pop_sum"].sum() == 5000


def test_R1_distance_test_is_one_buffer_in_metres(tmp_path):
    island = box(8.0, 50.0, 8.0 + 0.1 * KM_LON_50N, 50.001)
    unserved = gpd.GeoDataFrame({"pop_sum": [10]}, geometry=[island], crs=4326)
    near, far = tmp_path / "near.gpkg", tmp_path / "far.gpkg"
    _write_plants(near, [45], ["DE"])  # inside 50 km
    _write_plants(far, [60], ["DE"])  # outside 50 km, inside the old 2 x 50 km

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
    assert area == pytest.approx(np.pi * 1000.0**2, rel=0.01)


# --------------------------------------------------------------------------- #
# R4  one country per feature                                                 #
# --------------------------------------------------------------------------- #
@needs_spatial
def test_R4_country_join_returns_one_row_per_feature(tmp_path):
    countries = pd.DataFrame(
        {
            "country": ["DE", "DE", "FR"],
            "class": ["land", "maritime", "land"],
            "geometry": [to_wkb(box(5, 47, 15, 55)), to_wkb(box(4.9, 46.9, 15.1, 55.1)), to_wkb(box(-5, 42, 5, 51))],
        }
    )
    path = tmp_path / "countries.parquet"
    countries.to_parquet(path)

    features = gpd.GeoDataFrame(
        {"WASTE_ID": [1, 2, 3]},
        geometry=[Point(10, 50), box(4, 48, 9, 49), Point(-30, 0)],  # inside DE; mostly DE; in the sea
        crs=4326,
    )
    out = sa.intersects_with_country_db(
        features.copy(), str(path), polygon_country_col="country", output_country_col="ISO_2"
    )

    assert out["WASTE_ID"].tolist() == [1, 2, 3]
    assert out["ISO_2"].tolist()[:2] == ["DE", "DE"]
    assert pd.isna(out["ISO_2"].iloc[2])
    assert not {"LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"} & set(out.columns)


# --------------------------------------------------------------------------- #
# Config: every section of the shipped config.yaml must resolve               #
# --------------------------------------------------------------------------- #
def test_every_config_section_resolves():
    import yaml
    from wwtp.starter import load_config

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
    for section in ("create_voronoi", "eu_comparison"):
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

    regions, points = sa.weighted_voronoi(
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
    from wwtp.annotation_scripts import NEW_02_EXTRACTOSMDATAFULL_GEOJSON as osm

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


# --------------------------------------------------------------------------- #
# R8  reported-population filter and one-to-one reference matching            #
# --------------------------------------------------------------------------- #
def test_R8_reported_population_mask_reads_numbers_and_requires_the_column():
    from wwtp.utils import reported_population_mask

    df = pd.DataFrame({"QUAL_POP": ["1.0", 1, 2.0, None, "2"]})
    assert reported_population_mask(df).tolist() == [True, True, False, False, False]
    with pytest.raises(KeyError, match="QUAL_POP"):
        reported_population_mask(pd.DataFrame({"POP_SERVED": [1]}))


def test_R8_one_reference_plant_is_matched_by_at_most_one_cell():
    from wwtp.data_merge.merge_seg_results import assign_to_nearest

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
    from wwtp.pop_at_risk_river_calculations.impact_polygons_pop import mix_downstream_concentration

    c = 8.0  # mg/l leaving each tributary; Q = 10 m3/s each, 20 m3/s below the confluence
    after_first = mix_downstream_concentration(0.0, c, 10.0, 20.0)
    after_both = mix_downstream_concentration(after_first, c, 10.0, 20.0)

    assert after_both == pytest.approx(c)  # was 2c: concentrations were added undiluted


def test_R6_dilution_never_concentrates_and_handles_missing_discharge():
    from wwtp.pop_at_risk_river_calculations.impact_polygons_pop import mix_downstream_concentration

    assert mix_downstream_concentration(1.0, 4.0, 30.0, 20.0) == pytest.approx(5.0)  # Q_down < Q_up: no gain
    assert mix_downstream_concentration(0.0, 4.0, np.nan, 0.0, least_discharge_cms=0.5) == pytest.approx(4.0)


def test_R6_velocity_and_temperature_are_parameters():
    from wwtp.pop_at_risk_river_calculations import impact_polygons_pop as ipp

    assert ipp.calculate_kt(0.0, water_temp_equator_c=20.0) == pytest.approx(0.23)
    from wwtp.starter import load_config, get_runtime_params

    params = get_runtime_params(load_config(script_name="impact_polygons_pop"))
    assert params["velocity_m_s"] == 1.0 and params["water_temp_equator_c"] == 28.0


# --------------------------------------------------------------------------- #
# R9  population mosaics: nodata never counted, overlaps not summed, one year  #
# --------------------------------------------------------------------------- #
def _write_pop_tile(path, data, left, top, res=1.0, nodata=-99999.0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(left, top, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def test_R9_mosaic_ignores_nodata_and_does_not_sum_overlaps(tmp_path):
    from wwtp.download_pop import mosaic_large_rasters

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
    from wwtp.download_pop import mosaic_large_rasters

    _write_pop_tile(tmp_path / "a.tif", np.ones((2, 2)), 0, 2, res=1.0)
    _write_pop_tile(tmp_path / "b.tif", np.ones((2, 2)), 2, 2, res=0.5)
    with pytest.raises(ValueError, match="resolution"):
        mosaic_large_rasters([str(tmp_path / "a.tif"), str(tmp_path / "b.tif")], str(tmp_path / "o.tif"))


def test_R9_years_are_mosaicked_separately_and_2014_is_opt_in():
    from wwtp.download_pop import get_urls, group_rasters_by_year

    groups = group_rasters_by_year(
        ["x/deu_pop_2020_CN_100m.tif", "x/deu_pop_2021_CN_100m.tif", "x/deu_ppp_2014_1km.tif", "x/notes.tif"]
    )
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

    groups, boxes = sa.dissolve_overlapping_geometries_fast(sites, radius)

    side_m = np.sqrt(boxes.to_crs(6933).area.to_numpy())  # equal-area CRS
    np.testing.assert_allclose(side_m, 2 * radius, rtol=0.01)  # one central UTM zone: off by > 2x at the edges
    assert sorted(map(sorted, groups)) == [[0], [1], [2]]


# --------------------------------------------------------------------------- #
# R15  unserved basin parts: basin minus ALL its service areas, once per basin #
# --------------------------------------------------------------------------- #
@needs_spatial
def test_R15_basin_difference_is_computed_once_against_all_regions():
    from wwtp.pop_at_risk_river_calculations.find_diff_pop import find_differences

    basin = gpd.GeoDataFrame({"HYBAS_ID": [1]}, geometry=[box(8.0, 49.0, 8.4, 49.2)], crs="EPSG:4326")
    regions = gpd.GeoDataFrame(  # two plants' service areas in the same basin
        {"HYBAS_ID": [1, 1], "WASTE_ID": [10, 20]},
        geometry=[box(8.0, 49.0, 8.1, 49.2), box(8.1, 49.0, 8.2, 49.2)],
        crs="EPSG:4326",
    )

    diff = find_differences(basin, regions, is_parallel=False)

    assert len(diff) == 1  # was 2: one "difference" per plant
    got_km2 = diff.to_crs(6933).area.iloc[0] / 1e6
    expected_km2 = gpd.GeoSeries([box(8.2, 49.0, 8.4, 49.2)], crs="EPSG:4326").to_crs(6933).area.iloc[0] / 1e6
    assert got_km2 == pytest.approx(expected_km2, rel=0.01)  # was the whole basin, twice
    assert not diff.geometry.iloc[0].contains(Point(8.05, 49.1))  # served by plant 10


def test_industrial_section_has_every_key_run_voronoi_approach_reads():
    """The mock run stopped with KeyError: 'temp_voronoi_overwrite' here."""
    from wwtp.starter import load_config

    cfg = load_config(script_name="find_unconnected_industrial_areas")
    for key in (
        "temp_voronoi_overwrite",
        "overwrite_existing",
        "n_points",
        "threshold",
        "max_workers",
        "calculate_buffer_fn",
        "calculate_area_fn",
        "flush_size",
    ):
        assert key in cfg and cfg[key] is not None, key


def test_find_diff_pop_can_resolve_its_country_layer():
    """HydroBASINS has no country column; find_diff_pop now joins the boundary layer."""
    from wwtp.starter import load_config

    cfg = load_config(script_name="find_diff_pop")
    assert cfg["paths"]["overture"] and cfg["country_boundary_column"]


def test_plants_without_location_are_left_out_at_load(tmp_path, caplog):
    """Cluster run 2026-11-11: 5,042 plants with WWTP_exists False have no geometry. They reached
    create_voronoi, sat in no overlap group, and approach 0 stopped with KeyError: 53881."""
    path = tmp_path / "plants.gpkg"
    gpd.GeoDataFrame(
        {"WASTE_ID": [1, 2, 3], "WWTP_exists": [None, "False", "True"], "geometry": [Point(0, 0), None, Point(1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(path, driver="GPKG")
    with caplog.at_level("INFO"):
        plants = geo_utils.read_located_plants(str(path))
    assert plants["WASTE_ID"].tolist() == [1, 3]
    assert "1 of 3 plants have no location" in caplog.text


def test_a_geometry_that_meets_nothing_is_its_own_overlap_group():
    sites = gpd.GeoDataFrame(
        {"some_id": [0, 1, 2], "geometry": [Point(10, 50), Point(10.0001, 50), Point()]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    groups, _ = sa.dissolve_overlapping_geometries_fast(sites, 1000)
    assert sorted(sorted(g) for g in groups) == [[0, 1], [2]]
