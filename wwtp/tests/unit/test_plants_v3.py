"""Plant list v3: national plants, the work list and v3 itself, on small made-up worlds."""

import json
import math
import zipfile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from wwtp.data_merge import national_plants as npl
from wwtp.data_merge import plants_v3 as p3
from wwtp.data_merge import work_list as wl
from wwtp.starter import load_config

pytestmark = pytest.mark.unit


def offset(lon, lat, east_m=0.0, north_m=0.0):
    """A point east_m / north_m metres from (lon, lat) (small distances)."""
    return Point(lon + east_m / (111_320 * math.cos(math.radians(lat))), lat + north_m / 110_574)


# --------------------------------------------------------------------------- #
# national_plants                                                             #
# --------------------------------------------------------------------------- #
def write_p22(folder, plants):
    """One prefecture zip with a P22b shapefile, as MLIT ships them."""
    rows = []
    for name, pt, treats, pop, tree in plants:
        row = {field: -1 for field in npl.P22B}
        row.update(P22b_002=name, P22b_021=treats, P22b_028=pop, TreeCode=tree, PosStat="1", ORigna="x")
        rows.append({**row, "geometry": pt})
    shp = folder / "shp"
    shp.mkdir()
    gpd.GeoDataFrame(rows, crs=npl.P22_CRS).to_file(shp / "P22b-12_13.shp", encoding=npl.P22_ENCODING)
    with zipfile.ZipFile(folder / "P22-12_13_GML.zip", "w") as z:
        for f in shp.iterdir():
            z.write(f, f"P22-12_13_GML/{f.name}")


def hw_record(waste_id, iso3, pt, name, status="Operational", qual_loc=1, pop=1000.0):
    rec = {c: "" for c in npl.HW_COLUMNS}
    rec.update(WASTE_ID=waste_id, CNTRY_ISO=iso3, WWTP_NAME=name, STATUS=status, QUAL_LOC=qual_loc,
               POP_SERVED=pop, QUAL_POP=2, LON_WWTP=pt.x, LAT_WWTP=pt.y)  # fmt: skip
    return rec


@pytest.fixture
def national_world(tmp_path):
    a, b, c = Point(150.0, -33.0), Point(151.0, -33.0), Point(152.0, -30.0)
    au = lambda oid, name, subtype: json.dumps(  # noqa: E731
        {"objectid": oid, "feature_name": name, "status": "Operational", "feature_subtype": subtype}
    )
    ca = json.dumps({npl.WSER_ID: "X1", npl.WSER_NAME: "Ottawa test"})
    registers = gpd.GeoDataFrame(
        {
            "register": ["aus_ga", "aus_ga", "aus_ga", "can_wser", "can_wser"],
            "iso2": ["AU", "AU", "AU", "CA", "CA"],
            "lon": [a.x, b.x, c.x, -75.0, -76.0],
            "lat": [a.y, b.y, c.y, 45.0, 45.0],
            "attrs": [
                au(1, "Alpha WWTP", "Wastewater Treatment Plant"),
                au(2, "Beta", "Storm Sewage Treatment Plant"),
                au(3, "Gamma Creek STP", "Wastewater Treatment Plant"),
                ca,
                ca,
            ],  # fmt: skip
            "new": [True, True, True, True, False],
        },
        geometry=[a, b, c, Point(-75.0, 45.0), Point(-76.0, 45.0)],
        crs=4326,
    )
    registers.to_file(tmp_path / "register_plants.gpkg")

    tokyo, rotorua = Point(139.7, 35.7), Point(176.25, -38.1)
    pd.DataFrame([
        hw_record(1001, "AUS", offset(a.x, a.y, 300), "Alpha"),                       # the same plant (< 1 km)
        hw_record(1002, "AUS", offset(b.x, b.y, 300), "Beta"),                        # at a left-out plant
        hw_record(1003, "AUS", offset(c.x, c.y, 10_000), "Gamma Creek"),              # same name, 10 km
        hw_record(1004, "AUS", Point(130.0, -25.0), "Outback"),                       # HydroWASTE only
        hw_record(1005, "AUS", Point(131.0, -25.0), "Old", status="Decommissioned"),  # not operational
        hw_record(2001, "JPN", offset(tokyo.x, tokyo.y, 5_000), "Tokyo shifted"),     # complete register
        hw_record(3001, "NZL", offset(rotorua.x, rotorua.y, 3_000), "Te Ngae", qual_loc=2),  # approximate
        hw_record(3002, "NZL", Point(172.6, -43.5), "Christchurch"),                  # HydroWASTE only
        hw_record(4001, "DEU", Point(9.0, 50.0), "Not a country of this stage"),
    ]).to_csv(tmp_path / "hydrowaste.csv", index=False)  # fmt: skip

    p22 = tmp_path / "p22"
    p22.mkdir()
    write_p22(p22, [("東京処理場", tokyo, 1, 5000, "T1"), ("汚泥処理", Point(139.0, 35.0), 2, 0, "T2")])

    def osm_part(osm_id, name, pt, size=0.001):
        return {"name": name, "@osmId": osm_id, "@snapshotTimestamp": "2023-09-10", "geometry": box(
            pt.x, pt.y, pt.x + size, pt.y + size)}  # fmt: skip

    gpd.GeoDataFrame([
        osm_part("way/1", "Rotorua WWTP", rotorua), osm_part("way/2", None, offset(rotorua.x, rotorua.y, 300)),
        osm_part("way/3", "Pump Station", Point(174.0, -41.0)),
    ], crs=4326).to_file(tmp_path / "osm.geojson")  # fmt: skip

    schema = {c: ["x"] for c in npl.HW_COLUMNS} | {c: ["x"] for c in [
        "ISO_2", "lat", "lon", "neigh_lat", "neigh_lon", "combined_geometry", "HW_geometry", "hydrowaste", "distance",
        "corrected_geometry", "used_corrected_geometry", "newly_found_WWTPs", "osm", "matched_osm_geometry",
        npl.WSER_NAME]}  # fmt: skip
    schema.update(POP_SERVED=[1.0], epsg=[32633.0], idx=[1])
    gpd.GeoDataFrame(schema, geometry=[Point(0, 0)], crs=4326).to_file(tmp_path / "all_merged_v2.gpkg")

    cfg = load_config(script_name="national_plants")
    cfg["paths"].update(register_plants=str(tmp_path / "register_plants.gpkg"), p22_dir=str(p22),
                        osmgeo_filepath=str(tmp_path / "osm.geojson"), hydrowaste=str(tmp_path / "hydrowaste.csv"),
                        corrected_all_filepath=str(tmp_path / "all_merged_v2.gpkg"),
                        output_dir=str(tmp_path / "national_plants"))  # fmt: skip
    return cfg


def test_national_plants_from_registers_osm_and_hydrowaste(national_world):
    out, side, left_out, counts = npl.build(national_world)
    npl.write(national_world, out, side, left_out)

    by_id = side.set_index("WASTE_ID")
    # HydroWASTE ids as the plant list writes numbers ('1001.0'), the others named after their source
    assert sorted(out["WASTE_ID"]) == sorted(
        ["1001.0", "1003.0", "1004.0", "CA_WSER_X1", "JP_P22_T1", "3001.0", "3002.0"]
    )
    assert by_id.loc["1001.0", "match_rule"] == "1 km" and by_id.loc["1001.0", "register_id"] == "AU_GA_1"
    assert by_id.loc["1003.0", "match_rule"] == "name 25 km"
    assert by_id.loc["3001.0", "match_rule"] == "approximate 5 km"
    assert by_id.loc["3001.0", "register_id"] == "NZ_OSM_way-1"  # two parts within 500 m: one plant
    assert by_id.loc["1004.0", "sources"] == "hydrowaste"
    assert counts["AU"] == {"left_out_register_records": 1, "hydrowaste_with_them": 1}  # Beta and 1002

    plants = out.set_index("WASTE_ID")
    jp = plants.loc["JP_P22_T1"]
    assert jp["POP_SERVED"] == 5000 and jp["QUAL_POP"] == "1.0"  # the register's treated population
    assert plants.loc["CA_WSER_X1", npl.WSER_NAME] == "Ottawa test"
    assert plants.loc["1001.0"].geometry.equals(Point(150.0, -33.0))  # the register's location
    assert set(left_out["WASTE_ID"]) == {"2001.0"} and "shifted duplicate" in left_out["reason"].iloc[0]
    written = gpd.read_file(f"{national_world['paths']['output_dir']}/national_plants.gpkg", layer="national_plants")
    assert list(written.columns) == list(out.columns)


# --------------------------------------------------------------------------- #
# work_list                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def work_world(tmp_path):
    de, gr, gb, us = (10.0, 50.0), (23.0, 38.0), (-1.0, 52.0), (-100.0, 40.0)
    p = {
        "moved": offset(*de),                    # code DE1 in the file, 100 m off
        "near": offset(*de, north_m=20_000),     # no code, 200 m from DE2
        "gone": offset(*de, north_m=40_000),     # no code, far from the file: removed
        "gone_gr": offset(*gr),                  # Greece: EL codes in the file
        "outside": offset(*gb),                  # not a file country: unchanged
        "sized": offset(*de, north_m=60_000),    # has tanks: not in the list
        "lagoon": offset(*us),
    }  # fmt: skip
    pop = gpd.GeoDataFrame(
        {
            "SITE_UID": [11, 12, 13, 14, 15, 16, 17],
            "idx": [0, 1, 2, 3, 4, 5, 6],
            "WASTE_ID": [1, 2, 3, 4, 5, 6, 7],
            "OLD_WASTE_ID": ["a", "b", "c", "d", "e", "f", "g"],
            "WWTP_NAME": list(p),
            "uwwCode": ["DE1", None, None, None, None, None, None],
            "ISO_2": ["DE", "DE", "DE", "GR", "GB", "DE", "US"],
            "footprint_m2": [0, 0, 0, 0, 0, 50, 0],
            "lagoon_detected": ["False", None, None, None, None, None, "True"],
            "WKT_WWTP": [g.wkt for g in p.values()],
        },
        geometry=[g.buffer(0.01) for g in p.values()],  # the service areas
        crs=4326,
    )
    pop.to_file(tmp_path / "pop.gpkg")
    merged = gpd.GeoDataFrame(
        {"idx": [0, 1, 2, 3, 4, 5, 6, 20], "uwwCode": ["DE1", None, None, None, None, None, None, "DE4"]},
        geometry=[*p.values(), offset(*de, north_m=80_000)],
        crs=4326,
    )
    merged.to_file(tmp_path / "all_merged_v2.gpkg")
    eu = gpd.GeoDataFrame(
        {
            "uwwCode": ["DE1", "DE2", "DE3", "DE4", "DE5", "DE6", "EL1"],
            "uwwName": list("ABCDEFG"),
            "lat": [0.0] * 7,
            "lon": [0.0] * 7,
        },  # fmt: skip
        geometry=[
            offset(*de, east_m=100),
            offset(*de, north_m=20_000, east_m=200),
            offset(*de, north_m=100_000),  # new
            offset(*de, north_m=80_000),  # code in all_merged_v2
            offset(*de, north_m=60_000, east_m=300),  # within 500 m of an all_merged_v2 plant
            None,  # no location
            offset(*gr, north_m=50_000),  # new, in Greece
        ],
        crs=4326,
    )
    eu.to_file(tmp_path / "eu.geojson", driver="GeoJSON")
    (tmp_path / "national").mkdir()
    gpd.GeoDataFrame(
        {"WASTE_ID": ["AU_GA_1", "1004"], "ISO_2": ["AU", "AU"], "WWTP_NAME": ["x", "y"]},
        geometry=[Point(150, -33), Point(130, -25)],
        crs=4326,
    ).to_file(tmp_path / "national" / "national_plants.gpkg", layer="national_plants")

    cfg = load_config(script_name="work_list")
    out = tmp_path / "heigit_sizes"
    cfg["paths"].update(population_layer=str(tmp_path / "pop.gpkg"), corrected_all_filepath=str(tmp_path / "all_merged_v2.gpkg"),
                        eu_file=str(tmp_path / "eu.geojson"), national_plants_dir=str(tmp_path / "national"),
                        work_list=str(out / "work_list.gpkg"), removed_plants=str(out / "removed.csv"),
                        eu_file_not_added=str(out / "not_added.csv"))  # fmt: skip
    return cfg, p, eu


def test_work_list_takes_the_eu_file_as_truth(work_world):
    cfg, p, eu = work_world
    out, removed, not_added = wl.build(cfg)
    wl.write(cfg, out, removed, not_added)

    assert out["origin"].value_counts().to_dict() == {
        "without_heigit_size": 4, "new_plant_national": 2, "new_plant_eu_file": 2}  # fmt: skip
    old = out[out["origin"] == "without_heigit_size"].set_index("WWTP_NAME")
    assert set(old.index) == {"moved", "near", "outside", "lagoon"}
    assert old.loc["moved"].geometry.equals(eu.geometry[0]) and old.loc["moved", "location_shift_m"] == pytest.approx(
        100, abs=1
    )
    assert old.loc["near"].geometry.equals(eu.geometry[1]) and old.loc["near", "eu_file_uwwCode"] == "DE2"
    assert old.loc["outside", "location_source"] == "heigit_layer" and old.loc["outside"].geometry.equals(p["outside"])
    assert old.loc["lagoon", "heigit_size_group"] == cfg["size_groups"]["lagoon"]
    assert old.loc["moved", "heigit_size_group"] == cfg["size_groups"]["nothing"]
    assert sorted(removed["SITE_UID"]) == [13, 14]  # EL codes make Greece a file country

    new = out[out["origin"] != "without_heigit_size"]
    assert list(new["idx"]) == [21, 22, 23, 24]  # after all_merged_v2's highest, national first
    assert list(new["plant_id"]) == ["AU_GA_1", "1004", "DE3", "EL1"]
    assert list(new["ISO_2"]) == ["AU", "AU", "DE", "GR"]
    assert new["OLD_WASTE_ID"].iloc[0] == "AU_GA_1" and pd.isna(new["WASTE_ID"].iloc[0])
    # DE2's plant has no code in the population layer, but sits within 500 m of a v2 plant
    within = "within 500 m of an all_merged_v2 plant"
    assert dict(zip(not_added["uwwCode"], not_added["reason"])) == {
        "DE2": within, "DE4": "code in all_merged_v2", "DE5": within}  # fmt: skip
    assert gpd.read_file(cfg["paths"]["work_list"])["idx"].tolist() == out["idx"].tolist()


# --------------------------------------------------------------------------- #
# plants_v3                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def v3_world(tmp_path):
    here = (8.0, 47.0)
    v2 = [offset(*here, north_m=10_000 * i) for i in range(7)]
    base = gpd.GeoDataFrame(
        {
            "idx": range(7),
            "WASTE_ID": [str(i) for i in range(7)],
            "WWTP_NAME": [f"plant {i}" for i in range(7)],
            "uwwCode": [None, None, "CH2", "FR3", None, None, None],
            "uwwName": [None, None, "two", "three", None, None, None],
            "ISO_2": ["CH"] * 7,
            "POP_SERVED": [None, None, None, None, None, 500.0, 900.0],
            "num_detection_circle": [0.0] * 7,
            "diameters": ["[0]"] * 7,
            "num_detection_rect": [0.0] * 7,
            "wwtp_area_rect": ["[0]"] * 7,
            "round_area": [0.0] * 7,
            "category_number": ["1"] * 7,
            "category_name": ["old"] * 7,
            "justification": ["old"] * 7,
        },
        geometry=[None, *v2[1:]],  # plant 0 has no location
        crs=4326,
    )
    base.to_file(tmp_path / "all_annotated_v2.gpkg")
    pd.DataFrame({"idx": [1]}).to_csv(tmp_path / "removed.csv", index=False)

    moved_to = offset(v2[2].x, v2[2].y, east_m=100)
    far = offset(v2[3].x, v2[3].y, east_m=100_000)  # the place of plant 3's code: another plant
    national = offset(v2[6].x, v2[6].y, east_m=200)  # a national plant at v2 plant 6's site
    eu_new = offset(*here, north_m=-50_000)
    work = gpd.GeoDataFrame(
        {
            "idx": [2, 3, 4, 10, 11],
            "origin": ["without_heigit_size"] * 3 + ["new_plant_national", "new_plant_eu_file"],
            "location_source": [
                "corrected_eu_file",
                "corrected_eu_file",
                "heigit_layer",
                "national_plants",
                "corrected_eu_file",
            ],  # fmt: skip
            "location_shift_m": [100.0, 100_000.0, None, None, None],
            "WASTE_ID": [2.0, 3.0, 4.0, None, None],
            "OLD_WASTE_ID": ["2", "3", "4", "NZ_OSM_1", None],
            "WWTP_NAME": ["plant 2", "plant 3", "plant 4", "national", None],
            "uwwCode": ["CH2", "FR3", None, None, "CH9"],
            "uwwName": ["two", "three", None, None, "nine"],
            "ISO_2": ["CH", "CH", "CH", "NZ", "CH"],
            "POP_SERVED": [None, None, None, 700.0, None],
        },
        geometry=[moved_to, far, v2[4], national, eu_new],
        crs=4326,
    )
    work.to_file(tmp_path / "work_list.gpkg")
    gpd.GeoDataFrame({"uwwCode": ["CH2"], "uwwName": ["two"], "lat": [0.0], "lon": [0.0]}, geometry=[moved_to],
                     crs=4326).to_file(tmp_path / "eu.geojson", driver="GeoJSON")  # fmt: skip

    ref_idx = [11, 2, 3, 10]  # image <row>.png shows the plant ref_idx[row]
    images = tmp_path / "images"
    images.mkdir()
    gpd.GeoDataFrame({"idx": ref_idx}, geometry=list(work.set_index("idx").geometry[ref_idx]),
                     crs=4326).to_file(images / "ref.geojson", driver="GeoJSON")  # fmt: skip
    pd.DataFrame({"img_name": ["1.png", "2.png", "0.png"], "num_detection_circle": [1, 1, 2],
                  "diameters": ["[20]", "[10]", "[10 10]"], "num_detection_rect": [0, 0, 0],
                  "wwtp_area_rect": ["[0]"] * 3, "other": [1, 2, 3]}).to_csv(tmp_path / "seg.csv", index=False)  # fmt: skip
    text = "Analysis: a\n\nDecision: {}. {}\nJustification: j"
    pd.DataFrame({"image": ["bbox_2.png", "bbox_11.png", "bbox_99.png"],
                  "gen_text": [text.format(3, "Industrial"), text.format(2, "Residential"), text.format(1, "x")]}
                 ).to_csv(tmp_path / "i2t.csv", index=False)  # fmt: skip

    cfg = load_config(script_name="plants_v3")
    cfg["paths"].update(base_filepath=str(tmp_path / "all_annotated_v2.gpkg"), work_list=str(tmp_path / "work_list.gpkg"),
                        removed_plants=str(tmp_path / "removed.csv"), eu_file=str(tmp_path / "eu.geojson"),
                        images_ref=str(images / "ref.geojson"), segmentation_results=str(tmp_path / "seg.csv"),
                        image_text_results=str(tmp_path / "i2t.csv"), output=str(tmp_path / "all_annotated_v3.gpkg"),
                        summary=str(tmp_path / "summary.csv"))  # fmt: skip
    return cfg, v2, moved_to, far, national, eu_new


def test_plants_v3_applies_the_work_list_and_the_new_images(v3_world):
    cfg, v2, moved_to, far, national, eu_new = v3_world
    plants, counts = p3.build(cfg)
    p3.write(cfg, plants, counts)

    assert plants["idx"].tolist() == list(range(len(plants)))
    # 0 (no location) and 1 (removed) out; 6 and the national plant one; the split file plant added
    assert counts["v3 plants"] == len(plants) == 7
    by = {k: plants[plants["idx_v2"] == k].iloc[0] for k in (2, 3, 4, 5, 6, 11)}

    assert by[2].geometry.equals(moved_to) and by[2]["location_source"] == "corrected_eu_file"
    assert by[2]["geometry_v2"] == v2[2].wkt
    assert by[2]["size_source"] == "new_image" and by[2]["diameters"] == "[20]"
    assert by[2]["round_area"] == pytest.approx(math.pi * 100)
    assert by[2]["category_name"] == "Industrial"

    # plant 3 carried another plant's code: it stays, without the code; the file plant is added
    assert by[3].geometry.equals(v2[3]) and pd.isna(by[3]["uwwCode"]) and by[3]["size_source"] == "v2_image"
    split = plants[plants["idx_v2"].isna()].iloc[0]
    assert split.geometry.equals(far) and split["uwwCode"] == "FR3" and split["ISO_2"] == "FR"
    assert split["diameters"] == "[10]" and pd.isna(split["WWTP_NAME"])

    assert by[4]["size_source"] == "v2_image" and by[4]["category_name"] == "old"  # no new result
    assert by[5]["origin"] == "v2" and by[5]["category_name"] == "old"
    # v2 plant 6 and the national plant share a site: the national location, the larger population
    assert by[6]["same_site_plants"] == 2 and by[6].geometry.equals(national) and by[6]["POP_SERVED"] == 900
    assert by[11]["origin"] == "new_plant_eu_file" and by[11]["diameters"] == "[10 10]"
    assert by[11]["category_name"] == "Residential" and by[11].geometry.equals(eu_new)
    summary = pd.read_csv(cfg["paths"]["summary"]).set_index("step")["plants"]
    assert summary["v3 plants"] == 7 and summary["removed (not in the corrected EU file)"] == 1


def test_plants_v3_stops_when_the_images_belong_to_another_work_list(v3_world):
    cfg = v3_world[0]
    ref = gpd.read_file(cfg["paths"]["images_ref"])
    ref.loc[1, "geometry"] = offset(ref.geometry[1].x, ref.geometry[1].y, east_m=50)
    ref.to_file(cfg["paths"]["images_ref"], driver="GeoJSON")
    with pytest.raises(ValueError, match="1 of 4 images are farther than 10 m from their plant"):
        p3.build(cfg)


def test_same_sites_use_geodesic_metres():
    pts = gpd.GeoSeries([Point(0, 60), Point(0.0089, 60), Point(0.0091, 60)], crs=4326)  # 496 m, 508 m east
    assert p3.metres(pts.iloc[[0, 0]], pts.iloc[[1, 2]]) == pytest.approx([496.6, 507.8], abs=0.5)
    assert [sorted(g) for g in p3.same_sites(gpd.GeoDataFrame(geometry=pts.iloc[[0, 1]]), 500)] == [[0, 1]]
    assert p3.same_sites(gpd.GeoDataFrame(geometry=pts.iloc[[0, 2]]), 500) == []
    assert [sorted(g) for g in p3.same_sites(gpd.GeoDataFrame(geometry=pts), 500)] == [[0, 1, 2]]  # a chain
