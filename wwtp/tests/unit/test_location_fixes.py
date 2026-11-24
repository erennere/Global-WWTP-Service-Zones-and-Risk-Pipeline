"""Known location errors of plant list v3 (wwtp/data_merge/location_fixes.py), on small made-up lists."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from wwtp.data_merge import location_fixes as lf
from wwtp.data_merge import plants_v3 as p3
from wwtp.starter import load_config

pytestmark = pytest.mark.unit

FIX_COLUMNS = ["WASTE_ID", "uwwCode", "name", "wrong_lon", "wrong_lat", "action", "lon", "lat", "ISO_2",
               "location_source", "clear", "split", "add_uwwCode", "note"]  # fmt: skip


def offset(pt, east_m=0.0, north_m=0.0):
    return Point(pt.x + east_m / (111_320 * math.cos(math.radians(pt.y))), pt.y + north_m / 110_574)


def fixes(*rows):
    return pd.DataFrame([{c: "" for c in FIX_COLUMNS} | row for row in rows], columns=FIX_COLUMNS).astype(object)


@pytest.fixture
def cfg():
    return load_config(script_name="plants_v3")


def plant_list(rows, geoms):
    base = {"idx": None, "idx_v2": None, "WASTE_ID": None, "WWTP_NAME": None, "uwwCode": None, "uwwName": None,
            "uwwLoadEnteringUWWTP": None, "ISO_2": None, "POP_SERVED": None, "QUAL_POP": None, "epsg": None,
            "num_detection_circle": 0.0, "diameters": "[0]", "num_detection_rect": 0.0, "wwtp_area_rect": "[0]",
            "category_number": None, "category_name": None, "justification": None, "size_source": "v2_image",
            "origin": "v2", "location_source": "v2", "location_shift_m": None, "geometry_v2": None,
            "same_site_plants": 1}  # fmt: skip
    return gpd.GeoDataFrame([base | r for r in rows], geometry=geoms, crs=4326)


def eu_file(tmp_path, code, pt):
    path = tmp_path / "eu.geojson"
    gpd.GeoDataFrame({"uwwCode": [code], "uwwName": ["the file plant"], "uwwLoadEnteringUWWTP": [3000.0],
                      "lat": [pt.y], "lon": [pt.x]}, geometry=[pt], crs=4326).to_file(path, driver="GeoJSON")  # fmt: skip
    return str(path)


def test_a_plant_in_the_wrong_utm_zone_moves_and_its_image_stays_behind(cfg):
    wrong, right = Point(-4.081453, 52.108073), Point(25.918547, 52.108073)  # 4 zones off, as HydroWASTE 57058
    plants = plant_list(
        [{"idx": 0, "idx_v2": 0, "WASTE_ID": "57058.0", "WWTP_NAME": "belarus", "ISO_2": "GB", "POP_SERVED": 6306.0,
          "epsg": 32635.0, "num_detection_circle": 4.0, "diameters": "[28 28 28 28]", "category_name": "Residential"},
         {"idx": 1, "idx_v2": 1, "WASTE_ID": "1.0"}],
        [wrong, Point(0, 0)],
    )  # fmt: skip
    table = fixes({"WASTE_ID": "57058", "name": "HW 57058", "wrong_lon": wrong.x, "wrong_lat": wrong.y, "action": "move",
                   "lon": right.x, "lat": right.y, "ISO_2": "BY", "location_source": "v2", "split": "yes"})  # fmt: skip

    out, report = lf.apply_location_fixes(plants, table, cfg, merge=True)

    moved = out[out["WWTP_NAME"] == "belarus"].iloc[0]
    assert moved.geometry.equals(right) and moved["ISO_2"] == "BY" and moved["epsg"] == 32635.0
    assert moved["geometry_v2"] == wrong.wkt and moved["location_shift_m"] == pytest.approx(2_046_000, rel=0.01)
    assert moved["POP_SERVED"] == 6306.0 and moved["WASTE_ID"] == "57058.0"
    # what the image of the wrong place showed is not this plant's
    assert pd.isna(moved["diameters"]) and pd.isna(moved["category_name"]) and moved["size_source"] == "no_image"
    left = out[out["origin"] == "location_fix_split"].iloc[0]
    assert left.geometry.equals(wrong) and left["diameters"] == "[28 28 28 28]" and left["size_source"] == "v2_image"
    assert pd.isna(left["WASTE_ID"]) and pd.isna(left["POP_SERVED"]) and left["epsg"] == 32630.0
    assert left["idx"] == 2 and len(out) == 3 and "_location_fixed" not in out
    assert report["status"].tolist() == ["moved"]

    again, report = lf.apply_location_fixes(out, table, cfg, merge=True)
    assert report["status"].tolist() == ["already applied"] and len(again) == 3


def test_a_plant_with_another_plants_eu_code_keeps_its_place_and_the_file_plant_is_added(cfg, tmp_path):
    here, theirs = Point(18.387237, 47.184244), Point(6.385195, 47.187539)  # Szekesfehervar, Vercel
    plants = plant_list([{"idx": 0, "idx_v2": 0, "WASTE_ID": "12740.0", "WWTP_NAME": "Szekesfehervar",
                          "uwwCode": "FR060925601001", "uwwName": "VERCEL", "uwwLoadEnteringUWWTP": 14533.0,
                          "POP_SERVED": 201960.0, "num_detection_circle": 5.0}], [here])  # fmt: skip
    table = fixes({"WASTE_ID": "12740", "uwwCode": "FR060925601001", "wrong_lon": here.x, "wrong_lat": here.y,
                   "action": "keep", "ISO_2": "HU", "clear": "eu_code", "add_uwwCode": "FR060925601001"})  # fmt: skip
    eu = eu_file(tmp_path, "FR060925601001", theirs)

    out, report = lf.apply_location_fixes(plants, table, cfg, eu_file=eu, merge=True)

    kept = out[out["WWTP_NAME"] == "Szekesfehervar"].iloc[0]
    assert kept.geometry.equals(here) and pd.isna(kept["uwwCode"]) and pd.isna(kept["uwwLoadEnteringUWWTP"])
    assert kept["POP_SERVED"] == 201960.0 and kept["num_detection_circle"] == 5.0
    added = out[out["uwwCode"] == "FR060925601001"].iloc[0]
    assert added.geometry.equals(theirs) and added["origin"] == "new_plant_eu_file" and added["ISO_2"] == "FR"
    assert added["uwwLoadEnteringUWWTP"] == 3000.0 and added["size_source"] == "no_image"
    assert report.iloc[0]["added"].startswith("added")

    again, report = lf.apply_location_fixes(out, table, cfg, eu_file=eu, merge=True)
    assert report["status"].tolist() == ["already applied"] and len(again) == 2


def test_a_moved_plant_merges_with_the_plant_already_at_its_site(cfg):
    wrong, right = Point(14.2449, 48.6195), Point(20.242125, 48.617613)  # COV Jelsava
    plants = plant_list(
        [{"idx": 0, "idx_v2": 0, "WASTE_ID": "331.0", "WWTP_NAME": "COV Jelsava", "POP_SERVED": 684.0},
         {"idx": 1, "idx_v2": None, "uwwCode": "SKCAO60805425257911", "uwwName": "COV Jelsava",
          "origin": "new_plant_eu_file", "location_source": "corrected_eu_file"}],
        [wrong, offset(right, east_m=30)],
    )  # fmt: skip
    table = fixes({"WASTE_ID": "331", "wrong_lon": wrong.x, "wrong_lat": wrong.y, "action": "move", "lon": right.x,
                   "lat": right.y, "ISO_2": "SK", "location_source": "corrected_eu_file", "split": "no"})  # fmt: skip

    out, report = lf.apply_location_fixes(plants, table, cfg, merge=True)

    assert len(out) == 1
    one = out.iloc[0]
    assert one["same_site_plants"] == 2 and one["POP_SERVED"] == 684.0 and one["uwwCode"] == "SKCAO60805425257911"
    assert one["WASTE_ID"] == "331.0" and one["idx"] == 0
    assert report["status"].tolist() == ["moved", "same site: 2 plants merged"]


def test_a_wrong_hydrowaste_link_is_emptied_and_found_again_by_its_eu_code(cfg):
    wrong, right = Point(13.122223, 47.654174), Point(7.119042, 47.649499)  # WOLFERSDORF, HydroWASTE 56652 (Russia)
    plants = plant_list([{"idx": 0, "WASTE_ID": "56652.0", "WWTP_NAME": None, "uwwCode": "FR026837801946",
                          "POP_SERVED": 20287.0, "QUAL_POP": "4.0", "uwwLoadEnteringUWWTP": 3369.0}], [wrong])  # fmt: skip
    table = fixes({"WASTE_ID": "56652", "uwwCode": "FR026837801946", "wrong_lon": wrong.x, "wrong_lat": wrong.y,
                   "action": "move", "lon": right.x, "lat": right.y, "ISO_2": "FR",
                   "location_source": "corrected_eu_file", "clear": "hydrowaste", "split": "no"})  # fmt: skip

    out, report = lf.apply_location_fixes(plants, table, cfg, merge=True)
    one = out.iloc[0]
    assert one.geometry.equals(right) and pd.isna(one["WASTE_ID"]) and pd.isna(one["POP_SERVED"])
    assert one["uwwCode"] == "FR026837801946" and one["uwwLoadEnteringUWWTP"] == 3369.0
    assert lf.apply_location_fixes(out, table, cfg, merge=True)[1]["status"].tolist() == ["already applied"]


def test_a_plant_that_is_not_there_is_reported(cfg):
    plants = plant_list([{"idx": 0, "WASTE_ID": "7.0"}], [Point(1, 1)])
    table = fixes({"WASTE_ID": "8", "wrong_lon": 1, "wrong_lat": 1, "action": "move", "lon": 2, "lat": 2,
                   "ISO_2": "XX", "location_source": "v2", "split": "no"})  # fmt: skip
    out, report = lf.apply_location_fixes(plants, table, cfg)
    assert report["status"].iloc[0].startswith("not found") and out.geometry.iloc[0].equals(Point(1, 1))


def test_the_table_in_git_is_complete_and_points_the_right_way():
    table = lf.read_fixes(load_config(script_name="plants_v3")["paths"]["location_fixes"])
    assert len(table) == 21 and table["WASTE_ID"].is_unique
    moves = table[table["action"] == "move"]
    assert len(moves) == 17 and (moves["split"] == "yes").sum() == 10
    # a wrong UTM zone moves a plant by whole 6 degree zones at the same latitude (re-projected), or within about
    # 1 km of that (EU file location); one plant (HydroWASTE 53228, Argentina) was also in the wrong hemisphere
    lon, lat = moves["lon"].astype(float), moves["lat"].astype(float)
    wlon, wlat = moves["wrong_lon"].astype(float), moves["wrong_lat"].astype(float)
    flipped = lat * wlat < 0
    assert moves.loc[flipped, "WASTE_ID"].tolist() == ["53228"]
    zones = (lon - wlon)[~flipped] / 6
    assert np.allclose(zones, zones.round(), atol=0.002) and (zones.round() != 0).all()
    assert (lat - wlat)[~flipped].abs().max() < 0.01
    keeps = table[table["action"] == "keep"]
    assert (keeps["clear"] == "eu_code").all() and (keeps["add_uwwCode"] == keeps["uwwCode"]).all()
