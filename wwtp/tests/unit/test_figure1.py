"""Figure 1 scripts take their thresholds from config.yaml (figure1)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from wwtp.starter import load_config

pytestmark = pytest.mark.unit
FIG1 = Path(__file__).resolve().parents[3] / "figures" / "fig1"


def _setting(key):
    return subprocess.run(
        [sys.executable, str(FIG1 / "fig1_config.py"), key], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_settings_come_from_config():
    cfg = load_config(script_name="figure1")
    assert _setting("min_plants") == str(cfg["min_plants"]) == "5"
    assert _setting("tag") == f"r{int(cfg['match_radius_m'])}" == "r10000"
    assert _setting("match_dir").endswith("figures/fig1/match")


def test_no_script_hardcodes_the_threshold_or_the_radius():
    for path in FIG1.rglob("*.py"):
        text = path.read_text()
        assert "default=5)" not in text, path
        assert '"r10000"' not in text and '"r5000"' not in text, path


@pytest.mark.parametrize("script", sorted(p.name for p in FIG1.rglob("*.sh")))
def test_wrappers_parse_and_use_the_config(script):
    path = next(FIG1.rglob(script))
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "fig1_config.py" in text and "scripts/" not in text


@pytest.mark.parametrize("script", sorted(str(p.relative_to(FIG1)) for p in FIG1.rglob("*.py")))
def test_scripts_compile(script):
    import py_compile

    py_compile.compile(str(FIG1 / script), doraise=True)


def test_a_unit_with_4_plants_is_hatched_and_with_5_coloured():
    sys.path.insert(0, str(FIG1))
    from fig1_config import enough_plants

    import pandas as pd

    assert not enough_plants(4) and enough_plants(5)
    assert enough_plants(pd.Series([0, 4, 5, 12])).tolist() == [False, False, True, True]


def test_every_threshold_check_goes_through_enough_plants():
    import re

    for name in ("match/03_plot.py", "match/tanks_panel.py"):
        text = (FIG1 / name).read_text()
        assert "enough_plants(" in text
        assert not re.search(r"\b(n_ref|n_shift|n_tank|n_cat|n)\s*>=\s*a\.min_", text), name


def _release_inputs():
    import importlib.util

    spec = importlib.util.spec_from_file_location("release_inputs", FIG1 / "00a_release_inputs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_basin_ids_of_any_level_are_found_and_a_clash_stops(tmp_path):
    sys.path.insert(0, str(FIG1))
    from fig1_config import pfaf_lookup

    import pandas as pd

    pd.DataFrame({"HYBAS_ID": [1090000010], "PFAF_ID": [111011001], "SUB_AREA": [1.0]}).to_csv(
        tmp_path / "hybas_lev09_v1c_attributes.csv", index=False
    )
    pd.DataFrame({"HYBAS_ID": [1060000010], "PFAF_ID": [111011]}).to_csv(
        tmp_path / "hybas_lev06_pipeline_attributes.csv", index=False
    )
    assert pfaf_lookup(tmp_path) == {1090000010: 111011001, 1060000010: 111011}
    pd.DataFrame({"HYBAS_ID": [1060000010], "PFAF_ID": [111012]}).to_csv(
        tmp_path / "hybas_lev06_v1c_attributes.csv", index=False
    )
    with pytest.raises(AssertionError, match="two PFAF_IDs"):
        pfaf_lookup(tmp_path)


def test_plants_of_a_release_file_become_the_figure_inputs(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    ri = _release_inputs()
    g = gpd.GeoDataFrame(
        {
            "WASTE_ID": [0, 1],
            "HW_WASTE_ID": ["54120.0", None],
            "uwwCode": [None, "ATTP_1-00000001"],
            "ISO_2": ["NA", "AT"],
            "CNTRY_ISO": ["NAM", "AUT"],
            "HYBAS_ID": [1060000010.0, 2060000020.0],
            "SOURCE_DS": ["HW", "UWW"],
            "num_detections": [2, 0],
            "category_name": ["Residential", None],
            "POP_SERVED": [100.0, None],
            "round_area": [50.0, 0.0],
            "rect_area": [25.0, 0.0],
            "category_number": ["6.0", None],
            "epsg": [32633.0, 32633.0],
            "2024_zonal_sum": [1000.0, 2000.0],
            "2020_zonal_sum": [900.0, 1900.0],
        },
        geometry=[Point(17.0839, -22.5609), Point(17.049136, 47.778667)],
        crs=4326,
    )
    release = tmp_path / "HeiGIT_WWTP_test.gpkg"
    g.to_file(release, layer="WWTP_LOCATIONS", driver="GPKG")
    out = tmp_path / "data"
    out.mkdir()
    (out / "heigit_points_original.csv").write_text("left by an earlier 00_fix_utm run\n")
    ri.main(["plants", str(release), str(out)])
    pts = pd.read_csv(out / "heigit_points.csv", keep_default_na=False, na_values=[""])
    assert pts.fid.tolist() == [1, 2] and pts.ISO_2.tolist() == ["NA", "AT"]  # Namibia survives
    assert pts.HW_WASTE_ID.tolist()[0] == 54120 and pd.isna(pts.HW_WASTE_ID[1])
    assert pts.HYBAS_ID.tolist() == [1060000010, 2060000020]
    assert not (out / "heigit_points_original.csv").exists()
    pop = pd.read_csv(out / "heigit_pop.csv")
    assert pop.zonal_sum_2024.tolist() == [1000.0, 2000.0] and pop.zonal_sum_2020.tolist() == [900.0, 1900.0]
    attrs = pd.read_csv(out / "heigit_attrs.csv")
    assert {"fid", "round_area", "rect_area", "category_number", "category_name"} <= set(attrs.columns)
    assert pd.read_csv(out / "heigit_epsg.csv").columns.tolist() == ["fid", "epsg"]


def test_non_served_areas_become_points_without_exact_duplicates(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    ri = _release_inputs()
    sq = box(10, 50, 10.1, 50.1)
    g = gpd.GeoDataFrame(
        {"HYBAS_ID": [2060000010] * 3, "data_idx": [1, 1, 2], "tile": ["1-2-8"] * 3, "country": ["DE"] * 3,
         "pop_sum": [5000.0, 5000.0, 7000.0]},
        geometry=[sq, sq, box(11, 50, 11.1, 50.1)],
        crs=4326,
    ).to_crs(3857)
    layer = tmp_path / "non_served.gpkg"
    g.to_file(layer, driver="GPKG")
    ri.main(["non_served", str(layer), str(tmp_path / "out" / "points.csv")])
    pts = pd.read_csv(tmp_path / "out" / "points.csv")
    assert len(pts) == 2 and pts.pop_sum.sum() == 12000.0
    assert pts.lon.between(10, 11.1).all() and pts.lat.between(50, 50.1).all()
    report = json.loads((tmp_path / "out" / "points_report.json").read_text())
    assert report["exact_duplicates_dropped"] == 1


def test_pipeline_basins_are_written_as_an_attribute_table_of_their_level(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    ri = _release_inputs()
    g = gpd.GeoDataFrame(
        {"HYBAS_ID": [4060000010, 4060000080], "PFAF_ID": [421010, 421020], "SUB_AREA": [892.0, 2796.2]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=4326,
    )
    g.to_file(tmp_path / "hydrobase_lvl6_combined.gpkg", driver="GPKG")
    ri.main(["basins", str(tmp_path / "hydrobase_lvl6_combined.gpkg"), str(tmp_path / "hb")])
    t = pd.read_csv(tmp_path / "hb" / "hybas_lev06_pipeline_attributes.csv")
    assert t.PFAF_ID.tolist() == [421010, 421020]
