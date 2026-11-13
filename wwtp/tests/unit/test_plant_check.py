"""wwtp.calibration.plant_check: references from polygons, the stand-in per configuration, districts, the choice."""

import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from shapely.geometry import box

from wwtp.calibration import plant_check as pc
from wwtp.calibration.connection_radius import aggregate_cells, sphere_xyz, weighted_nearest

pytestmark = pytest.mark.unit


def _raster(path, values, origin=(5.0, 52.0), cell=0.01):
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(origin[0], origin[1], cell, cell),
        "nodata": -99999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype("float32"), 1)


def _plants(lon, lat, iso2="NL", size=None):
    n = len(lon)
    size = np.full(n, 1e4) if size is None else np.asarray(size, float)
    return pd.DataFrame({"iso2": iso2, "lon": lon, "lat": lat, "uwwCode": None, "cwns": None, "size_reported": size})


def test_degrees_from_census_strings_and_localities_from_the_zip(tmp_path):
    assert pc.parse_dms("102°17'45.768\" W") == pytest.approx(-(102 + 17 / 60 + 45.768 / 3600))
    assert pc.parse_dms("21°52'47.362\" N") == pytest.approx(21 + 52 / 60 + 47.362 / 3600)
    assert np.isnan(pc.parse_dms(""))
    rows = pd.DataFrame(
        {"ENTIDAD": ["01", "01", "01", "01"], "MUN": ["000", "001", "001", "001"], "LOC": ["0000", "0001", "0002", "9998"],
         "LONGITUD": ["", '102°17\'45.768" W', '102°18\'00.000" W', ""], "LATITUD": ["", '21°52\'47.362" N', '21°53\'00.000" N', ""],
         "POBTOT": ["1000", "800", "150", "50"], "VIVPAR_HAB": ["300", "200", "40", "*"], "VPH_DRENAJ": ["250", "190", "*", "1"]}
    )  # fmt: skip
    with zipfile.ZipFile(tmp_path / "iter.zip", "w") as z:
        z.writestr("data/iter.csv", "﻿" + rows.to_csv(index=False))
    spec = {
        "member": "data/iter.csv",
        "columns": {"state": "ENTIDAD", "municipality": "MUN", "locality": "LOC", "lon": "LONGITUD",
                    "lat": "LATITUD", "people": "POBTOT", "dwellings": "VIVPAR_HAB", "drainage": "VPH_DRENAJ"},
        "aggregate_localities": ["0000", "9998", "9999"],
        "total_municipality": "000",
    }  # fmt: skip
    loc = pc.read_localities(tmp_path / "iter.zip", spec)
    # totals and grouped localities out; the locality with a confidential value ('*') out
    assert len(loc) == 1 and loc["district"].iloc[0] == "01001" and loc["drainage"].iloc[0] == 190


def test_polygons_take_our_plants_inside_or_near_and_count_their_people(tmp_path):
    values = np.ones((100, 100))
    _raster(tmp_path / "nl.tif", values)  # 5.00-6.00 E, 51.00-52.00 N, one person per cell
    polygons = gpd.GeoDataFrame(geometry=[box(5.1, 51.1, 5.2, 51.2), box(5.5, 51.5, 5.6, 51.6)], crs=4326)
    plants = pc.plant_points(
        pd.DataFrame(
            {"iso2": ["NL", "NL", "NL", "BE"], "lon": [5.15, 5.605, 5.9, 5.15], "lat": [51.15, 51.55, 51.9, 51.15],
             "uwwCode": None, "cwns": None}
        )
    )  # fmt: skip
    units, matches = pc.polygon_units(polygons, plants, "NL", 1000.0, str(tmp_path / "nl.tif"))
    assert sorted(matches["SITE_UID"]) == [0, 1]  # 350 m east of the second polygon counts; far and Belgian do not
    assert units.set_index("unit")["ref"].to_numpy() == pytest.approx([100.0, 100.0])  # 10 x 10 cells each


def _world(tmp_path, monkeypatch):
    """A country of 60 x 60 cells with eight plants and the stand-in's own people per plant for (k 2, 5 km)."""
    rng = np.random.default_rng(3)
    _raster(tmp_path / "nl.tif", rng.uniform(0, 100, (60, 60)), cell=0.02)
    monkeypatch.setattr(pc, "raster_for_year", lambda iso2, tif_dir, year: str(tmp_path / "nl.tif"))
    lon, lat = rng.uniform(5.1, 6.1, 8), rng.uniform(50.9, 51.9, 8)
    plants = _plants(lon, lat, size=rng.lognormal(9, 1, 8))
    cfg = {
        "year": 2022, "aggregate_cells": 1, "candidate_plants": 8, "max_buffer": 50000.0, "fit_reference": "jmp",
        "district_references": {}, "paths": {"pop_tif_dir": str(tmp_path)},
    }  # fmt: skip
    countries = pd.DataFrame({"iso2": ["NL"], "iso3": ["NLD"], "total_people": [1e6], "sum_size_reported": [4.5e5]})
    references = pd.DataFrame({"iso3": ["NLD"], "jmp_share": [0.9]})  # connected 9e5: k = 2 within 0.1-10
    calibration = pd.DataFrame(
        {"size_measure": "reported", "radius_exponent": 0.3, "scale_min": [1.0, 0.1, 0.1], "scale_max": [1.0, 10.0, 10.0],
         "cap_km": [3.0, 5.0, 40.0], "a": [9.0, 9.0, 9.0], "b": [0.0, 0.0, 0.0],  # exp(9): the cap is the radius
         "mean_abs_error_pp": [10.0, 8.0, 9.0], "sdg631_mean_abs_error_pp": [10.0, 8.0, 7.0]}
    )  # fmt: skip
    cells_lon, cells_lat, people = aggregate_cells(str(tmp_path / "nl.tif"), 1)
    winner, dist = weighted_nearest(
        cKDTree(sphere_xyz(lon, lat)), sphere_xyz(cells_lon, cells_lat), plants["size_reported"].to_numpy(), 8
    )
    truth = pc.served_by_plant(winner, dist, people, plants["size_reported"].to_numpy(), 2.0, 0.3, 5.0, 50000.0)
    keep = np.flatnonzero(truth > 0)
    units = pd.DataFrame({"reference": "nl_units", "iso2": "NL", "unit": keep.astype(str), "ref": truth[keep]})
    matches = pd.DataFrame({"reference": "nl_units", "unit": keep.astype(str), "SITE_UID": keep})
    return cfg, plants, countries, references, calibration, units, matches


def test_the_configuration_that_made_the_references_has_no_plant_error(tmp_path, monkeypatch):
    cfg, plants, countries, references, calibration, units, matches = _world(tmp_path, monkeypatch)
    rows, districts, unit_preds = pc.check(
        cfg, plants, countries, references, calibration, units, matches, {"reported": 0.3}, {"reported": 3.75}
    )
    assert len(rows) == 4 and districts.empty  # fixed radius + three rows of calibrate
    made = rows[rows["cap_km"] == 5.0].iloc[0]
    assert made["k"] == pytest.approx(2.0) and made["r10k_km"] == pytest.approx(5.0)
    assert made["class_balanced_error"] == pytest.approx(0.0) and made["total_ratio"] == pytest.approx(1.0)
    fixed = rows[rows["cap_km"].isna()].iloc[0]
    assert fixed["k"] == 1.0 and fixed["r10k_km"] == 3.75 and fixed["class_balanced_error"] > 0
    assert rows[rows["cap_km"] == 3.0]["total_ratio"].iloc[0] < 1 < rows[rows["cap_km"] == 40.0]["total_ratio"].iloc[0]
    choice, _ = pc.choose(rows, calibration, ["jmp", "sdg631"])
    choice = choice.set_index("left_out")
    # national: cap 3 -> 10, cap 5 -> 8, cap 40 -> 8 (9 and 7); cap 40 serves too much, so cap 5
    assert choice.loc["", "cap_km"] == 5.0 and choice.loc["", "national_error_pp"] == pytest.approx(8.0)
    # the same with the paired bootstrap: the configuration that made the references is never worse
    assert len(unit_preds) == 4 * len(units) and set(unit_preds["conf"]) == {0, 1, 2, 3}
    boot = pc.bootstrap_errors(units, unit_preds, 4, 200, 0)
    assert boot[0].shape == (200, 4, 1) and np.all(boot[0][:, 2, 0] == 0.0)
    choice, table = pc.choose(rows, calibration, ["jmp", "sdg631"], boot, {"nl_units": "NL"}, 0.95)
    assert choice.set_index("left_out").loc["", "cap_km"] == 5.0
    assert table.set_index("cap_km").loc[5.0, "share_worse"] == 0.0 and bool(
        table.set_index("cap_km").loc[5.0, "allowed"]
    )


def test_the_choice_takes_the_best_national_error_the_plants_allow():
    key = {"size_measure": "reported", "radius_exponent": 0.3, "scale_min": 0.1, "scale_max": 10.0}
    err = {  # log10 errors per country: GB, NL
        np.nan: (0.20, 0.20),  # fixed radius
        4.0: (0.18, 0.20),
        6.0: (0.19, 0.20),
        8.0: (0.30, 0.05),  # better on average than fixed, worse in GB
    }
    rows = pd.DataFrame(
        [{**key, "conf": n, "cap_km": cap, "iso2": c, "reference": c.lower(), "class_balanced_error": e[i]}
         for n, (cap, e) in enumerate(err.items()) for i, c in enumerate(("GB", "NL"))]
    )  # fmt: skip
    rows.loc[rows["cap_km"].isna(), ["scale_min", "scale_max"]] = 1.0
    calibration = pd.DataFrame(
        {**key, "cap_km": [4.0, 6.0, 8.0], "a": 1.0, "b": 0.2, "mean_abs_error_pp": [9.0, 8.0, 5.0],
         "sdg631_mean_abs_error_pp": [9.0, 8.0, 5.0]}
    )  # fmt: skip
    choice = pc.choose(rows, calibration, ["jmp", "sdg631"])[0].set_index("left_out")
    assert choice.loc["", "cap_km"] == 8.0  # mean plant error 0.175 <= 0.20, best national
    assert choice.loc["NL", "cap_km"] == 6.0  # judged on GB alone, cap 8 is worse than the fixed radius
    assert choice.loc["GB", "cap_km"] == 8.0
    assert choice.loc["", "baseline_plants_x"] == pytest.approx(10**0.2)
    worse = rows.assign(class_balanced_error=np.where(rows["cap_km"].isna(), 0.0, rows["class_balanced_error"]))
    none = pc.choose(worse, calibration, ["jmp", "sdg631"])[0].set_index("left_out")
    assert none.loc["", "choice"] == "fixed radius" and np.isnan(none.loc["", "cap_km"])


def test_every_polygon_layer_of_a_folder_is_read_and_other_files_are_left_out(tmp_path):
    folder = tmp_path / "fra"
    folder.mkdir()
    gpd.GeoDataFrame({"nom": ["a"]}, geometry=[box(3.0, 45.7, 3.1, 45.8)], crs=4326).to_file(folder / "agglo.geojson")
    shp = tmp_path / "shp"
    shp.mkdir()
    gpd.GeoDataFrame(
        {"nom": ["b", "c"]}, geometry=[box(3.2, 45.7, 3.3, 45.8), box(3.4, 45.7, 3.5, 45.8)], crs=4326
    ).to_crs(2154).to_file(shp / "agglo2.shp")  # another CRS, zipped
    gpd.GeoDataFrame({"nom": ["step"]}, geometry=gpd.points_from_xy([3.05], [45.75]), crs=4326).to_file(
        shp / "step.shp"
    )
    with zipfile.ZipFile(folder / "download.zip", "w") as z:
        for f in os.listdir(shp):
            z.write(shp / f, f)
    (folder / "notes.txt").write_text("not spatial")
    polygons = pc.read_polygons(str(folder))
    assert len(polygons) == 3 and polygons.crs.to_epsg() == 4326  # the plant points are left out
    assert polygons.total_bounds == pytest.approx([3.0, 45.7, 3.5, 45.8], abs=1e-6)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("nothing")
    with pytest.raises(FileNotFoundError, match="no polygon layer"):
        pc.read_polygons(str(empty))
