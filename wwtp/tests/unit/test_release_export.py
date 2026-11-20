"""The release format (wwtp.release_export): two layers with the configured columns in order."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from wwtp import release_export as rx
from wwtp.starter import load_config

pytestmark = pytest.mark.unit

FACILITY = "Identification de l'installation/ Facility Identification"


def _layer(cfg):
    """A population layer with every column the export reads: one plant per source dataset."""
    renames = {new: old for old, new in cfg["renames"].items()}
    derived = {"SOURCE_DS", "SOURCE_ID", "HW_WASTE_ID", *cfg["empty_columns"]}
    columns = [renames[c] if c in renames else c for c in cfg["columns"] if c not in derived]
    columns += list(cfg["hw_id_columns"])
    columns += [c for rule in cfg["sources"] for c in [*rule["present"], *rule["id_columns"], *rule["where"]]]
    rows = [  # (ISO_2, basin, column values)
        ("JP", 2.0, {"OLD_WASTE_ID": "JP_P22_G01"}),  # national register plant
        ("JP", 2.0, {"OLD_WASTE_ID": "55668.0"}),  # a national plant from HydroWASTE
        ("AE", 1.0, {"OLD_WASTE_ID": "54120.0", "File Name": "1.png"}),  # HydroWASTE (its image does not count)
        ("DE", 3.0, {"uwwCode": "DETP_1", "OLD_WASTE_ID": "1.0"}),  # UWWTD over HydroWASTE
        ("AE", 1.0, {"File Name": "5548.png"}),  # a detection of the location model
        ("US", 4.0, {"CWNS_2022_ID": "4002701005.0", "HW_unique_ID": "25391.0"}),  # US file: CWNS ID, HW ID
        ("TH", 5.0, {"pond": "1.0"}),  # Thailand file: no ID
        ("NZ", 6.0, {"OLD_WASTE_ID": "NZ_OSM_way-997826589"}),
        ("CA", 7.0, {FACILITY: "24705", "uwwCode": "<NA>"}),  # '<NA>' text is no UWWTD code
    ]
    data = {c: [None] * len(rows) for c in dict.fromkeys(columns)}
    for i, (iso2, basin, values) in enumerate(rows):
        data["ISO_2"][i], data["HYBAS_ID"][i] = iso2, basin
        for column, value in values.items():
            data[column][i] = value
    data["WKT_WWTP"] = [f"POINT ({x} 0)" for x in range(len(rows))]
    data["weights"] = [0.2] * len(rows)
    data["2024_zonal_sum"] = np.arange(len(rows), dtype=float)
    return gpd.GeoDataFrame(data, geometry=[Point(x, 0).buffer(0.4) for x in range(len(rows))], crs=4326)


def _by_point(areas):
    return areas.set_index(areas["WKT_WWTP"])


def test_two_layers_with_the_columns_in_order():
    cfg = load_config("release_export")
    layer = _layer(cfg)
    layer.loc[3, "geometry"] = None  # no service area: left out
    areas, points = rx.build(layer, cfg, np.full(len(layer), 100.0))

    assert list(areas.columns) == [*cfg["columns"], "geometry"]
    assert list(points.columns) == [c for c in cfg["columns"] if c != "WKT_WWTP"] + ["WKT_SERVICE_AREA", "geometry"]
    assert areas["WASTE_ID"].tolist() == list(range(len(layer) - 1)) and len(points) == len(layer) - 1
    assert areas["ISO_2"].tolist()[:2] == ["AE", "AE"]  # sorted by country, basin, source
    assert areas["2014_zonal_sum"].isna().all() and areas["2024_zonal_sum"].notna().all()
    assert points.geometry.geom_type.eq("Point").all()
    assert gpd.GeoSeries.from_wkt(points["WKT_SERVICE_AREA"]).geom_type.eq("Polygon").all()
    assert points.geometry.iloc[0].equals(gpd.GeoSeries.from_wkt(areas["WKT_WWTP"]).iloc[0])  # the same plant


def test_every_plant_is_traced_to_its_dataset_and_its_id_there():
    cfg = load_config("release_export")
    by = _by_point(rx.build(_layer(cfg), cfg, np.ones(9))[0])

    def source(x):
        row = by.loc[f"POINT ({x} 0)"]
        return row["SOURCE_DS"], row["SOURCE_ID"], row["HW_WASTE_ID"]

    assert source(0)[:2] == ("JP_P22", "G01") and pd.isna(source(0)[2])  # the register's ID, no HydroWASTE ID
    assert source(1) == ("HW", "55668", "55668.0")
    assert source(2) == ("HW", "54120", "54120.0")
    assert source(3) == ("UWW", "DETP_1", "1.0")
    assert source(4)[:2] == ("DETECTED", "5548.png")
    assert source(5) == ("US_CWNS", "04002701005", "25391.0")  # CWNS ID of 11 digits; the HW ID from the US file
    assert source(6)[0] == "TH_GEOAI" and pd.isna(source(6)[1])
    assert source(7)[:2] == ("OSM", "way-997826589")
    assert source(8)[:2] == ("CA_WSER", "24705") and pd.isna(by.loc["POINT (8 0)", "uwwCode"])


def test_weights_are_the_size_weights_normalised_within_the_basin():
    cfg = load_config("release_export")
    sizes = np.array([100.0, 400.0, 900.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0])
    by = _by_point(rx.build(_layer(cfg), cfg, sizes)[0])
    assert by.loc["POINT (2 0)", "weights"] == pytest.approx(30 / 40)  # AE basin 1: sqrt 900 and sqrt 100
    assert by.loc["POINT (4 0)", "weights"] == pytest.approx(10 / 40)
    assert by.loc[["POINT (0 0)", "POINT (1 0)"], "weights"].tolist() == pytest.approx([10 / 30, 20 / 30])


def test_a_missing_column_stops_the_export():
    cfg = load_config("release_export")
    with pytest.raises(KeyError, match="uwwName"):
        rx.build(_layer(cfg).drop(columns=["uwwName"]), cfg, np.ones(9))


def test_both_layers_are_written(tmp_path):
    import pyogrio

    cfg = load_config("release_export")
    areas, points = rx.build(_layer(cfg), cfg, np.ones(9))
    path = tmp_path / "release.gpkg"
    rx.write(areas, points, cfg, str(path))
    assert sorted(name for name, _ in pyogrio.list_layers(path)) == ["WWTP_LOCATIONS", "WWTP_SERVICE_AREAS"]
    assert len(gpd.read_file(path, layer="WWTP_LOCATIONS")) == 9
