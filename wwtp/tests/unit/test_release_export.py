"""The release format (wwtp.release_export): two layers with the configured columns in order."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from wwtp import release_export as rx
from wwtp.starter import load_config

pytestmark = pytest.mark.unit


def _layer(cfg):
    """A small population layer with every column the export reads."""
    renames = {new: old for old, new in cfg["renames"].items()}
    columns = [
        renames[c] if c in renames else c for c in cfg["columns"] if c not in {"SOURCE_DS", *cfg["empty_columns"]}
    ]
    n = 5
    data = {c: [None] * n for c in columns}
    data.update(
        WASTE_ID=[10, 11, 12, 13, 14],
        ISO_2=["JP", "JP", "AE", "DE", "AE"],
        HYBAS_ID=[2.0, 2.0, 1.0, 3.0, 1.0],
        OLD_WASTE_ID=["JP_P22_G01", "55668.0", "54120.0", None, "54122.0"],
        uwwCode=[None, None, None, "DETP_1", None],
        origin=["new_plant_national", "new_plant_national", "v2", "new_plant_eu_file", "v2"],
        WKT_WWTP=[f"POINT ({x} 0)" for x in range(n)],
        weights=[0.2] * n,
        **{"2024_zonal_sum": [1.0, 2.0, 3.0, 4.0, 5.0]},
    )
    return gpd.GeoDataFrame(data, geometry=[Point(x, 0).buffer(0.4) for x in range(n)], crs=4326)


def test_two_layers_with_the_columns_in_order_and_the_v3_fixes():
    cfg = load_config("release_export")
    layer = _layer(cfg)
    layer.loc[3, "geometry"] = None  # no service area: left out
    sizes = np.array([100.0, 400.0, 900.0, 1.0, 100.0])
    areas, points = rx.build(layer, cfg, sizes)

    assert list(areas.columns) == [*cfg["columns"], "geometry"]
    assert list(points.columns) == [c for c in cfg["columns"] if c != "WKT_WWTP"] + ["WKT_SERVICE_AREA", "geometry"]
    assert areas["WASTE_ID"].tolist() == [0, 1, 2, 3] and len(points) == 4
    # sorted by country, basin, source: AE (basin 1) first
    assert areas["ISO_2"].tolist() == ["AE", "AE", "JP", "JP"]
    by_id = areas.set_index(areas["WKT_WWTP"])
    # a register ID is not a HydroWASTE ID; NAT for the register plant, HW for the HydroWASTE one
    assert pd.isna(by_id.loc["POINT (0 0)", "HW_WASTE_ID"]) and by_id.loc["POINT (0 0)", "SOURCE_DS"] == "NAT"
    assert by_id.loc["POINT (1 0)", "HW_WASTE_ID"] == "55668.0" and by_id.loc["POINT (1 0)", "SOURCE_DS"] == "HW"
    # weights: sqrt(size) normalised within the basin (AE basin 1: sqrt 900 and sqrt 100)
    assert by_id.loc["POINT (2 0)", "weights"] == pytest.approx(30 / 40)
    assert by_id.loc["POINT (4 0)", "weights"] == pytest.approx(10 / 40)
    assert areas.groupby("HYBAS_ID")["weights"].sum().to_numpy() == pytest.approx([1, 1])
    assert areas["2014_zonal_sum"].isna().all() and areas["2024_zonal_sum"].notna().all()
    # the points layer: the plant point, the polygon as WKT
    assert points.geometry.geom_type.eq("Point").all()
    assert gpd.GeoSeries.from_wkt(points["WKT_SERVICE_AREA"]).geom_type.eq("Polygon").all()
    assert points.geometry.iloc[0].equals(Point(2, 0))


def test_a_missing_column_stops_the_export():
    cfg = load_config("release_export")
    with pytest.raises(KeyError, match="uwwName"):
        rx.build(_layer(cfg).drop(columns=["uwwName"]), cfg, np.ones(5))


def test_both_layers_are_written(tmp_path):
    cfg = load_config("release_export")
    areas, points = rx.build(_layer(cfg), cfg, np.ones(5))
    path = tmp_path / "release.gpkg"
    rx.write(areas, points, cfg, str(path))
    import pyogrio

    assert sorted(name for name, _ in pyogrio.list_layers(path)) == ["WWTP_LOCATIONS", "WWTP_SERVICE_AREAS"]
    assert len(gpd.read_file(path, layer="WWTP_LOCATIONS")) == 5
