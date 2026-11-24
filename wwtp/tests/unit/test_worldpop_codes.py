"""Codes WorldPop files under another ISO-3: Kosovo (XKX), Gaza Strip and West Bank (PSE), Golan Heights (ISR)."""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from wwtp import add_pop, download_pop
from wwtp.pop_at_risk_river_calculations import create_rasters as cr

pytestmark = pytest.mark.unit
X0, Y0, RES = 35.0, 32.0, 0.001


def _raster(path, value=2.0, n=100):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs="EPSG:4326",
        transform=from_origin(X0, Y0, RES, RES), nodata=-99999,
    ) as dst:
        dst.write(np.full((n, n), value, dtype="float32"), 1)


def test_kosovo_and_the_palestinian_codes_find_their_rasters(tmp_path):
    for iso3 in ("xkx", "pse", "isr", "deu"):
        (tmp_path / iso3).mkdir()
        (tmp_path / iso3 / f"{iso3}_pop_2026_CN_100m_R2024B_v1.tif").write_bytes(b"")
    found = add_pop.find_country_tif_files(["XK", "XG", "XW", "XH", "DE"], str(tmp_path))
    assert {c: os.path.basename(os.path.dirname(v[0])) for c, v in found.items()} == {
        "XK": "xkx", "XG": "pse", "XW": "pse", "XH": "isr", "DE": "deu"
    }


def test_download_pop_fetches_kosovo():
    urls = download_pop.get_urls(2026, 2026)
    assert urls["xkx"] == [
        "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/2026/XKX/v1/100m/constrained/"
        "xkx_pop_2026_CN_100m_R2024B_v1.tif"
    ]
    assert "pse" in urls and "isr" in urls


def test_rows_without_population_are_filled_and_the_others_kept(tmp_path):
    tifs = tmp_path / "tifs"
    for iso3 in ("xkx", "pse"):
        for year in (2024, 2026):
            _raster(str(tifs / iso3 / f"{iso3}_pop_{year}_CN_100m_R2024B_v1.tif"), value=float(year - 2020))
    layer = tmp_path / "pop.gpkg"
    gpd.GeoDataFrame(
        {
            "WASTE_ID": [0, 1, 2],
            "ISO_2": ["DE", "XK", "XG"],
            "2024_zonal_sum": [123.0, np.nan, np.nan],
            "2024_zonal_std": [1.0, np.nan, np.nan],
            "2026_zonal_sum": [456.0, np.nan, np.nan],
            "2026_zonal_std": [2.0, np.nan, np.nan],
        },
        geometry=[box(8, 50, 8.1, 50.1), box(X0, Y0 - 0.01, X0 + 0.01, Y0), box(X0, Y0 - 0.02, X0 + 0.02, Y0)],
        crs=4326,
    ).to_file(layer)
    before = gpd.read_file(layer).dtypes
    assert add_pop.fill_missing_population(str(layer), str(tifs), max_workers=1, years={2024, 2026}) == 2
    out = gpd.read_file(layer).set_index("WASTE_ID")
    assert gpd.read_file(layer).dtypes.equals(before)  # values updated in place: same columns, same types
    assert len(gpd.read_file(layer, bbox=(X0, Y0 - 0.02, X0 + 0.02, Y0))) == 2  # spatial index intact
    assert out.loc[0, "2024_zonal_sum"] == 123.0 and out.loc[0, "2026_zonal_sum"] == 456.0  # untouched
    assert out.loc[1, "2024_zonal_sum"] == pytest.approx(100 * 4.0)  # 10 x 10 pixels of 4 people (2024)
    assert out.loc[2, "2026_zonal_sum"] == pytest.approx(400 * 6.0)  # 20 x 20 pixels of 6 people (2026)
    mtime = os.path.getmtime(layer)
    assert add_pop.fill_missing_population(str(layer), str(tifs), max_workers=1, years={2024, 2026}) == 0
    assert os.path.getmtime(layer) == mtime  # nothing missing: not rewritten


def test_codes_sharing_a_raster_are_one_country_in_create_rasters(tmp_path, monkeypatch):
    tifs = tmp_path / "tifs"
    _raster(str(tifs / "pse" / "pse_pop_2026_CN_100m_R2024B_v1.tif"), value=2.0)
    served = gpd.GeoDataFrame(
        {"ISO_2": ["XG", "XW"]},
        geometry=[box(X0, Y0 - 0.01, X0 + 0.01, Y0), box(X0 + 0.05, Y0 - 0.05, X0 + 0.07, Y0 - 0.03)],
        crs=4326,
    )
    layer, ws = tmp_path / "areas.gpkg", tmp_path / "basins.gpkg"
    served.to_file(layer)
    gpd.GeoDataFrame({"HYBAS_ID": [1]}, geometry=[box(X0 - 1, Y0 - 1, X0 + 1, Y0 + 1)], crs=4326).to_file(ws)
    cfg = {
        "population_year": 2026, "threshold_value": 0, "min_pixels": 1, "served_all_touched": False,
        "annotations": {"max_workers": 1}, "figures": {"approach": "1"}, "country_output_column": "ISO_2",
        "basin_column_name": "HYBAS_ID",
        "paths": {"pop_tif_dir": str(tifs), "watershed": str(ws), "non_served_outpath": str(tmp_path / "ns.gpkg"),
                  "csv_output_filepath": str(tmp_path / "st.csv")},
    }
    monkeypatch.setattr(cr, "load_config", lambda **kw: cfg)
    monkeypatch.setattr(cr.provenance, "write_run_manifest", lambda *a, **k: None)
    monkeypatch.setattr(cr, "create_pop_output_paths", lambda c: {"voronoi": {"1": str(layer)}})
    cr.main(["--out-dir", str(tmp_path / "out")])
    st = pd.read_csv(tmp_path / "out" / "st.csv", keep_default_na=False)
    assert st.country.tolist() == ["PS"]  # one pass over the PSE raster, not one per code
    assert st.population_served[0] == pytest.approx((100 + 400) * 2.0)  # both codes' areas are served
    assert st.population_total[0] == pytest.approx(100 * 100 * 2.0)
