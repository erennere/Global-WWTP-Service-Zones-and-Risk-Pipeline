"""wwtp.figures_scripts.validation_figures: tables, maps and plots from a small sweep_validation output."""

import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from wwtp.figures_scripts import validation_figures as vf
from wwtp.starter import load_config

pytestmark = pytest.mark.unit

CAND = "v2_lvl6_bfrep_sz_mult__a1"
COUNTRIES = {  # iso2: (iso3, region, lon0, lat0)
    "US": ("USA", "Europe and Northern America", -100, 35),
    "GB": ("GBR", "Europe and Northern America", -3, 52),
    "NL": ("NLD", "Europe and Northern America", 5, 52),
    "CN": ("CHN", "Eastern and South-Eastern Asia", 100, 30),
    "BR": ("BRA", "Latin America and the Caribbean", -50, -10),
}


def _sweep_folder(root, rng):
    os.makedirs(root / "references")
    refs, units, matches, plants = [], [], [], []
    uid = 0
    for ref, iso2, n in (("gb", "GB", 60), ("nl", "NL", 40), ("us", "US", 80), ("hw", "CN", 50), ("hw", "BR", 30)):
        truth = 10 ** rng.uniform(2.5, 5.5, n)
        for k in range(n):
            unit = f"{iso2}{k}"
            refs.append({"reference": ref, "unit": unit, "ref": truth[k], "year": 2024, "group": "ver"})
            units.append(
                {"candidate": CAND, "reference": ref, "unit": unit, "pred": truth[k] * 10 ** rng.normal(0, 0.15)}
            )
            units.append({"candidate": "other", "reference": ref, "unit": unit, "pred": truth[k] * 3})
            matches.append({"reference": ref, "unit": unit, "SITE_UID": uid})
            plants.append({"SITE_UID": uid, "iso2": iso2, "group": "ver"})
            uid += 1
    units[0]["pred"] = 0.0  # a unit with nobody in its service area
    pd.DataFrame(refs).to_csv(root / "references/units.csv", index=False)
    pd.DataFrame(matches).to_csv(root / "references/matches.csv", index=False)
    pd.DataFrame(plants).to_csv(root / "references/plants.csv", index=False)
    pd.DataFrame(units).to_parquet(root / "units.parquet", index=False)
    pd.DataFrame([{"candidate": CAND, "recommended": True}, {"candidate": "other", "recommended": False}]).to_csv(
        root / "choice.csv", index=False
    )
    rows = []
    for iso2, (iso3, region, *_) in COUNTRIES.items():
        served = rng.uniform(0.2, 0.9)
        for ref, share in (("jmp", served + 0.05), ("oecd", served + 0.1), ("eurostat", served + 0.08)):
            if ref == "eurostat" and iso2 not in ("GB", "NL"):
                continue
            rows.append({"candidate": CAND, "reference": ref, "iso2": iso2, "iso3": iso3, "region_sdg": region,
                         "people": 1e7, "served_share": served, "ref_share": min(share, 1.0), "ref_year": 2022})  # fmt: skip
    nat = pd.DataFrame(rows)
    nat.to_csv(root / "national_countries.csv", index=False)
    reg = (
        nat.groupby(["candidate", "reference", "region_sdg"])
        .agg(people=("people", "sum"), served_share=("served_share", "mean"), ref_share=("ref_share", "mean"))
        .reset_index()
    )
    reg["error_pp"] = (reg["served_share"] - reg["ref_share"]) * 100
    reg.to_csv(root / "national_regions.csv", index=False)


def _boundaries(path):
    shapes = [box(lon, lat, lon + 8, lat + 6) for (_, _, lon, lat) in COUNTRIES.values()]
    shapes.append(box(20, 0, 30, 10))  # a country without data
    gpd.GeoDataFrame(
        {"ISO_A2_EH": [*COUNTRIES, "XX"], "NAME": [*COUNTRIES, "Nowhere"]}, geometry=shapes, crs=4326
    ).to_file(path)


def _layer(path, rng):
    """A population layer with the columns register_units reads: one plant per row."""
    rows = []
    uid = 1000
    for iso2, (_, _, lon, lat) in COUNTRIES.items():
        for k in range(30):
            value = float(10 ** rng.uniform(3, 5))
            rows.append(
                {
                    "SITE_UID": uid,
                    "ISO_2": iso2,
                    "2022_zonal_sum": value * 10 ** rng.normal(0, 0.15),
                    "2024_zonal_sum": value * 10 ** rng.normal(0, 0.15),
                    "POP_SERVED": value if iso2 in ("CN", "BR") else np.nan,
                    "QUAL_POP": (1 if k % 3 else 2) if iso2 in ("CN", "BR") else np.nan,
                    "EPA POP 2022": value if iso2 == "US" else np.nan,
                    "uwwLoadEnteringUWWTP": value if iso2 in ("GB", "NL") else np.nan,
                    "geometry": box(lon, lat, lon + 0.1, lat + 0.1),
                }
            )
            uid += 1
    gpd.GeoDataFrame(rows, crs=4326).to_file(path, driver="GPKG")


def test_register_units_take_reported_numbers_from_the_layer(tmp_path):
    rng = np.random.default_rng(2)
    _layer(tmp_path / "layer.gpkg", rng)
    cfg = load_config(script_name="validation_figures")
    national = pd.DataFrame({"iso2": list(COUNTRIES), "region_sdg": [v[1] for v in COUNTRIES.values()]})
    units = vf.register_units(cfg["register_references"], str(tmp_path / "layer.gpkg"), "ISO_2", national)
    counts = units.groupby(["reference", "iso2"]).size().to_dict()
    assert counts[("hw_reported", "CN")] == 20  # QUAL_POP 2 left out (every third plant)
    assert counts[("cwns_2022", "US")] == 30 and counts[("uwwtd_load", "NL")] == 30
    assert units["region_sdg"].notna().all() and (units["pred"] > 0).all()


def test_tables_maps_and_zip(tmp_path):
    sweep = tmp_path / "sweep"
    _sweep_folder(sweep, np.random.default_rng(0))
    _boundaries(tmp_path / "countries.shp")
    cand, units, national, regions = vf.read_inputs(str(sweep))
    assert cand == CAND and len(units) == 260 and set(units["iso2"]) == set(COUNTRIES)
    assert units["region_sdg"].notna().all()

    cfg = load_config(script_name="validation_figures")
    _layer(tmp_path / "layer.gpkg", np.random.default_rng(3))
    registers = vf.register_units(cfg["register_references"], str(tmp_path / "layer.gpkg"), "ISO_2", national)
    cfg["paths"]["output_dir"] = str(tmp_path / "out")
    cfg["paths"]["boundaries"] = str(tmp_path / "countries.shp")
    vf.render(cfg, cand, pd.concat([units, registers], ignore_index=True), national, regions)

    out = tmp_path / "out"
    for name in ("map_plants_catchments", "map_plants_reported", "plants_countries", "regions_countries",
                 "map_national_jmp", "map_national_oecd", "map_national_eurostat"):  # fmt: skip
        assert os.path.getsize(out / f"{name}.png") > 1000
    plants = pd.read_csv(out / "countries_plants.csv", keep_default_na=False, na_values=[""])
    reported = plants[(plants["map"] == "reported") & (plants["size"] == "all")].set_index("iso2")
    assert reported.loc["US", "references"] == "cwns_2022" and reported.loc["CN", "references"] == "hw_reported"
    plants = plants[plants["map"] == "catchments"]
    us_all = plants[(plants["iso2"] == "US") & (plants["size"] == "all")].iloc[0]
    assert us_all["n"] == 80 and 1.0 < us_all["typical_error_x"] < 1.6
    assert set(plants["size"]) == {"all", "<10k", "10k-100k", ">100k"}
    gb = plants[(plants["iso2"] == "GB") & (plants["size"] == "all")].iloc[0]
    assert gb["references"] == "gb"
    summary = pd.read_csv(out / "summary_national.csv").set_index("reference")
    assert summary.loc["jmp", "n_countries"] == 5 and summary.loc["jmp", "bias_pp"] == pytest.approx(-5, abs=0.01)
    wide = pd.read_csv(out / "countries_national.csv", keep_default_na=False, na_values=[""]).set_index("iso2")
    assert wide.loc["NL", "eurostat_diff_pp"] == pytest.approx(-8, abs=0.01) and np.isnan(
        wide.loc["US", "eurostat_pct"]
    )
    with zipfile.ZipFile(out / "validation_results.zip") as z:
        names = set(z.namelist())
    assert {"map_plants_reported.pdf", "countries_plants.csv", "summary_national.csv"} <= names


def test_typical_error_counts_empty_service_areas_as_wrong():
    df = pd.DataFrame({"pred": [0.0, 100.0, 100.0], "ref": [100.0, 100.0, 200.0]})
    m = vf.plant_measures(df)
    assert m["typical_error_x"] == pytest.approx(2.0)  # median of (inf, 1, 2)
    assert m["within_factor_2"] == pytest.approx(2 / 3)


def test_recommended_candidate_must_be_unique(tmp_path):
    sweep = tmp_path / "sweep"
    _sweep_folder(sweep, np.random.default_rng(1))
    pd.DataFrame([{"candidate": CAND, "recommended": False}]).to_csv(sweep / "choice.csv", index=False)
    with pytest.raises(ValueError, match="recommended"):
        vf.read_inputs(str(sweep))
