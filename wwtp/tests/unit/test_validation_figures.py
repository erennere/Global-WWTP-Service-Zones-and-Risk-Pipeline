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
                    "uwwLoadEnteringUWWTP": value * 200 if iso2 in ("GB", "NL") else np.nan,  # p.e. > connected
                    "geometry": box(lon, lat, lon + 0.1, lat + 0.1),
                }
            )
            uid += 1
    gpd.GeoDataFrame(rows, crs=4326).to_file(path, driver="GPKG")


def test_register_units_take_reported_numbers_from_the_layer(tmp_path):
    rng = np.random.default_rng(2)
    _layer(tmp_path / "layer.gpkg", rng)
    cfg = load_config(script_name="validation_figures")
    national = pd.DataFrame({"reference": "eurostat", "iso2": list(COUNTRIES), "people": 1e7, "ref_share": 0.1,
                             "region_sdg": [v[1] for v in COUNTRIES.values()]})  # fmt: skip
    units = vf.register_units(cfg["register_references"], cfg["pe_conversion"], str(tmp_path / "layer.gpkg"),
                              "ISO_2", national)  # fmt: skip
    counts = units.groupby(["reference", "iso2"]).size().to_dict()
    assert counts[("hw_reported", "CN")] == 20  # QUAL_POP 2 left out (every third plant)
    assert counts[("cwns_2022", "US")] == 30 and counts[("uwwtd_load", "NL")] == 30
    assert units["region_sdg"].notna().all() and (units["pred"] > 0).all()
    loads = units[units["reference"] == "uwwtd_load"]  # p.e. (UWWTD) in residents, people registers as reported
    assert np.allclose(loads["ref"] * loads["pe_per_resident"], loads["ref_reported"])
    people = units[units["reference"] != "uwwtd_load"]
    assert people["pe_per_resident"].isna().all() and (people["ref"] == people["ref_reported"]).all()


def test_loads_in_pe_become_residents():
    registers = {
        "load": {"column": "x", "only": {}, "year": 2024, "unit": "pe"},
        "count": {"column": "y", "only": {}, "year": 2024, "unit": "people"},
    }
    units = pd.DataFrame(
        {
            "reference": ["load", "load", "load", "load", "load", "count"],
            "unit": ["a1", "a2", "i1", "p1", "x1", "a3"],
            "ref": [100.0, 200.0, 50.0, 60.0, 10.0, 100.0],
            "pred": 1.0,
            "iso2": ["AT", "AT", "IE", "PT", "XX", "AT"],
        }
    )
    national = pd.DataFrame(
        {
            "reference": ["eurostat", "jmp", "eurostat", "jmp", "jmp"],
            "iso2": ["AT", "AT", "IE", "PT", "XX"],
            "people": [1000.0, 1000.0, 1000.0, 100.0, 100.0],
            "ref_share": [0.2, 0.5, 0.1, 0.5, np.nan],
        }
    )
    conversion = {"national_references": ["eurostat", "oecd", "jmp"], "min_pe_per_resident": 0.74}
    out = vf.pe_to_residents(units, registers, conversion, national).set_index("unit")
    # AT: 300 p.e. / 200 connected (Eurostat before JMP) = 1.5; PT: 60 / 50 (JMP) = 1.2
    assert (
        out.loc["a1", "pe_per_resident"] == pytest.approx(1.5) and out.loc["a1", "pe_per_resident_from"] == "eurostat"
    )
    assert out.loc["p1", "pe_per_resident"] == pytest.approx(1.2) and out.loc["p1", "pe_per_resident_from"] == "jmp"
    # IE (50 / 100 = 0.5: loads missing) and XX (no national number) take AT and PT together: 360 / 250
    for unit in ("i1", "x1"):
        assert out.loc[unit, "pe_per_resident"] == pytest.approx(1.44)
        assert out.loc[unit, "pe_per_resident_from"] == "other countries"
    assert out.loc["i1", "pe_per_resident_national"] == pytest.approx(0.5)
    assert out.loc["a2", "ref"] == pytest.approx(200 / 1.5) and out.loc["a2", "ref_reported"] == 200
    assert out.loc["a3", "ref"] == 100 and np.isnan(out.loc["a3", "pe_per_resident"])
    with pytest.raises(ValueError, match="no country"):
        vf.pe_to_residents(units, registers, {**conversion, "min_pe_per_resident": 5.0}, national)


def test_tables_maps_and_zip(tmp_path):
    sweep = tmp_path / "sweep"
    _sweep_folder(sweep, np.random.default_rng(0))
    _boundaries(tmp_path / "countries.shp")
    cand, units, national, regions = vf.read_inputs(str(sweep))
    assert cand == CAND and len(units) == 260 and set(units["iso2"]) == set(COUNTRIES)
    assert units["region_sdg"].notna().all()

    cfg = load_config(script_name="validation_figures")
    _layer(tmp_path / "layer.gpkg", np.random.default_rng(3))
    registers = vf.register_units(cfg["register_references"], cfg["pe_conversion"], str(tmp_path / "layer.gpkg"),
                                  "ISO_2", national)  # fmt: skip
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
    assert {"map_plants_reported.pdf", "countries_plants.csv", "summary_national.csv", "pe_per_resident.csv"} <= names
    pe = pd.read_csv(out / "pe_per_resident.csv", keep_default_na=False, na_values=[""]).set_index("iso2")
    assert set(pe.index) == {"GB", "NL"} and (pe["pe_per_resident_from"] == "eurostat").all()


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
