"""wwtp.calibration.connection_radius: raster blocks, served-share curves, references and the fit."""

import json
import math
import zipfile

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

from wwtp.calibration import connection_radius as cr

pytestmark = pytest.mark.unit


def _raster(path, values, nodata=-99999.0, origin=(5.0, 52.0), cell=0.01):
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(origin[0], origin[1], cell, cell),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype("float32"), 1)


def test_blocks_sum_people_and_skip_nodata(tmp_path):
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 10, size=(23, 31))
    values[0, :5] = -99999.0  # nodata
    values[5, 5] = -3.0  # negative counts as 0
    values[6, 6] = np.nan
    _raster(tmp_path / "x.tif", values)
    lon, lat, people = cr.aggregate_cells(str(tmp_path / "x.tif"), 4)
    clean = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    padded = np.pad(clean, ((0, (-23) % 4), (0, (-31) % 4)))
    expected = padded.reshape(6, 4, 8, 4).sum(axis=(1, 3))
    assert people.sum() == pytest.approx(clean.sum(), rel=1e-5)
    assert np.sort(people) == pytest.approx(np.sort(expected[expected > 0]), rel=1e-5)
    # the last block column holds 3 source columns: its centre sits 1.5 cells in
    assert lon.max() == pytest.approx(5.0 + (28 + 1.5) * 0.01)
    assert lat.max() == pytest.approx(52.0 - 2 * 0.01)


def test_served_people_equals_the_rule_applied_cell_by_cell():
    rng = np.random.default_rng(1)
    plants_lonlat = rng.uniform([5.0, 51.0], [6.0, 52.0], size=(40, 2))
    cells_lonlat = rng.uniform([5.0, 51.0], [6.0, 52.0], size=(3000, 2))
    people = rng.uniform(0, 100, 3000)
    size = rng.lognormal(9, 1.5, 40)
    cells, sites = cr.sphere_xyz(*cells_lonlat.T), cr.sphere_xyz(*plants_lonlat.T)
    winner, dist = cr.weighted_nearest(cKDTree(sites), cells, size, candidates=40)
    full = np.linalg.norm(cells[:, None, :] - sites[None, :, :], axis=-1)
    expected_winner = np.argmin(full / np.sqrt(size), axis=1)
    assert np.array_equal(winner, expected_winner)
    for r10k in (1.0, 3.75, 10.0):
        radius = np.minimum(r10k * (size / 1e4) ** 0.3 * 1000, 50_000)
        expected = people[full[np.arange(3000), expected_winner] <= radius[expected_winner]].sum()
        assert cr.served_people(people, winner, dist, size, 0.3, r10k, 50_000) == pytest.approx(expected)


def test_chord_distance_matches_ground_distance_at_service_scale():
    a, b = cr.sphere_xyz([5.0, 5.0], [52.0, 52.0 + 50_000 / 111_195])  # 50 km north
    assert np.linalg.norm(a - b) == pytest.approx(50_000, rel=1e-4)


def test_the_raster_of_the_configured_year(tmp_path):
    folder = tmp_path / "nld"
    folder.mkdir()
    for year in (2020, 2022):
        (folder / f"nld_pop_{year}_CN_100m_R2024B_v1.tif").touch()
    assert cr.raster_for_year("NL", str(tmp_path), 2022).endswith("nld_pop_2022_CN_100m_R2024B_v1.tif")
    assert cr.raster_for_year("NL", str(tmp_path), 2019) is None
    assert cr.raster_for_year("DE", str(tmp_path), 2022) is None
    (folder / "nld_pop_2022_copy.tif").touch()
    with pytest.raises(ValueError, match="several rasters"):
        cr.raster_for_year("NL", str(tmp_path), 2022)


def test_curves_for_a_country_with_plants_and_a_raster(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    (tmp_path / "pop" / "nld").mkdir(parents=True)
    _raster(tmp_path / "pop" / "nld" / "nld_pop_2022_CN_100m_R2024B_v1.tif", rng.uniform(0, 50, (60, 80)))
    plants = pd.DataFrame(
        {
            "iso2": ["NL"] * 5 + ["BE"],
            "lon": [5.1, 5.3, 5.5, 5.7, 5.2, 4.0],
            "lat": [51.9, 51.7, 51.6, 51.5, 51.5, 51.0],
            "reported": [5e4, np.nan, 2e4, 1e5, np.nan, 1e4],
            "size_reported": [5e4, 3e3, 2e4, 1e5, 2.6e3, 1e4],
            "size_footprint": [4e4, 3e3, 3e4, 8e4, 2.6e3, 1e4],
        }
    )
    monkeypatch.setattr(cr, "load_plants", lambda cfg: plants)
    cfg = {
        "year": 2022,
        "aggregate_cells": 5,
        "candidate_plants": 64,
        "size_measures": ["reported", "footprint"],
        "radius_exponents": [0.3, 0.5],
        "radius_grid": {"start": 0.5, "stop": 40.0, "count": 12},
        "max_buffer": 50_000,
        "max_workers": 1,
        "paths": {"pop_tif_dir": str(tmp_path / "pop"), "output_dir": str(tmp_path / "out")},
    }
    countries, curves = cr.compute_curves(cfg)
    nl = countries.set_index("iso2").loc["NL"]
    assert nl["status"] == "ok" and nl["n_plants"] == 5 and nl["n_plants_reported"] == 3
    assert nl["reported_coverage"] == pytest.approx(1.7e5 / nl["total_people"])
    assert nl["sum_size_reported"] == pytest.approx(1.756e5) and nl["sum_size_footprint"] == pytest.approx(1.556e5)
    assert countries.set_index("iso2").loc["BE", "status"] == "no raster"
    assert len(curves) == 2 * 2 * 12
    assert pd.read_csv(tmp_path / "out" / "plants.csv")["size_footprint"].tolist() == plants["size_footprint"].tolist()
    for _, curve in curves.groupby(["size_measure", "radius_exponent"]):
        served = curve.sort_values("radius_at_10k_people_km")["served_people"].to_numpy()
        assert np.all(np.diff(served) >= 0) and served[-1] <= nl["total_people"] + 1e-6
    assert (tmp_path / "out" / "countries.csv").exists() and (tmp_path / "out" / "served_share_curves.csv").exists()


def test_needed_radius_inside_and_outside_the_grid():
    radii = np.array([1.0, 2.0, 4.0, 8.0])
    shares = np.array([0.2, 0.4, 0.6, 0.8])
    r, where = cr.needed_radius(radii, shares, 0.5)
    assert where == "ok" and r == pytest.approx(math.sqrt(2.0 * 4.0))  # halfway in log radius
    assert cr.needed_radius(radii, shares, 0.1) == (1.0, "below grid")
    assert cr.needed_radius(radii, shares, 0.9) == (8.0, "above grid")
    assert cr.share_at(radii, shares, r) == pytest.approx(0.5)


def _synthetic_countries(a, b, n=30, seed=3):
    """Countries whose needed r10k is exactly exp(a + b logit(C)), with curves 1 - exp(-r / r0)."""
    rng = np.random.default_rng(seed)
    radii = cr.radius_grid({"start": 0.5, "stop": 40.0, "count": 60})
    countries, curves, refs = [], [], []
    for k in range(n):
        iso2, iso3 = f"Q{chr(65 + k % 26)}{k}"[:2] + str(k), f"Q{k:02d}"
        share = rng.uniform(0.2, 0.9)
        r_needed = math.exp(a + b * cr.logit(share))
        r0 = -r_needed / math.log(1 - share)
        total = 1e6
        countries.append(
            {
                "iso2": iso2,
                "iso3": iso3,
                "status": "ok",
                "total_people": total,
                "n_plants": 10,
                "reported_coverage": share,
            }
        )
        for r in radii:
            curves.append(
                {
                    "iso2": iso2,
                    "size_measure": "reported",
                    "radius_exponent": 0.3,
                    "radius_at_10k_people_km": r,
                    "served_people": total * (1 - math.exp(-r / r0)),
                }
            )
        refs.append({"iso3": iso3, "jmp_share": share, "oecd_share": share if k % 2 else np.nan})
    return pd.DataFrame(countries), pd.DataFrame(curves), pd.DataFrame(refs)


def test_the_fit_recovers_a_known_formula_out_of_sample():
    countries, curves, refs = _synthetic_countries(a=1.5, b=0.4)
    cfg = {"fit_reference": "jmp", "fit_folds": 5, "fit_seed": 0, "fit_predictors": ["reference", "reported_coverage"]}
    summary, per_country = cr.fit_connection_radius(cfg, countries, curves, refs, {"reported": (0.3, 3.75)})
    fitted = summary[summary["model"].str.startswith("r10k = exp")]
    assert fitted["a"].to_numpy() == pytest.approx([1.5, 1.5], abs=0.02)
    assert fitted["b"].to_numpy() == pytest.approx([0.4, 0.4], abs=0.02)
    assert (fitted["mean_abs_error_pp"] < 0.5).all()
    # the other source is scored on its own countries; here it equals the reference
    assert (fitted["check oecd: n"] == 15).all() and (fitted["check oecd: mean_abs_error_pp"] < 0.5).all()
    assert fitted["served / reference people"].to_numpy() == pytest.approx([1.0, 1.0], abs=0.01)
    current = summary[summary["model"].str.startswith("current")]
    assert float(current["mean_abs_error_pp"].iloc[0]) > 5  # one radius for all misses the spread
    assert set(per_country["predictor"]) == {"reference", "reported_coverage"} and len(per_country) == 60


def test_references_are_saved_with_a_manifest_and_failures_fail_the_run(tmp_path, monkeypatch):
    class Response:
        def __init__(self, url):
            self.url, self.content = url, b"col\n1\n"

        def raise_for_status(self):
            if "broken" in self.url:
                raise OSError("403 Forbidden")

    class Session:
        def get(self, url, timeout):
            return Response(url)

    monkeypatch.setattr(cr, "requests_session_with_retries", lambda: Session())
    cfg = {
        "reference_urls": {"a.csv": "https://example.org/a?x=1", "b.csv": "https://broken.example.org/b"},
        "request_timeout_seconds": 5,
        "paths": {"output_dir": str(tmp_path)},
    }
    with pytest.raises(RuntimeError, match=r"\['b.csv'\]"):
        cr.download_references(cfg)
    manifest = json.loads((tmp_path / "references" / "manifest.json").read_text())
    assert manifest["a.csv"]["bytes"] == 6 and manifest["a.csv"]["final_url"] == "https://example.org/a"
    assert "403" in manifest["b.csv"]["error"]
    assert (tmp_path / "references" / "a.csv").read_bytes() == b"col\n1\n"


def test_references_are_read_per_source_in_the_year_nearest_the_target(tmp_path):
    folder = tmp_path / "references"
    folder.mkdir()
    pd.DataFrame(
        {
            "iso3": ["NLD", "NLD", "IND", "ARG"],
            "year": [2021, 2022, 2022, 2010],
            "san_sew_sm_t": [97.0, 97.3, 5.2, 28.0],
            "san_sew_t": [99.0, 99.0, 14.3, 60.0],
            "region_sdg": ["Europe and Northern America"] * 2
            + ["Central and Southern Asia", "Latin America and the Caribbean"],
        }
    ).to_excel(folder / "jmp.xlsx", sheet_name="san", index=False)
    (folder / "eurostat.tsv").write_text(
        "freq,ww_tp,unit,geo\\TIME_PERIOD\t2020 \t2022 \t2023 \n"
        "A,URB_OTH_CS,PC,NL\t99.4 \t99.5 e\t: \n"
        "A,URB_OTH_CS,PC,EL\t93.1 \t: \t94.0 \n"
        "A,URB_OTH_CS,PC,EU27_2020\t90.0 \t91.0 \t: \n"
        "A,URB_OTH_CS,THS_PER,NL\t17000 \t17100 \t: \n"
        "A,URB_CS,PC,NL\t99.9 \t99.9 \t: \n"
    )
    pd.DataFrame(
        {
            "REF_AREA": ["NLD", "NLD", "CHN"],
            "MEASURE": ["PUBTOTTR", "OTHER", "PUBTOTTR"],
            "TIME_PERIOD": [2022, 2022, 2004],
            "OBS_VALUE": [99.6, 1.0, 32.5],
        }
    ).to_csv(folder / "oecd.csv", index=False)
    sdg = {
        "sheet": "Countries",
        "iso3": "Country ID (iso3)",
        "total_volume": "Volume, Total",
        "sewer_volume": "Volume, Sewers",
        "sewer_delivered_percent": "Delivered, Sewers",
        "year": 2022,
    }
    pd.DataFrame(  # India: 40 % of household wastewater in sewers, a quarter of it delivered to treatment
        {"Country ID (iso3)": ["IND", "NLD", "AND"], "Volume, Total": [100.0, 50.0, 0.0],
         "Volume, Sewers": [40.0, 49.0, 0.0], "Delivered, Sewers": [25.0, 100.0, 100.0]}
    ).to_excel(tmp_path / "sdg.xlsx", sheet_name="Countries", index=False)  # fmt: skip
    cfg = {
        "year": 2022,
        "reference_files": {"jmp": "jmp.xlsx", "eurostat": "eurostat.tsv", "oecd": "oecd.csv"},
        "reference_series": {"jmp": "san_sew_sm_t", "eurostat": "URB_OTH_CS", "oecd": "PUBTOTTR"},
        "eurostat_geo_aliases": {"EL": "GR", "UK": "GB"},
        "reference_max_year_gap": 5,
        "sdg631": sdg,
        "paths": {"output_dir": str(tmp_path), "sdg631": str(tmp_path / "sdg.xlsx")},
    }
    refs = cr.read_references(cfg).set_index("iso3")
    refs = refs[refs[["jmp_share", "eurostat_share", "oecd_share"]].notna().any(axis=1)]  # ARG keeps only its region
    assert sorted(refs.index) == ["GRC", "IND", "NLD"]  # ARG 2010 and CHN 2004 are too old; EU27 is no country
    assert refs.loc["IND", "region_sdg"] == "Central and Southern Asia"
    assert refs.loc["NLD", "jmp_share"] == pytest.approx(0.973) and refs.loc["IND", "jmp_share"] == pytest.approx(0.052)
    assert refs.loc["NLD", "eurostat_share"] == pytest.approx(0.995)  # flag dropped, PC and URB_OTH_CS only
    assert refs.loc["GRC", "eurostat_share"] == pytest.approx(0.94) and refs.loc["GRC", "eurostat_year"] == 2023
    assert refs.loc["NLD", "oecd_share"] == pytest.approx(0.996) and np.isnan(refs.loc["IND", "oecd_share"])
    assert refs.loc["IND", "sdg631_share"] == pytest.approx(0.1) and refs.loc["NLD", "sdg631_share"] == pytest.approx(
        0.98
    )
    assert "AND" not in refs.index or np.isnan(refs.loc["AND", "sdg631_share"])  # no volume: no share


def _capped_world(a, b, cap, size_error, exponent=0.3, n=40, seed=5):
    """Countries that need r10k = min(cap, exp(a + b logit C)) once their plant sizes add up to their
    connected people; the sizes at hand are off by the factor ``size_error(k)`` (1 = right)."""
    rng = np.random.default_rng(seed)
    radii = cr.radius_grid({"start": 0.5, "stop": 40.0, "count": 60})
    countries, curves, refs = [], [], []
    for k in range(n):
        iso2, iso3, share = f"Q{k:02d}", f"QQ{k:02d}", rng.uniform(0.05, 0.95)
        error = size_error(k)  # sizes at hand = right sizes * error
        needed = min(cap, math.exp(a + b * cr.logit(share))) / error**exponent  # too-large sizes need a smaller r10k
        r0 = -needed / math.log(1 - share)
        countries.append(
            {"iso2": iso2, "iso3": iso3, "status": "ok", "total_people": 1e6, "sum_size_reported": share * 1e6 * error}
        )
        # a second, not fitted reference: 5 pp above JMP in every country
        refs.append({"iso3": iso3, "jmp_share": share, "sdg631_share": share + 0.05, "region_sdg": f"region {k % 4}"})
        for r in radii:
            curves.append(
                {
                    "iso2": iso2,
                    "size_measure": "reported",
                    "radius_exponent": exponent,
                    "radius_at_10k_people_km": r,
                    "served_people": 1e6 * (1 - math.exp(-r / r0)),
                }
            )
    cfg = {
        "fit_reference": "jmp",
        "check_references": ["sdg631"],
        "calibration_caps_km": [5.0, 40.0],
        "calibration_size_scale_limits": [{"min": 1.0, "max": 1.0}, {"min": 0.1, "max": 10.0}],
        "radius_grid": {"start": 0.5, "stop": 40.0, "count": 60},
    }
    return cfg, pd.DataFrame(countries), pd.DataFrame(curves), pd.DataFrame(refs)


def test_the_capped_rule_is_recovered_and_tested_by_leaving_regions_out():
    """Countries whose needed r10k is min(cap, exp(a + b logit C)): the fit returns a and b, errors near zero."""
    table, detail = cr.calibrate_by_region(*_capped_world(1.4, 0.2, 5.0, size_error=lambda k: 1.0))
    for limits in (1.0, 0.1):  # sizes that already add up to the connected people: the scale changes nothing
        t = table[table["scale_min"] == limits].set_index("cap_km")
        # the line is fitted to min(needed, cap), so a and b are not those of the uncapped rule
        assert t.loc[5.0, "b"] > 0 and t.loc[40.0, "b"] > 0
        assert (
            t.loc[5.0, "mean_abs_error_pp"] < 0.5 * t.loc[0.0, "mean_abs_error_pp"]
        )  # one radius for all misses the spread
        assert t.loc[5.0, "mean_abs_error_pp"] < 3
        # the same served shares against the reference 5 pp higher: about 5 pp too low
        assert t.loc[5.0, "sdg631_bias_pp"] == pytest.approx(t.loc[5.0, "bias_pp"] - 5.0)
        assert t.loc[5.0, "sdg631_n"] == 40
    assert len(detail) == 2 * 2 * 40 and detail["r10k"].max() <= 40.0
    assert (detail["sdg631"] - detail["reference"]).to_numpy() == pytest.approx(0.05)
    assert detail["size_scale"].to_numpy() == pytest.approx(1.0)


def test_the_size_scale_removes_country_errors_of_the_plant_sizes():
    """Sizes 4x too large or too small by country: scaled to the connected people the rule fits again."""
    world = _capped_world(1.4, 0.2, 5.0, size_error=lambda k: (4.0, 0.25, 1.0)[k % 3])
    table, detail = cr.calibrate_by_region(*world)
    t = table.set_index(["scale_min", "cap_km"])
    assert t.loc[(0.1, 5.0), "mean_abs_error_pp"] < 3 < t.loc[(1.0, 5.0), "mean_abs_error_pp"]
    scaled = detail[(detail["scale_min"] == 0.1) & (detail["cap_km"] == 5.0)].set_index("iso2")
    assert scaled.loc[["Q00", "Q01", "Q02"], "size_scale"].to_numpy() == pytest.approx([0.25, 4.0, 1.0])
    assert scaled["needed_r10k"].max() == pytest.approx(5.0, rel=0.01)  # the needed radius of the scaled sizes
    assert detail.loc[detail["scale_min"] == 1.0, "needed_r10k"].max() > 7.0  # 5 km * 4 ** 0.3 without the scale
    assert cr.size_scale([50.0, 5.0, 100.0], [10.0, 100.0, 100.0], (1 / 3, 3)) == pytest.approx([3.0, 1 / 3, 1.0])


def test_country_sizes_and_the_packed_results(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for country, n, offset, unsized in (("AA", 80, 0.5, 9e4), ("BB", 80, -0.3, 1e3), ("CC", 10, 2.0, 5e5)):
        footprint = 10 ** rng.uniform(3, 5, n)
        people = 10 ** (1.5 + offset + 0.7 * np.log10(footprint) + rng.normal(0, 0.05, n))
        sized = np.arange(n) % 2 == 0
        reported = np.where(sized, people, unsized * 10 ** rng.normal(0, 0.05, n))
        rows.append(pd.DataFrame({"iso2": country, "footprint_m2": footprint, "sized": sized, "reported": reported}))
    plants = pd.concat(rows, ignore_index=True)
    plants.loc[plants["iso2"] == "BB", "reported"] = plants["reported"].where(plants["sized"])  # no size-free records
    cfg = {
        "footprint_to_people": {
            "intercept": 1.5,
            "slope": 0.7,
            "country_offsets": {"AA": 0.45},
            "unsized_people_by_country": {"BB": 1234.0},
        },
        "size_country_min_plants": 30,
    }
    t = cr.country_sizes(cfg, plants).set_index("country")
    assert t.loc["AA", "offset"] == pytest.approx(0.5, abs=0.03)
    assert t.loc["BB", "offset"] == pytest.approx(-0.3, abs=0.03)
    assert np.isnan(t.loc["CC", "offset"]) and t.loc["CC", "unsized_from"] == ""  # 10 plants: too few for a value
    assert t.loc["AA", "unsized_people"] == pytest.approx(9e4, rel=0.05) and t.loc["AA", "unsized_from"] == "own"
    # BB has no size-free plant with a record: the ratio of AA (its only source) times BB's median sized plant
    ratio = t.loc["AA", "unsized_median"] / t.loc["AA", "sized_median"]
    assert t["unsized_ratio"].iloc[0] == pytest.approx(ratio)
    assert t.loc["BB", "unsized_people"] == pytest.approx(ratio * t.loc["BB", "sized_median"])
    assert t.loc["BB", "unsized_from"] == "ratio"
    assert t.loc["AA", "config_offset"] == 0.45 and t.loc["BB", "config_unsized_people"] == 1234.0

    out = tmp_path / "out"
    out.mkdir()
    for name in cr.RESULT_FILES:
        (out / name).write_text(name)
    with zipfile.ZipFile(tmp_path / "national.zip", "w") as z:
        z.writestr("new_plants.csv", "register\n")
    cr.pack_results(str(out), str(tmp_path / "national.zip"))
    with zipfile.ZipFile(out / "calibration_results.zip") as z:
        assert set(z.namelist()) == {*cr.RESULT_FILES, "national_data/new_plants.csv"}
        assert z.read("national_data/new_plants.csv") == b"register\n"
