"""wwtp.figures_scripts.calibration_figures: the three figures and the numbers file from small calibration tables."""

import os
import re

import numpy as np
import pandas as pd
import pytest

from wwtp.figures_scripts import calibration_figures as cf
from wwtp.starter import load_config

pytestmark = pytest.mark.unit

COUNTRIES = {  # iso2: (iso3, SDG region, JMP share, people)
    "IN": ("IND", "Central and Southern Asia", 0.05, 1.4e9),
    "BR": ("BRA", "Latin America and the Caribbean", 0.45, 2.1e8),
    "CN": ("CHN", "Eastern and South-Eastern Asia", 0.65, 1.4e9),
    "US": ("USA", "Europe and Northern America", 0.75, 3.3e8),
    "DE": ("DEU", "Europe and Northern America", 0.95, 8.3e7),
    "GB": ("GBR", "Europe and Northern America", 0.97, 6.7e7),
    "NL": ("NLD", "Europe and Northern America", 0.99, 1.7e7),
    "IE": ("IRL", "Europe and Northern America", 0.70, 5.0e6),
}
KEY = ["size_measure", "radius_exponent", "scale_min", "scale_max", "cap_km"]
REFERENCES = {"us_sewersheds": "US", "gb_catchments": "GB", "nl_cbs": "NL", "ie_agglomerations": "IE"}


def _tables(rng):
    radii = np.geomspace(0.5, 40, 12)
    countries, curves, refs, by_country = [], [], [], []
    for iso2, (iso3, region, jmp, people) in COUNTRIES.items():
        countries.append({"iso2": iso2, "iso3": iso3, "status": "ok", "total_people": people})
        mid = 1.0 + 3 * jmp  # the curve reaches half its top at this r10k
        shares = 0.98 / (1 + (mid / radii) ** 2)
        curves += [{"iso2": iso2, "size_measure": "reported", "radius_exponent": 0.3,
                    "radius_at_10k_people_km": r, "served_people": s * people} for r, s in zip(radii, shares)]  # fmt: skip
        refs.append({"iso3": iso3, "region_sdg": region, "jmp_share": jmp, "sdg631_share": jmp * 0.9,
                     "eurostat_share": jmp if region.startswith("Europe") else np.nan, "oecd_share": np.nan})  # fmt: skip
    confs = pd.DataFrame(
        [("reported", 0.3, 1.0, 1.0, np.nan, np.nan, np.nan)]
        + [("reported", 0.3, lo, hi, cap, 1.2, 0.15) for lo, hi in ((1.0, 1.0), (0.1, 10.0)) for cap in (5.0, 40.0)],
        columns=[*KEY, "a", "b"],
    )
    confs.insert(0, "conf", np.arange(len(confs)))
    by_cap = confs[confs["cap_km"].notna()].assign(
        mean_abs_error_pp=[12.0, 10.0, 9.0, 8.0], people_weighted_abs_error_pp=[9.0, 8.0, 7.0, 6.0],
        bias_pp=-2.0, sdg631_mean_abs_error_pp=[15.0, 14.0, 13.0, 12.0], sdg631_bias_pp=-5.0,
        eurostat_mean_abs_error_pp=10.0, eurostat_bias_pp=-4.0, oecd_mean_abs_error_pp=np.nan, oecd_bias_pp=np.nan,
    )  # fmt: skip
    for r in by_cap.itertuples():
        for iso2, (_, _, jmp, _) in COUNTRIES.items():
            by_country.append({**{k: getattr(r, k) for k in KEY}, "iso2": iso2, "reference": jmp,
                               "size_scale_unlimited": 10 ** rng.normal(0, 0.5), "needed_r10k": 1.0 + 3 * jmp,
                               "served_share_region_left_out": jmp + rng.normal(0, 0.05)})  # fmt: skip
    plants = pd.DataFrame(
        {"iso2": rng.choice(list(COUNTRIES), 300), "footprint_m2": 10 ** rng.uniform(2, 5, 300), "sized": True}
    )
    plants["reported"] = 10 ** (1.523 + 0.7 * np.log10(plants["footprint_m2"]) + rng.normal(0, 0.3, 300))
    units, preds, check = [], [], []
    for ref, iso2 in REFERENCES.items():
        truth = 10 ** rng.uniform(2.5, 5.5, 40)
        units.append(pd.DataFrame({"reference": ref, "iso2": iso2, "unit": [f"u{i}" for i in range(40)], "ref": truth}))
        for c in confs.itertuples():
            pred = truth * 10 ** rng.normal(0, 0.15, 40)
            preds.append(
                pd.DataFrame({"conf": c.conf, "reference": ref, "unit": [f"u{i}" for i in range(40)], "pred": pred})
            )
            check.append({"conf": c.conf, **{k: getattr(c, k) for k in KEY}, "reference": ref, "iso2": iso2,
                          "class_balanced_error": float(np.median(np.abs(np.log10(pred / truth)))),
                          "total_ratio": float(pred.sum() / truth.sum())})  # fmt: skip
    configurations = confs[confs["cap_km"].notna()].assign(
        plants_x=[1.40, 1.45, 1.38, 1.50],
        national_error_pp=[13.0, 12.0, 11.0, 10.0],
        allowed=[True, False, True, False],
        share_worse=[0.4, 0.99, 0.4, 0.99],
    )
    choice = pd.DataFrame(
        [
            {
                "size_measure": "reported",
                "left_out": "",
                "choice": "rule",
                "scale_min": 0.1,
                "scale_max": 10.0,
                "cap_km": 5.0,
                "a": 1.2,
                "b": 0.15,
                "plants_x": 1.38,
                "share_worse": 0.4,
                "national_error_pp": 11.0,
            },
            {
                "size_measure": "reported",
                "left_out": "GB",
                "choice": "rule",
                "scale_min": 0.1,
                "scale_max": 10.0,
                "cap_km": 40.0,
                "a": 1.2,
                "b": 0.15,
                "plants_x": 1.4,
                "share_worse": 0.5,
                "national_error_pp": 10.0,
            },
        ]  # fmt: skip
    )
    return {
        "countries": pd.DataFrame(countries),
        "curves": pd.DataFrame(curves),
        "plants": plants,
        "references": pd.DataFrame(refs),
        "sizes": pd.DataFrame(
            {
                "country": ["IN", "US"],
                "n_fit": [40, 120],
                "offset": [0.8, -0.2],
                "unsized_from": ["own", "ratio"],
                "unsized_ratio": 0.409,
            }
        ),  # fmt: skip
        "by_cap": by_cap,
        "by_country": pd.DataFrame(by_country),
        "plant_check": pd.DataFrame(check),
        "plant_status": pd.DataFrame(
            [{"reference": r, "iso2": i, "status": "ok", "units": 40, "people": 4e6} for r, i in REFERENCES.items()]
        ),  # fmt: skip
        "districts": pd.DataFrame(
            [
                {
                    **{k: v for k, v in zip(KEY, ("reported", 0.3, 0.1, 10.0, 5.0))},
                    "reference": "mx_iter_2020",
                    "n_districts": 2469,
                    "served_share": 0.46,
                    "drainage_share": 0.96,
                }
            ]
        ),  # fmt: skip
        "plant_units": pd.concat(preds, ignore_index=True),
        "plant_references": pd.concat(units, ignore_index=True),
        "choice": choice,
        "configurations": configurations,
    }


def test_render_writes_figures_and_numbers(tmp_path):
    cfg = load_config(script_name="calibration_figures")
    cfg["paths"]["output_dir"] = str(tmp_path)
    cf.render(cfg, load_config(script_name="connection_radius"), load_config(script_name="create_voronoi"),
              _tables(np.random.default_rng(0)))  # fmt: skip
    for name in ("ed_plant_size", "ed_service_radius", "ed_catchments"):
        for ext in ("pdf", "png"):
            assert os.path.getsize(tmp_path / f"{name}.{ext}") > 1000
    numbers = dict(
        re.findall(r"\\newcommand\{\\cal(\w+)\}\{([^}]*)\}", (tmp_path / "calibration_numbers.tex").read_text())
    )
    assert numbers["Cap"] == "5" and numbers["ScaleMin"] == "0.1" and numbers["ScaleMax"] == "10"
    assert numbers["Countries"] == "8"
    assert numbers["ErrRuleJmp"] == "9.0" and numbers["ErrRuleSdg"] == "13.0"
    assert numbers["AllowedConfigurations"] == "2" and numbers["Configurations"] == "4"
    assert numbers["UnitsUsSewersheds"] == "40" and numbers["DrainageMxIter"] == "96"
    assert numbers["EurostatCountries"] == "5" and numbers["OecdCountries"] == "0"
    assert numbers["ShareWorse"] == "40.0" and numbers["LeaveOneOutCapHigh"] == "40"
    assert numbers["LeaveOneOutSameLimits"] == "1" and "TypicalIeAgglomerationsChosen" in numbers
    assert all(re.fullmatch(r"[A-Za-z]+", k) for k in numbers)  # usable LaTeX macro names


def test_fixed_radius_choice_stops():
    tables = _tables(np.random.default_rng(0))
    tables["choice"].loc[0, "choice"] = "fixed radius"
    with pytest.raises(ValueError, match="fixed radius"):
        cf.chosen(tables, "reported")


def test_fixed_shares_follow_the_curves():
    tables = _tables(np.random.default_rng(0))
    curves = cf.country_curves(tables, "reported", 0.3, ["sdg631"])
    shares = cf.national_shares(curves, tables["choice"].iloc[0], 0.3, 3.75, tables["by_country"])
    de = shares.set_index("iso2").loc["DE"]
    # the curve of DE is 0.98 / (1 + (3.85 / r)^2): at r = 3.75 about 0.47 (interpolated in log r)
    assert de["fixed"] == pytest.approx(0.98 / (1 + (3.85 / 3.75) ** 2), abs=0.02)
    assert de["sdg631"] == pytest.approx(0.95 * 0.9)
