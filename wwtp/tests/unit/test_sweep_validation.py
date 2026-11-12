"""wwtp.sensitivity_analysis_scripts.sweep_validation: references, scores and the choice on a small world."""

import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from wwtp import workflow
from wwtp.sensitivity_analysis_scripts import sweep_validation as sv

pytestmark = pytest.mark.unit

# plant: (iso2, lon, lat, basin, total_area, uwwCode, CWNS, POP_SERVED, QUAL_POP)
PLANTS = [
    ("GB", -1.20, 52.20, 1, 900.0, None, None, 5000.0, 2.0),  # linked catchment c0 (UWWTD plant 100 m away)
    ("GB", -1.00, 52.00, 1, 800.0, None, None, 7000.0, 1.0),  # basin 1: both detected -> ver
    ("GB", -0.80, 51.80, 2, 0.0, None, None, None, None),  # basin 2: half detected -> unver
    ("GB", -0.70, 51.70, 2, 500.0, None, None, None, None),
    ("NL", 5.00, 52.00, 3, 300.0, "NL00001", None, 20000.0, 1.0),  # basin 3: single
    ("NL", 5.30, 52.10, 4, 300.0, "NL00002", None, None, None),
    ("NL", 5.60, 52.20, 4, 300.0, "NL00003", None, None, None),
    ("US", -72.0, 42.0, 5, 300.0, None, "1000127002", None, None),
    ("US", -72.3, 42.2, 5, 300.0, None, "1000127005", None, None),
    ("US", -72.6, 42.4, 6, 300.0, None, "1000139001", None, None),
]
HALF = 0.02  # half width of a GB catchment (degrees)


def _square(lon, lat, half):
    return box(lon - half, lat - half, lon + half, lat + half)


def _world(tmp_path):
    """Reference files, a GB raster (1 person per 0.001 degree cell) and the config."""
    v = tmp_path / "validation_data"
    # GB catchments, Waterbase lookup and coordinates
    (v / "gbr_catchments").mkdir(parents=True)
    gb = [p for p in PLANTS if p[0] == "GB"]
    catch = gpd.GeoDataFrame(
        {"identifier": [f"c{i}" for i in range(len(gb))]},
        geometry=[_square(p[1], p[2], HALF) for p in gb],
        crs=4326,
    )
    shp = tmp_path / "shp"
    shp.mkdir()
    catch.to_file(shp / "catchments_consolidated.shp")
    with zipfile.ZipFile(v / "gbr_catchments" / "catchments_consolidated.zip", "w") as z:
        for f in os.listdir(shp):
            z.write(shp / f, f)
    pd.DataFrame({"identifier": ["c0"], "uwwCode": ["UKENG0"]}).to_csv(
        v / "gbr_catchments" / "waterbase_catchment_lookup.csv", index=False
    )
    pd.DataFrame(
        {
            "uwwCode": ["UKENG0", "UKENG0"],
            "year": [2016, 2018],
            "uwwLatitude": [52.2, 52.2009],
            "uwwLongitude": [-1.2, -1.2],
        }
    ).to_csv(v / "gbr_catchments" / "waterbase_consolidated.csv", index=False)
    # CBS: three junk rows, then the table; one RWZI split over two municipalities, plus the 99999 row
    (v / "nld_cbs_rwzi").mkdir()
    cbs = pd.DataFrame(
        {
            "regio_type": ["GM", "GM", "GM", "GM", "GM", "LD"],
            "rwzi_code": [1, 1, 2, 3, 99999, 1],
            "rwzi_naam": ["a", "a", "b", "c", "none", "a"],
            "inwoners": [30000, 30000, 8000, 150000, 50, 30000],
        }
    )
    cbs.to_excel(v / "nld_cbs_rwzi" / "cbs.xlsx", sheet_name="Tabel 1", startrow=3, index=False)
    # EPA sewersheds in a zipped FileGDB; one flagged, one with two CWNS IDs
    (v / "usa_epa_sewersheds").mkdir()
    us = [p for p in PLANTS if p[0] == "US"]
    sew = gpd.GeoDataFrame(
        {
            "CWNS_ID": ["01000127002", "01000127005", "01000139001", "01000139009; 01000139010"],
            "Method": ["Sourced", "Sourced", "Sourced", "Sourced"],
            "Flag": [None, None, "needs_v2_model", None],
            "Pop_2020": [26000, 4000, 900, 500],
        },
        geometry=[_square(p[1], p[2], 0.05) for p in us] + [_square(-75, 40, 0.05)],
        crs=4326,
    )
    gdb = tmp_path / "Version_1_3.gdb"
    sew.to_file(gdb, layer="Sewersheds", driver="OpenFileGDB")
    with zipfile.ZipFile(v / "usa_epa_sewersheds" / "Current_Release.zip", "w") as z:
        for f in os.listdir(gdb):
            z.write(gdb / f, f"Version_1_3.gdb/{f}")
    # UWWTD capacities next to the NL plants (and one far away)
    nl = [p for p in PLANTS if p[0] == "NL"]
    uww = gpd.GeoDataFrame(
        {"uwwCapacity": [25000.0, 9000.0, 160000.0, 1000.0]},
        geometry=[Point(p[1] + 0.001, p[2]) for p in nl] + [Point(6.5, 53.0)],
        crs=4326,
    )
    uww.to_file(tmp_path / "uwwtd.gpkg", driver="GPKG")
    # population rasters, as download_pop names them: GB 1 person per 0.001 degree cell (1 M), NL 1.5 M, US 0.5 M
    for iso3, origin, cell, shape, value in (
        ("gbr", (-1.5, 52.5), 0.001, (1000, 1000), 1.0),
        ("nld", (4.5, 52.5), 0.01, (100, 150), 100.0),
        ("usa", (-73.0, 43.0), 0.01, (100, 100), 50.0),
    ):
        (tmp_path / "pop" / iso3).mkdir(parents=True)
        profile = {
            "driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1, "dtype": "float32",
            "crs": "EPSG:4326", "transform": from_origin(*origin, cell, cell), "nodata": -99999.0,
        }  # fmt: skip
        with rasterio.open(tmp_path / "pop" / iso3 / f"{iso3}_pop_2024_CN_100m_R2024B_v1.tif", "w", **profile) as dst:
            dst.write(np.full((1, *shape), value, dtype="float32"))
    for folder in ("jmp_household", "eu_eurostat", "oecd_water_treat"):
        (v / folder).mkdir()
    return {
        "country_output_column": "ISO_2",
        "basin_column_name": "HYBAS_ID",
        "max_workers": 1,
        "threshold": 500,
        "eu_reference_factor": 1.0,
        "percent_verification": 0.8,
        "reference_years": {"gb": 2024, "nl": 2024, "us": 2020, "hw": 2024, "uwwtd": 2024},
        "gb_match_distance_m": 1000.0,
        "us_match_distance_m": 500.0,
        "national_year": 2024,
        "national_max_year_gap": 5,
        "national_min_plants": 3,
        "national_series": {
            "jmp": "san_sew_sm_t",
            "jmp_sewer": "san_sew_t",
            "eurostat": "URB_OTH_CS",
            "oecd": "PUBTOTTR",
        },
        "eurostat_geo_aliases": {"EL": "GR", "UK": "GB"},
        "decision_weights": {"gb": 0.25, "nl": 0.25, "jmp": 0.5},
        "bootstrap_samples": 200,
        "bootstrap_seed": 0,
        "confidence": 0.95,
        "paths": {
            "pop_tif_dir": str(tmp_path / "pop"),
            "eu_ref_filepath": str(tmp_path / "uwwtd.gpkg"),
            "gb_catchments": str(v / "gbr_catchments" / "catchments_consolidated.zip"),
            "gb_waterbase_lookup": str(v / "gbr_catchments" / "waterbase_catchment_lookup.csv"),
            "gb_waterbase": str(v / "gbr_catchments" / "waterbase_consolidated.csv"),
            "nl_cbs": str(v / "nld_cbs_rwzi" / "cbs.xlsx"),
            "us_sewersheds": str(v / "usa_epa_sewersheds" / "Current_Release.zip"),
            "jmp": str(v / "jmp_household" / "jmp_household.xlsx"),
            "eurostat": str(v / "eu_eurostat" / "eurostat_env_ww_con.tsv"),
            "oecd": str(v / "oecd_water_treat" / "oecd_water_treat.csv"),
            "output_dir": str(tmp_path / "out"),
        },
    }


def _layer(path, people, gb_area):
    """A candidate's population layer: ``people`` per plant (2020 and 2024), GB service areas of half width ``gb_area``."""
    rows = []
    for i, (iso2, lon, lat, basin, area, uww, cwns, pop, qual) in enumerate(PLANTS):
        rows.append(
            {
                "SITE_UID": 1000 + i,
                "ISO_2": iso2,
                "HYBAS_ID": basin,
                "total_area": area,
                "WKT_WWTP": Point(lon, lat).wkt,
                "uwwCode": uww,
                "CWNS_2022_ID": cwns,
                "POP_SERVED": pop,
                "QUAL_POP": qual,
                "2020_zonal_sum": people[i],
                "2024_zonal_sum": people[i],
                "geometry": _square(lon, lat, gb_area if iso2 == "GB" else 0.01),
            }
        )
    gpd.GeoDataFrame(rows, crs=4326).to_file(path, driver="GPKG")


def _truth(cfg):
    """People per plant that every reference agrees with exactly (the GB values from the raster)."""
    raster = sv.country_raster("GB", cfg["paths"]["pop_tif_dir"], 2024)
    gb = gpd.GeoSeries([_square(p[1], p[2], HALF) for p in PLANTS if p[0] == "GB"], crs=4326)
    return np.concatenate([sv.zonal_people(gb, raster), [30000, 8000, 150000, 26000, 4000, 900]])


def _national_files(cfg, shares):
    """JMP, Eurostat and OECD files whose connected shares are ``shares`` (iso3 -> 0-1)."""
    paths = cfg["paths"]
    jmp = pd.DataFrame(
        {
            "iso3": list(shares) + ["GBR"],
            "year": [2022] * len(shares) + [2010],  # the 2010 row is too far from 2024
            "san_sew_sm_t": [100 * v for v in shares.values()] + [1.0],
            "san_sew_t": [120 * v for v in shares.values()] + [2.0],
            "region_sdg": ["Europe and Northern America"] * len(shares) + ["Europe and Northern America"],
        }
    )
    jmp.to_excel(paths["jmp"], sheet_name="san", index=False)
    nl, gb = 100 * shares["NLD"], 100 * shares["GBR"]
    with open(paths["eurostat"], "w") as fh:
        fh.write("freq,ww_tp,unit,geo\\TIME_PERIOD\t2022 \t2023 \n")
        fh.write(f"A,URB_OTH_CS,PC,NL\t{nl} e\t: \nA,URB_OTH_CS,PC,UK\t{gb} \t: \nA,URB_CS,PC,NL\t99.9 \t: \n")
    pd.DataFrame(
        {"REF_AREA": ["USA", "USA"], "MEASURE": ["PUBTOTTR", "OTHER"], "TIME_PERIOD": [2022, 2022],
         "OBS_VALUE": [100 * shares["USA"], 1.0]}
    ).to_csv(paths["oecd"], index=False)  # fmt: skip


def test_references_and_choice_end_to_end(tmp_path, monkeypatch):
    cfg = _world(tmp_path)
    truth = _truth(cfg)
    population = {
        c: sv.raster_total(sv.country_raster(c, cfg["paths"]["pop_tif_dir"], 2024)) for c in ("GB", "NL", "US")
    }
    served = {"GBR": truth[:4].sum() / population["GB"], "NLD": truth[4:7].sum() / population["NL"]}
    _national_files(cfg, {**served, "USA": truth[7:].sum() / population["US"]})
    good, bad = tmp_path / "good.gpkg", tmp_path / "bad.gpkg"
    _layer(good, truth, HALF)  # matches the catchments and the counts
    _layer(bad, truth * 3, HALF / 2)  # three times too many people, a quarter of each GB catchment
    cands = pd.DataFrame(
        {
            "candidate": ["bad", "good"],
            "level": ["6", "6"],
            "approach": ["1", "1"],
            "buffer": ["size", 9000],
            "size_measure": ["reported", ""],
            "weight_method": ["size", "linear"],
            "weight_func": ["mult", "none"],
            "uses_reported_sizes": [True, False],
            "configured": [True, False],
            "path": [str(bad), str(good)],
        }
    )
    monkeypatch.setattr(sv, "candidates", lambda: cands)
    monkeypatch.setattr(sv, "configured_layer", lambda: str(bad))

    sv.build_references(cfg)
    units, matches, catchments, national = sv.read_references(cfg)
    nat = national.set_index("iso2")
    assert nat.loc["NL", "people"] == pytest.approx(1.5e6) and nat.loc["US", "people"] == pytest.approx(5e5)
    assert nat.loc["NL", "eurostat_share"] == pytest.approx(served["NLD"]) and nat.loc["GB", "jmp_year"] == 2022
    assert np.isnan(nat.loc["US", "eurostat_share"]) and nat.loc["US", "oecd_share"] > 0
    n = units.groupby("reference").size().to_dict()
    assert n == {"gb": 4, "nl": 3, "us": 2, "hw": 2, "uwwtd": 3}  # US: flagged and two-ID polygons left out
    gb = units[units["reference"] == "gb"].set_index("unit")
    assert gb["ref"].to_numpy() == pytest.approx(truth[:4], rel=1e-6)
    assert gb.loc["1000", "group"] == "ver" and gb.loc["1002", "group"] == "unver"
    nl = units[units["reference"] == "nl"].set_index("unit")
    assert nl.loc["NL00001", "ref"] == 30000 and nl.loc["NL00001", "group"] == "single"  # municipalities not summed
    assert len(catchments) == 4

    scores, nat_scores, choice = sv.score_all(cfg)
    n = nat_scores.set_index(["candidate", "reference"])
    assert n.loc[("good", "jmp"), "mean_abs_error_pp"] == pytest.approx(0, abs=1e-9)
    assert n.loc[("good", "jmp"), "n_countries"] == 3 and n.loc[("good", "eurostat"), "n_countries"] == 2
    assert n.loc[("bad", "jmp"), "world_ratio"] == pytest.approx(3.0)
    assert n.loc[("good", "jmp_sewer"), "bias_pp"] < 0 and n.loc[("good", "jmp_sewer"), "role"] == "upper bound"
    s = scores[scores["group"] == "all"].set_index(["candidate", "reference"])
    for ref in ("gb", "nl", "us"):
        assert s.loc[("good", ref), "class_balanced_error_x"] == pytest.approx(1.0, abs=1e-6)
        assert s.loc[("bad", ref), "class_balanced_error_x"] == pytest.approx(3.0, rel=1e-6)
        assert s.loc[("bad", ref), "total_ratio"] == pytest.approx(3.0, rel=1e-6)
    assert s.loc[("good", "gb"), "capture"] == pytest.approx(1.0, rel=1e-3)
    assert s.loc[("bad", "gb"), "capture"] == pytest.approx(0.25, rel=1e-2)
    assert s.loc[("bad", "hw"), "role"] == "input" and s.loc[("good", "hw"), "role"] == "independent"
    assert set(scores["group"]) >= {"all", "ver", "unver", "single"}
    c = choice.set_index("candidate")
    assert c.loc["good", "best"] and c.loc["good", "recommended"] and not c.loc["bad", "shortlist"]
    out = cfg["paths"]["output_dir"]
    for name in ("units.parquet", "national_countries.csv", "national_regions.csv", "summary.md"):
        assert os.path.exists(os.path.join(out, name))
    regions = pd.read_csv(os.path.join(out, "national_regions.csv"))
    world = regions[
        (regions["region_sdg"] == "World") & (regions["candidate"] == "good") & (regions["reference"] == "jmp")
    ]
    assert float(world["error_pp"].iloc[0]) == pytest.approx(0, abs=1e-9)


def _people(errors, refs):
    """People tables for choose(): candidate x unit predicted = ref * factor."""
    rows, units = [], []
    for ref, values in refs.items():
        for j, r in enumerate(values):
            units.append({"reference": ref, "unit": f"{ref}{j}", "ref": r})
            for cand, factor in errors.items():
                rows.append({"candidate": cand, "reference": ref, "unit": f"{ref}{j}", "pred": r * factor[ref][j]})
    return pd.DataFrame(rows), pd.DataFrame(units)


def _long(offsets, ref_share):
    """National shares for choose(): candidate x country served = reference + offset."""
    return pd.DataFrame(
        [
            {"candidate": c, "reference": "jmp", "iso2": f"C{j}", "served_share": r + off[j], "ref_share": r}
            for c, off in offsets.items()
            for j, r in enumerate(ref_share)
        ]
    )


def test_the_best_replaces_the_configured_only_if_better_overall_and_nowhere_worse():
    rng = np.random.default_rng(1)
    refs = {"gb": 10 ** rng.uniform(3, 6, 60), "nl": 10 ** rng.uniform(3, 6, 60)}  # every size class
    noise = {r: 10 ** rng.normal(0, 0.15, 60) for r in refs}
    share = rng.uniform(0.1, 0.9, 80)
    gap = rng.normal(0, 0.05, 80)
    cands = pd.DataFrame(
        {"configured": [True, False, False], "uses_reported_sizes": [True, True, False]}, index=["conf", "a", "b"]
    )
    weights = {"gb": 0.25, "nl": 0.25, "jmp": 0.5}
    # a: much better nationally, but clearly worse in nl -> the configured candidate stays
    errors = {
        "conf": {"gb": noise["gb"], "nl": noise["nl"]},
        "a": {"gb": noise["gb"], "nl": noise["nl"] * 1.8},
        "b": {"gb": noise["gb"] * 1.2, "nl": noise["nl"]},
    }
    long = _long({"conf": gap - 0.15, "a": gap, "b": gap - 0.15}, share)
    people, units = _people(errors, refs)
    table, decision = sv.choose(people, units, long, cands, weights, 300, 0, 0.95)
    t = table.set_index("candidate")
    assert t.loc["a", "best"] and decision["worse"]["nl"] and t.loc["conf", "recommended"]
    assert decision["best_by"] == {"plants": "conf", "countries": "a"}
    assert t.loc["conf", "score"] == pytest.approx(1.0)
    assert t.loc["a", "mean_abs_error_pp jmp"] < t.loc["conf", "mean_abs_error_pp jmp"]
    # a as good as conf in the plants -> recommended
    errors["a"]["nl"] = noise["nl"]
    people, units = _people(errors, refs)
    table, decision = sv.choose(people, units, long, cands, weights, 300, 0, 0.95)
    assert table.set_index("candidate").loc["a", "recommended"] and decision["better_overall"]
    assert not any(decision["worse"].values())


def test_metrics_count_an_empty_area_as_a_failure():
    m = sv.metrics([0.0, 2e3, 2e4, 2e5], [1e3, 1e3, 1e4, 1e5], capture=[0, 500, 5e3, 5e4])
    assert m["n_without_people"] == 1 and m["within_factor_2"] == 0.75
    assert m["typical_error_x"] == pytest.approx(2.0)  # median of (inf, log 2, log 2, log 2)
    assert m["class_balanced_error_x"] == np.inf  # the <10k class: median of (inf, log 2) is inf
    assert m["capture"] == pytest.approx(55500 / 112000)


def test_candidates_cover_the_sweep_once_per_approach_0(monkeypatch):
    cands = sv.candidates()
    default = workflow.default_variant()
    assert cands["configured"].sum() == 1
    a0 = cands[cands["approach"] == "0"]
    assert (a0["level"] == "any").all() and len(a0) == len(workflow.sweep_variants()) // len(
        workflow.config("sweep")["sweep_levels"]
    )
    conf = cands[cands["configured"]].iloc[0]
    assert conf["approach"] == workflow.figure_approach(default) and conf["uses_reported_sizes"]
    rigid = cands[(cands["buffer"] != "size") & (cands["weight_method"] != "size")]
    assert not rigid["uses_reported_sizes"].any()
