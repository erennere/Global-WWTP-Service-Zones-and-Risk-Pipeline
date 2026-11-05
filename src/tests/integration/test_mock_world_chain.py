"""The whole pipeline on the mock world, stage by stage, through the real CLIs.

Every stage must finish, and the hand-offs must keep their invariants:
service areas never overlap, never hold more people than the rasters, and
the non-served layer stays outside them.
"""
import glob
import os
import shutil
import sys

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from src.tests.fixtures.mock_world import COUNTRIES, build_mock_world
from src.tests.harness import mock_before_after as harness

pytestmark = [pytest.mark.integration, pytest.mark.slow]
EQUAL_AREA = 6933


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    work = tmp_path_factory.mktemp("chain")
    data_dir = work / "data"
    build_mock_world(data_dir, seed=0)
    cfg_path = harness.write_config(harness.REPO, data_dir, work / "config.yaml")
    harness.PYTHON = sys.executable
    paths = harness.resolve_paths(harness.REPO, cfg_path)
    logs = work / "logs"
    logs.mkdir()
    status = {}
    for name, args in harness.STAGES:
        if name == "add_pop":
            files = harness._gpkgs(paths["add_pop"]["voronoi_dir"])
            runs = [harness.run_module(harness.REPO, cfg_path, ["src.add_pop", "--index", str(i)], logs, f"add_pop_{i}")
                    for i in range(len(files))]
            status[name] = next((r for r in runs if r["returncode"]), runs[0] if runs else {"returncode": 1, "error": "no input"})
        else:
            status[name] = harness.run_module(harness.REPO, cfg_path, args, logs, name)
    sites = len(gpd.read_file(data_dir / "all_annotated_v2.gpkg"))
    metrics = harness.collect(harness.REPO, cfg_path, paths, str(data_dir), sites, {"before": None, "after": None})
    yield {"status": status, "paths": paths, "data_dir": data_dir, "metrics": metrics}
    shutil.rmtree(work, ignore_errors=True)


@pytest.mark.parametrize("stage", [name for name, _ in harness.STAGES])
def test_stage_finishes(chain, stage):
    result = chain["status"][stage]
    assert result["returncode"] == 0, result["error"]


def _service_areas(chain, approach):
    files = [f for f in harness._gpkgs(chain["paths"]["create_voronoi"]["voronoi_dir"])
             if os.path.basename(f).startswith(f"appr_{approach}_v")]
    assert len(files) == 1, files
    return gpd.read_file(files[0]).to_crs(EQUAL_AREA)


@pytest.mark.parametrize("approach", ["0", "1"])
def test_service_areas_never_overlap(chain, approach):
    areas = _service_areas(chain, approach)
    union = areas.geometry.union_all().area
    assert areas.geometry.area.sum() == pytest.approx(union, rel=1e-3)


def test_service_areas_hold_no_more_people_than_the_rasters(chain):
    raster_total = 0.0
    for iso3, *_ in COUNTRIES.values():
        tifs = sorted(glob.glob(os.path.join(str(chain["data_dir"]), "**", f"{iso3}_pop_2024_*.tif"), recursive=True))
        assert tifs, iso3
        with rasterio.open(tifs[0]) as src:
            raster_total += float(src.read(1, masked=True).sum())
    for f in harness._gpkgs(chain["paths"]["add_pop"]["pop_output_dir"]):
        pop = gpd.read_file(f)
        col = harness._latest_zonal_col(pop)
        served = float(np.nansum(pop[col]))
        assert 0 < served <= raster_total * 1.0001, (os.path.basename(f), served, raster_total)


def test_non_served_areas_stay_outside_the_service_areas(chain):
    path = chain["paths"]["find_unserved_pop"]["non_served_above_threshold_outpath"]
    non_served = gpd.read_file(path).to_crs(EQUAL_AREA)
    assert len(non_served), "the mock world has urban people outside every service area"
    approach = chain["paths"]["find_unserved_pop"]["__figures_approach"] or "1"
    served = _service_areas(chain, approach).geometry.union_all()
    overlap = non_served.geometry.intersection(served).area.sum()
    assert overlap <= 0.01 * non_served.geometry.area.sum()


GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "mock_world_chain.json"


def _flat(metrics):
    return {f"{out}|{key}": value for out, values in metrics.items() if isinstance(values, dict)
            for key, value in values.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}


def test_outputs_match_the_golden_run(chain):
    """Restructuring must not change any output. After an intended change in results,
    regenerate with WWTP_UPDATE_GOLDEN=1 and say why in the commit."""
    current = _flat(chain["metrics"])
    if os.environ.get("WWTP_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(dict(sorted(current.items())), indent=1) + "\n")
    golden = json.loads(GOLDEN.read_text())
    assert sorted(current) == sorted(golden), set(current) ^ set(golden)
    for key, value in golden.items():
        assert current[key] == pytest.approx(value, rel=1e-6, abs=1e-6), key
