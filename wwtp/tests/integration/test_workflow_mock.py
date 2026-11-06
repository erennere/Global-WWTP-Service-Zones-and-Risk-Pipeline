"""The Snakemake workflow runs the configured variant on the mock world and gets the golden results."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pytest

from wwtp import workflow as wf
from wwtp.tests.fixtures.mock_world import build_mock_world
from wwtp.tests.harness import mock_before_after as harness

pytestmark = [pytest.mark.integration, pytest.mark.slow]
SNAKEFILE = harness.REPO / "workflow" / "Snakefile"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "mock_world_chain.json"


def _snakemake(work, *args):
    env = {**os.environ, "WWTP_SERVICE_PIPELINE_CONFIG": str(work / "config.yaml"), "MPLBACKEND": "Agg"}
    env.pop("WWTP_CONFIG_OVERLAYS", None)
    return subprocess.run(
        [sys.executable, "-m", "snakemake", "-s", str(SNAKEFILE), "-c", "4", *args],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    work = tmp_path_factory.mktemp("workflow")
    data = work / "data"
    build_mock_world(data, seed=0)
    shutil.copytree(harness.REPO / "data" / "boundaries", data / "boundaries")  # Natural Earth, for the maps
    harness.write_config(harness.REPO, data, work / "config.yaml")
    stamps = data / "workflow" / "stamps"
    stamps.mkdir(parents=True)
    (stamps / "download_pop.done").touch()  # the mock world brings its rasters; nothing to download
    result = _snakemake(work, "--keep-going")
    yield {"work": work, "data": data, "result": result}
    shutil.rmtree(work, ignore_errors=True)


def test_every_job_succeeds(run):
    assert run["result"].returncode == 0, run["result"].stderr[-3000:]


def test_results_match_the_golden_run_of_the_stage_by_stage_chain(run):
    cfg = run["work"] / "config.yaml"
    paths = harness.resolve_paths(harness.REPO, cfg)
    sites = len(gpd.read_file(run["data"] / "all_annotated_v2.gpkg"))
    metrics = harness.collect(harness.REPO, cfg, paths, str(run["data"]), sites, {"before": None, "after": None})
    current = {
        f"{out}|{key}": value
        for out, values in metrics.items()
        if isinstance(values, dict)
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    golden = json.loads(GOLDEN.read_text())
    assert sorted(current) == sorted(golden), set(current) ^ set(golden)
    for key, value in golden.items():
        assert current[key] == pytest.approx(value, rel=1e-6, abs=1e-6), key


def _variant_config(run, stage):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WWTP_SERVICE_PIPELINE_CONFIG", str(run["work"] / "config.yaml"))
        wf.config.cache_clear()
        try:
            return wf.config(stage, wf.default_variant())
        finally:
            wf.config.cache_clear()


@pytest.mark.parametrize(
    "stage, key",
    [
        ("piechart_figure", "static_piechart_filepath"),
        ("piechart_interactive", "interactive_piechart_html_filepath"),
        ("sizes_interactive_map", "sizes_interactive_html_filepath"),
        ("composite_area_population_plots", "composite_scatter_filepath"),
        ("interactive_unconnected_industrial_map", "industrial_unconnected_html_filepath"),
    ],
)
def test_each_figure_job_ran_its_own_script(run, stage, key):
    """Each figure rule must run its own module (a loop variable in a shell string once made them all run the last)."""
    assert os.path.exists(_variant_config(run, stage)["paths"][key])


def test_create_rasters_ran_as_shards_and_merged(run):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WWTP_SERVICE_PIPELINE_CONFIG", str(run["work"] / "config.yaml"))
        wf.config.cache_clear()
        variant = wf.default_variant()
        shard_files = [f for i in range(wf.raster_shards(variant)) for f in wf.raster_shard_files(variant, i)]
        stats = wf.raster_stats(variant)
        wf.config.cache_clear()
    assert len(shard_files) == 20 and all(os.path.exists(f) for f in shard_files + stats)


def test_a_second_run_has_nothing_to_do(run):
    again = _snakemake(run["work"], "-n")
    assert again.returncode == 0 and "Nothing to be done" in again.stdout + again.stderr


def test_fig1_targets_chain_from_the_prepared_inputs(tmp_path):
    """`snakemake fig1` plans every Figure 1 step from the inputs prepared in match/data and cartogram/data."""
    data = tmp_path / "data"
    fig1 = data / "figures" / "fig1"
    for folder, names in {
        "match/data": [
            "heigit_points.csv",
            "uwwtd_points.csv",
            "HydroWASTE_v10.csv",
            "ne_50m_admin_0_countries.geojson",
            "HeiGIT_WWTP_LVL9_BFK0_8_LOG_MULT.gpkg",
        ],
        "cartogram/data": [
            "non_served_points.csv",
            "heigit_points.csv",
            "heigit_pop.csv",
            "ne_50m_admin_0_countries.geojson",
        ],
        "hydrobasins": ["hybas_lev03_v1c.gpkg"],
    }.items():
        (fig1 / folder).mkdir(parents=True, exist_ok=True)
        for name in names:
            (fig1 / folder / name).touch()
    harness.write_config(harness.REPO, data, tmp_path / "config.yaml")
    plan = _snakemake(tmp_path, "-n", "fig1")
    assert plan.returncode == 0, plan.stderr[-2000:]
    jobs = {line.split()[0] for line in plan.stdout.splitlines() if line.startswith("fig1")}
    assert jobs == {
        "fig1",
        "fig1_reconcile",
        "fig1_aggregate",
        "fig1_aggregate_basins",
        "fig1_maps",
        "fig1_tanks",
        "fig1_service",
        "fig1_cartogram",
        "fig1_cartogram_figures",
        "fig1_combined",
    }
