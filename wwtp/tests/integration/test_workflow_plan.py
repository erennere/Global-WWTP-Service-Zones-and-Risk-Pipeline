"""What `snakemake -n` plans: each target only its part, one-time stages only when named."""

import os
import subprocess
import sys

import pytest

from wwtp import workflow as wf
from wwtp.tests.harness import mock_before_after as harness

pytestmark = [pytest.mark.integration, pytest.mark.slow]
SNAKEFILE = harness.REPO / "workflow" / "Snakefile"
ONE_TIME_RULES = {rule for rules in wf.ONE_TIME_STAGES.values() for rule in rules}


@pytest.fixture
def work(tmp_path, monkeypatch):
    """A data folder with the annotated plants and the download_pop stamp, nothing in between."""
    harness.write_config(harness.REPO, tmp_path / "data", tmp_path / "config.yaml")
    monkeypatch.setenv("WWTP_SERVICE_PIPELINE_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.delenv("WWTP_CONFIG_OVERLAYS", raising=False)
    wf.config.cache_clear()
    for path in (wf.config("create_voronoi")["paths"]["annotated_all_filepath"], wf.stamp("download_pop")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a").close()
    yield tmp_path
    wf.config.cache_clear()


def _plan(work, *args):
    result = subprocess.run(
        [sys.executable, "-m", "snakemake", "-s", str(SNAKEFILE), "-n", "-c", "1", *args],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=600,
    )
    jobs = set()
    in_stats = False
    for line in result.stdout.splitlines():
        if line.startswith("job "):
            in_stats = True
        elif in_stats and line.startswith("total"):
            break
        elif in_stats and line.split() and not line.startswith("-"):
            jobs.add(line.split()[0])
    return result, jobs


def _kind(job):
    """Rule name without the variant tag or basin level: create_voronoi__v2_lvl6_..._a1 -> create_voronoi."""
    return job.split("__")[0].split("_lvl")[0]


RISK = {
    "create_rasters",
    "find_unserved_pop",
    "find_intersection_river",
    "impact_polygons_pop",
    "find_pop_in_danger_pop",
}
FIGURES = {
    "piechart_figure",
    "piechart_interactive",
    "sizes_interactive_map",
    "composite_area_population_plots",
    "pop_at_risk_figures",
    "interactive_unconnected_industrial_map",
    "figures",
}
INDUSTRIAL = {"find_unconnected_industrial_areas", "download_and_vectorize"}  # configured variant only
COMPARISONS = {"verification_script", "hw_comparison", "eu_comparison"}
SWEEP = {"compare_pop_sweep_hw_eu", "sweep_ver_ranking_task", "sweep_ver_ranking_merge", "sweep_layers", "sweep"}


def test_snakemake_alone_runs_nothing(work):
    result, jobs = _plan(work)
    assert result.returncode == 0, result.stderr[-2000:]
    assert jobs == {"help"}


@pytest.mark.parametrize(
    "target, wanted, unwanted",
    [
        ("service_areas", {"create_voronoi", "add_pop"} | COMPARISONS, RISK | INDUSTRIAL | FIGURES | SWEEP),
        ("risk", RISK | INDUSTRIAL | {"find_diff_pop"}, COMPARISONS | FIGURES | SWEEP),
        ("analysis", RISK | INDUSTRIAL | COMPARISONS | {"find_diff_pop"}, FIGURES | SWEEP),
        ("figures", FIGURES | RISK, SWEEP | {"hw_comparison", "eu_comparison", "find_diff_pop"}),
        (
            "sweep",
            SWEEP - {"sweep_layers"},
            FIGURES | RISK | INDUSTRIAL | {"hw_comparison", "eu_comparison", "find_diff_pop"},
        ),
        ("all", FIGURES | RISK | (SWEEP - {"sweep_layers", "sweep"}), set()),  # target rules are not jobs here
    ],
)
def test_each_target_plans_only_its_part(work, target, wanted, unwanted):
    result, jobs = _plan(work, target)
    assert result.returncode == 0, result.stderr[-2000:]
    kinds = {_kind(job) for job in jobs}
    assert wanted <= kinds, wanted - kinds
    assert not kinds & unwanted, kinds & unwanted
    assert not kinds & ONE_TIME_RULES


def test_a_run_starts_after_the_one_time_stages(work):
    """The annotation intermediates (OSM layers, image stamps) are missing, yet nothing of the chain is planned."""
    result, jobs = _plan(work, "analysis")
    assert result.returncode == 0, result.stderr[-2000:]
    assert any(job.startswith("create_voronoi__") for job in jobs)
    assert not jobs & ONE_TIME_RULES


def test_forcing_a_one_time_stage_needs_it_named(work):
    result, _ = _plan(work, "analysis", "-R", "annotation_grids")
    assert result.returncode != 0 and "No rule to produce annotation_grids" in result.stdout + result.stderr


def test_a_missing_stamp_stops_the_plan_until_its_stage_is_named(work):
    os.remove(wf.stamp("download_pop"))
    result, _ = _plan(work, "analysis")
    assert result.returncode != 0 and "download_pop.done" in result.stdout + result.stderr
    result, jobs = _plan(work, "analysis", "--config", "run_once=download_pop")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "download_pop" in jobs and not jobs & (ONE_TIME_RULES - {"download_pop"})


def test_the_sweep_makes_approach0_once_per_weighting_and_extent(work):
    result, jobs = _plan(work, "sweep")
    assert result.returncode == 0, result.stderr[-2000:]
    voronoi_a0 = {job for job in jobs if job.startswith("create_voronoi__") and job.endswith("__a0")}
    assert voronoi_a0 and all("_lvl6_" in job for job in voronoi_a0)
    # The scoring reads the population layers; a shared service-area file is linked only when asked for.
    shared = {job for job in jobs if job.startswith("share_approach0_pop__")}
    assert shared and all("_lvl6_" not in job for job in shared)
