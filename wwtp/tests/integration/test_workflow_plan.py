"""What `snakemake -n` plans: the one-time stages stay out unless the run names them."""

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


def test_a_run_starts_after_the_one_time_stages(work):
    """The annotation intermediates (OSM layers, image stamps) are missing, yet nothing of the chain is planned."""
    result, jobs = _plan(work)
    assert result.returncode == 0, result.stderr[-2000:]
    assert any(job.startswith("create_voronoi__") for job in jobs)
    assert not jobs & ONE_TIME_RULES


def test_forcing_a_one_time_stage_needs_it_named(work):
    result, _ = _plan(work, "-R", "annotation_grids")
    assert result.returncode != 0 and "No rule to produce annotation_grids" in result.stdout + result.stderr


def test_a_missing_stamp_stops_the_plan_until_its_stage_is_named(work):
    os.remove(wf.stamp("download_pop"))
    result, _ = _plan(work)
    assert result.returncode != 0 and "download_pop.done" in result.stdout + result.stderr
    result, jobs = _plan(work, "--config", "run_once=download_pop")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "download_pop" in jobs and not jobs & (ONE_TIME_RULES - {"download_pop"})
