"""One-time stages (plant locations, image annotation, WorldPop download) run only when a run names them."""

import typing

import pytest

from wwtp import settings
from wwtp import workflow as wf

pytestmark = pytest.mark.unit
ANNOTATION_RULES = {"annotation_grids", "wastewater_join", "download_bing_annotate", "merge_annotations"}


def test_a_run_includes_none_by_default():
    assert wf.config("workflow")["run_once"] == []
    assert wf.one_time_rules({}) == set()


@pytest.mark.parametrize(
    "value, rules",
    [
        ("annotations", ANNOTATION_RULES),
        ("annotations,download_pop", ANNOTATION_RULES | {"download_pop"}),
        (" merge_annotations , download_pop ", {"merge_annotations", "download_pop"}),
        (["combine_locations", "wastewater_join"], {"combine_locations", "wastewater_join"}),
        ("", set()),
    ],
)
def test_the_command_line_names_them_for_one_run(value, rules):
    """`snakemake --config run_once=...` gives a comma-separated string, or a list in YAML form."""
    assert wf.one_time_rules({"run_once": value}) == rules


def test_an_unknown_name_stops_the_run():
    with pytest.raises(settings.SettingsError, match=r"unknown stage\(s\) \['annotation'\]"):
        wf.one_time_rules({"run_once": "annotation"})


def test_config_yaml_and_the_workflow_know_the_same_stages():
    assert set(typing.get_args(settings.OneTimeStage)) == set(wf.ONE_TIME_STAGES)
    with pytest.raises(settings.SettingsError, match="workflow.run_once.0"):
        settings.parse("workflow", {**wf.config("workflow"), "run_once": ["annotation"]})
