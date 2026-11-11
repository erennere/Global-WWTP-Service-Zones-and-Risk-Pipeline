"""wwtp.workflow: the variants and file lists the Snakemake workflow is built from."""

import re
from pathlib import Path

import pytest

from wwtp import starter
from wwtp import workflow as wf

pytestmark = pytest.mark.unit
SNAKEFILE = Path(__file__).resolve().parents[3] / "workflow" / "Snakefile"


def test_the_sweep_grid_comes_from_config():
    variants = wf.sweep_variants()
    assert {v.level for v in variants} == {"5", "6", "7"}
    assert {v.weight_func for v in variants} == {"mult", ""}  # 'add' only when sweep_weight_funcs lists it
    assert not [v for v in variants if v.weight_func == "" and v.weight_method != "linear"]
    assert not [v for v in variants if v.weight_method == "size" and v.weight_func != "mult"]
    assert {v.size_measure for v in variants if v.dynamic_buffering} == {"reported", "footprint"}
    assert len({v.tag for v in variants}) == len(variants) == 108
    assert {v.weight_func for v in wf.sweep_variants(["add"])} == {"add"}


def test_default_variant_is_the_configured_run_at_basin_level_6():
    v = wf.default_variant()
    assert (v.level, v.weight_method, v.weight_func, v.dynamic_buffering, v.size_measure) == (
        "6",
        "size",
        "mult",
        True,
        "reported",
    )
    assert v.tag == "v2_lvl6_bfrep_sz_mult"


@pytest.mark.parametrize("variant", [wf.default_variant(), *wf.sweep_variants()[::17]], ids=lambda v: v.tag)
def test_command_line_flags_give_back_the_same_variant(variant):
    """Every stage parses Variant.cli(); an empty weight_func must stay '' (not fall back to the config's 'mult')."""
    overrides = starter.parse_config_overrides(argv=["prog", *variant.cli()])
    cfg = starter.load_config(script_name="create_voronoi", **overrides)
    assert wf.Variant.from_section(cfg) == variant


def test_file_names_carry_the_tag():
    v = wf.sweep_variants()[0]
    assert v.tag in Path(wf.service_areas(v, "1")).name and v.tag in Path(wf.population_layer(v, "1")).name
    assert all(v.tag in Path(p).name for p in wf.raster_stats(v) + wf.impact_polygons(v))
    assert Path(wf.raster_shard_files(v, 3)[0]).name.endswith(f"_shard3of{wf.raster_shards(v)}.csv")
    assert "lvl7" in wf.watershed(7)


def test_every_stage_of_the_snakefile_has_resources():
    stages = set(re.findall(r'(?:res|threads)\("(\w+)"\)', SNAKEFILE.read_text()))
    stages |= {
        "hw_comparison",
        "eu_comparison",
        "piechart_figure",
        "piechart_interactive",
        "sizes_interactive_map",
        "composite_area_population_plots",
        "pop_at_risk_figures",
        "interactive_unconnected_industrial_map",
    }
    for stage in stages:
        assert set(wf.resources(stage)) == {"threads", "mem_mb", "runtime"}, stage


def test_no_two_variants_write_the_same_file():
    """Every per-variant output carries the whole tag (an industrial name without weight_func once made
    the 'li' and 'li_mult' variants write one file, and Snakemake refused the sweep)."""
    for outputs in (
        [wf.service_areas(v, a) for v in wf.sweep_variants() for a in wf.approaches(v)],
        [wf.population_layer(v, a) for v in wf.sweep_variants() for a in wf.approaches(v)],
        [
            wf.config("find_unconnected_industrial_areas", v)["paths"]["industrial_unconnected_output"]
            for v in wf.sweep_variants()
        ],
    ):
        assert len(set(outputs)) == len(outputs)


def test_approach0_is_made_once_per_weighting_and_extent():
    """Approach 0 never reads the basins: levels 5 and 7 link the level-6 files instead of recomputing them."""
    sweep = wf.sweep_variants()
    tags = {v.tag for v in sweep}
    sources = [wf.approach0_source(v) for v in sweep]
    assert sum(source is not None for source in sources) == 72
    assert all(source.level == "6" and source.tag in tags for source in sources if source is not None)
    assert all(wf.approach0_source(v) is None for v in sweep if v.level == "6")
    assert wf.approach0_source(wf.default_variant()) is None


@pytest.mark.parametrize("variant", [wf.default_variant(), *wf.sweep_variants()[::5]], ids=lambda v: v.tag)
def test_the_shell_command_snakemake_writes_gives_back_the_variant(variant):
    """Through Snakemake's own {params.cli:q} formatting: it leaves an empty element unquoted, which once
    dropped weight_func '' from the command line (create_voronoi: --weight-func expected one argument)."""
    import shlex
    from types import SimpleNamespace

    from snakemake.utils import format as snakemake_format

    command = snakemake_format("{params.cli:q}", params=SimpleNamespace(cli=["--approach", "0", *variant.cli()]))
    overrides = starter.parse_config_overrides(argv=["prog", *shlex.split(command)[2:]])
    cfg = starter.load_config(script_name="create_voronoi", **overrides)
    assert wf.Variant.from_section(cfg) == variant
