"""Every config section resolves, and bad values fail at load time."""

import pytest
import yaml

from wwtp import starter

pytestmark = pytest.mark.unit

SECTIONS = list(yaml.safe_load(open(starter.resolve_config_path(None)).read()))


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_resolves(section):
    cfg = starter.load_config(script_name=section)
    assert isinstance(cfg, dict)
    for key, value in (cfg.get("paths") or {}).items():
        assert not (isinstance(value, str) and "{" in value), f"{section}.paths.{key} left unexpanded: {value}"


@pytest.mark.parametrize(
    "override, message",
    [
        (dict(size_measure="capacity"), "size_measure"),
        (dict(weight_method="cubic"), "create_voronoi.weight_method"),
        (dict(weight_func="add"), "needs weight_func 'mult'"),  # default weight_method is 'size'
        (dict(weight_func="pow"), "weight_func"),
    ],
)
def test_bad_values_fail_at_load(override, message):
    with pytest.raises(ValueError, match=message):
        starter.load_config(script_name="create_voronoi", **override)


def test_stages_share_one_basin_level():
    levels = {
        str(starter.load_config(script_name=s)["level"])
        for s in SECTIONS
        if "level" in starter.load_config(script_name=s)
    }
    assert levels == {"6"}


def test_every_package_is_installed_by_pyproject():
    """A package missing from [tool.setuptools] packages works from the repo but not when installed."""
    import tomllib
    from pathlib import Path

    root = Path(starter.__file__).resolve().parent
    listed = set(tomllib.loads((root.parent / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"])
    on_disk = {
        ".".join(p.parent.relative_to(root.parent).parts)
        for p in root.rglob("__init__.py")
        if "tests" not in p.parts and "maps" not in p.parts
    }
    assert on_disk <= listed, on_disk - listed


def test_overlays_and_set_apply_in_order(tmp_path, monkeypatch):
    cluster = tmp_path / "config.cluster.yaml"
    cluster.write_text("create_voronoi:\n  max_workers: 64\n  n_points: 50\n")
    monkeypatch.setenv(starter.OVERLAYS_ENV_VAR, str(cluster))
    sets = starter.parse_set_assignments(["create_voronoi.n_points=200", "combine_watersheds.level=7"])
    cfg = starter.load_config(script_name="create_voronoi", sets=sets)
    assert cfg["max_workers"] == 64  # overlay
    assert cfg["n_points"] == 200  # --set wins over the overlay
    assert str(cfg["level"]) == "7"  # --set on the section it is inherited from
    assert cfg["config_layers"][1] == str(cluster) and cfg["config_layers"][-1].startswith(
        "--set combine_watersheds.level"
    )


def test_local_overlay_next_to_the_config_is_picked_up(tmp_path, monkeypatch, write_test_config):
    config_path = write_test_config()
    (config_path.parent / "config.local.yaml").write_text("create_voronoi:\n  n_points: 75\n")
    monkeypatch.setattr(starter, "LOCAL_OVERLAY_FILENAME", "config.local.yaml")
    assert starter.load_config(script_name="create_voronoi", config=str(config_path))["n_points"] == 75


@pytest.mark.parametrize("overlay", ["create_voronoi:\n  n_pointz: 3\n", "create_voronoy:\n  n_points: 3\n"])
def test_overlay_typos_fail_at_load(tmp_path, monkeypatch, overlay):
    bad = tmp_path / "bad.yaml"
    bad.write_text(overlay)
    monkeypatch.setenv(starter.OVERLAYS_ENV_VAR, str(bad))
    with pytest.raises(KeyError, match="not a key of the base config"):
        starter.load_config(script_name="create_voronoi")


@pytest.mark.parametrize("item", ["n_points=3", "create_voronoi.n_points", ".=3"])
def test_bad_set_assignments_fail(item):
    with pytest.raises(ValueError, match="Invalid --set"):
        starter.parse_set_assignments([item])


def test_set_reaches_stages_through_the_shared_flags():
    overrides = starter.parse_config_overrides(
        argv=["prog", "--set", "create_voronoi.n_points=150", "--set=create_voronoi.sigma=2"]
    )
    assert overrides["sets"] == [(("create_voronoi", "n_points"), 150), (("create_voronoi", "sigma"), 2)]


def test_show_prints_the_resolved_section(capsys):
    starter.main(["show", "create_voronoi", "--set", "create_voronoi.n_points=123"])
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["n_points"] == 123 and shown["config_layers"][-1] == "--set create_voronoi.n_points=123"


def test_cluster_example_overlay_is_valid(monkeypatch):
    from pathlib import Path

    example = Path(starter.__file__).resolve().parent / "config.cluster.example.yaml"
    monkeypatch.setenv(starter.OVERLAYS_ENV_VAR, str(example))
    cfg = starter.load_config(script_name="add_pop")
    assert cfg["paths"]["data_dir"] == "/path/to/cluster/data"
