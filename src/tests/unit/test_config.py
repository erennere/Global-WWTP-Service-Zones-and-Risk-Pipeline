"""Every config section resolves, and bad values fail at load time."""
import pytest
import yaml

from src import starter

pytestmark = pytest.mark.unit

SECTIONS = list(yaml.safe_load(open(starter.resolve_config_path(None)).read()))


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_resolves(section):
    cfg = starter.load_config(script_name=section)
    assert isinstance(cfg, dict)
    for key, value in (cfg.get("paths") or {}).items():
        assert not (isinstance(value, str) and "{" in value), f"{section}.paths.{key} left unexpanded: {value}"


@pytest.mark.parametrize("override, message", [
    (dict(size_measure="capacity"), "size_measure"),
    (dict(weight_method="cubic"), "Invalid weight_method"),
    (dict(weight_func="add"), "needs weight_func 'mult'"),  # default weight_method is 'size'
    (dict(weight_func="pow"), "weight_func"),
])
def test_bad_values_fail_at_load(override, message):
    with pytest.raises(ValueError, match=message):
        starter.load_config(script_name="create_voronoi", **override)


def test_stages_share_one_basin_level():
    levels = {str(starter.load_config(script_name=s)["level"]) for s in SECTIONS
              if "level" in starter.load_config(script_name=s)}
    assert levels == {"6"}


def test_every_package_is_installed_by_pyproject():
    """A package missing from [tool.setuptools] packages works from the repo but not when installed."""
    import tomllib
    from pathlib import Path

    root = Path(starter.__file__).resolve().parent
    listed = set(tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"])
    on_disk = {"src." + str(p.parent.relative_to(root)).replace("/", ".") if p.parent != root else "src"
               for p in root.rglob("__init__.py") if "tests" not in p.parts and "maps" not in p.parts}
    assert on_disk <= listed, on_disk - listed
