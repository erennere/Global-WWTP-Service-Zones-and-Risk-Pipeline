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
