"""Typed settings: config.yaml and wwtp/settings.py describe the same sections, and bad values stop at load."""

import pytest

from wwtp import settings, starter

pytestmark = pytest.mark.unit


def _raw():
    raw, _ = starter.load_layered_config(starter.resolve_config_path(None))
    return raw


def test_every_section_has_a_model_and_every_model_a_section():
    assert set(_raw()) == set(settings.SECTIONS)


@pytest.mark.parametrize("section", list(settings.SECTIONS))
def test_config_yaml_validates_without_changing_values(section):
    resolved = starter.resolve_config(section, _raw())
    validated = settings.validate(section, resolved)
    assert list(validated) == list(resolved)
    assert validated == resolved  # 1 == 1.0, so float-typed ints compare equal


def test_level_and_version_are_strings_whatever_the_yaml_type():
    cfg = starter.load_config(script_name="create_voronoi", sets=[(("combine_watersheds", "level"), 7)])
    assert cfg["level"] == "7" and "lvl7" in cfg["paths"]["voronoi_dir"]
    with pytest.raises(settings.SettingsError, match="combine_watersheds.level: String should match"):
        starter.load_config(script_name="combine_watersheds", sets=[(("combine_watersheds", "level"), 13)])


def test_every_problem_is_listed_with_its_key():
    values = starter.resolve_config("create_voronoi", _raw())
    values.update(weight_method="sqrt", n_points=0, typo=1)
    del values["sigma"]
    with pytest.raises(settings.SettingsError) as err:
        settings.validate("create_voronoi", values)
    message = str(err.value)
    for line in (
        "create_voronoi.weight_method: Input should be",
        "create_voronoi.n_points: Input should be greater",
        "create_voronoi.sigma: Field required",
        "create_voronoi.typo: Extra inputs are not permitted",
    ):
        assert line in message


@pytest.mark.parametrize(
    "section, change, message",
    [
        ("create_voronoi", {"weight_func": "add"}, "needs weight_func 'mult'"),
        ("create_voronoi", {"dynamic_buffering": True, "size_measure": None}, "size_measure must be configured"),
        ("download_pop", {"start_year": 2030, "end_year": 2020}, "start_year 2030 is after end_year 2020"),
        ("sweep", {"sweep_size_measures": ["capacity"]}, "sweep_size_measures.0"),
        ("create_rasters", {"shards": 0}, "create_rasters.shards"),
        ("impact_polygons_pop", {"impact_polygons_pop_params": {"org_per_pop": 1}}, "c_limit: Field required"),
    ],
)
def test_cross_field_and_allowed_value_rules(section, change, message):
    values = {**starter.resolve_config(section, _raw()), **change}
    with pytest.raises(settings.SettingsError, match=message):
        settings.validate(section, values)


def test_unknown_section_is_named():
    with pytest.raises(settings.SettingsError, match="no settings model"):
        settings.parse("not_a_stage", {})


def test_parse_gives_attribute_access():
    model = settings.parse("create_voronoi", starter.resolve_config("create_voronoi", _raw()))
    assert model.weight_method == "size" and model.radius_calibration["reported"].radius_at_10k_people_km > 0
