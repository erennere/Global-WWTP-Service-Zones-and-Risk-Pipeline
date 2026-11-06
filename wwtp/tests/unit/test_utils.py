from __future__ import annotations

import os

import geopandas as gpd
import pytest
from shapely.geometry import box

from wwtp import utils

pytestmark = pytest.mark.unit


def test_ensure_output_dir_for_file_creates_parent_directory(tmp_path):
    target = tmp_path / "nested" / "deeper" / "output.gpkg"

    utils.ensure_output_dir_for_file(str(target))

    assert target.parent.is_dir()


def test_ensure_output_dir_for_file_handles_bare_filename():
    # Should not raise when the path has no directory component.
    utils.ensure_output_dir_for_file("output.gpkg")


def test_get_iso_codes_maps_known_country_both_directions():
    alpha_3_to_2, alpha_2_to_3, alpha_3_to_names, alpha_2_to_names = utils.get_iso_codes()

    assert alpha_3_to_2["DEU"] == "DE"
    assert alpha_2_to_3["DE"] == "DEU"
    assert alpha_3_to_names["DEU"] == "Germany"
    assert alpha_2_to_names["DE"] == "Germany"


def test_quote_sql_identifier_wraps_plain_name():
    assert utils.quote_sql_identifier("ISO_2") == '"ISO_2"'


def test_quote_sql_identifier_escapes_embedded_quotes():
    assert utils.quote_sql_identifier('weird"name') == '"weird""name"'


# --- moved from the coverage-campaign files ---


@pytest.mark.real_atomic_output
def test_atomic_output_replaces_only_on_success(tmp_path):
    target = tmp_path / "out" / "layer.gpkg"
    with utils.atomic_output(target) as tmp:
        assert os.path.basename(tmp) == "layer.gpkg" and not target.exists()  # same name: GPKG layer name kept
        gpd.GeoDataFrame({"a": [1]}, geometry=[box(0, 0, 1, 1)], crs=4326).to_file(tmp, driver="GPKG")
    assert gpd.list_layers(target)["name"].tolist() == ["layer"]
    with pytest.raises(RuntimeError), utils.atomic_output(target) as tmp:
        open(tmp, "w").write("half")
        raise RuntimeError("writer failed")
    assert len(gpd.read_file(target)) == 1  # the old file is untouched
    assert sorted(p.name for p in target.parent.iterdir()) == ["layer.gpkg"]  # no temporary folder left
