from __future__ import annotations


import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from wwtp.annotation_scripts import merge_annotations


pytestmark = pytest.mark.unit


def test_decode_gen_text_parses_number_name_and_justification():
    number, name, why = merge_annotations.decode_gen_text(
        "Analysis: tanks are visible.\n\n"
        "Decision: 3. Industrial\n"
        "Justification: visible clarifier tanks"
    )

    assert number == "3"
    assert name == "Industrial"
    assert why == "visible clarifier tanks"


def test_decode_gen_text_returns_none_fields_for_malformed_or_non_string_inputs():
    assert merge_annotations.decode_gen_text(123) == (None, None, None)
    # No "Decision:" line at all - there is nothing to parse.
    assert merge_annotations.decode_gen_text("just some prose") == (None, None, None)
    # Bracketed placeholders mean "model did not fill this in"; _clean_field
    # drops them, so only the number survives.
    assert merge_annotations.decode_gen_text(
        "Decision: 3. [unclear]\nJustification: [unclear]"
    ) == ("3", None, None)


def test_parse_idx_from_image_name_extracts_trailing_digits_or_none():
    assert merge_annotations.parse_idx_from_image_name("tile_00123.png") == 123
    assert merge_annotations.parse_idx_from_image_name("folder/abc42.jpg") == 42
    assert merge_annotations.parse_idx_from_image_name("no_digits_here.png") is None
    assert merge_annotations.parse_idx_from_image_name(None) is None


def test_merge_annotations_main_raises_for_missing_required_columns(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "annotated_images_output_dir": str(tmp_path / "images"),
            "annotations_results_filepath": str(tmp_path / "annotations.csv"),
            "corrected_all_filepath": str(tmp_path / "corrected_all.gpkg"),
        }
    }

    monkeypatch.setattr(merge_annotations.os, "chdir", lambda path: None)
    monkeypatch.setattr(merge_annotations, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(merge_annotations, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(merge_annotations.pd, "read_csv", lambda path: pd.DataFrame({"image": ["tile_1.png"]}))

    with pytest.raises(KeyError, match="Missing expected annotation columns"):
        merge_annotations.main()


def test_merge_annotations_main_merges_by_idx_and_writes_output(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "annotated_images_output_dir": str(tmp_path / "images"),
            "annotations_results_filepath": str(tmp_path / "annotations.csv"),
            "corrected_all_filepath": str(tmp_path / "corrected_all.gpkg"),
            "annotated_all_filepath": str(tmp_path / "annotated_all.gpkg"),
        }
    }
    captured = {}

    annotations_df = pd.DataFrame(
        {
            "image": ["tile_1.png", "tile_2.png", "badname.png"],
            "gen_text": [
                "Decision: 1. Residential\nJustification: lagoon",
                "Decision: 3. Industrial\nJustification: reactors",
                "noise",
            ],
        }
    )
    points_df = gpd.GeoDataFrame(
        {
            "idx": [1, 2, 3],
            "category_name": ["old", "old", "old"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(merge_annotations.os, "chdir", lambda path: None)
    monkeypatch.setattr(merge_annotations, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(merge_annotations, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(merge_annotations.pd, "read_csv", lambda path: annotations_df.copy())
    monkeypatch.setattr(merge_annotations.gpd, "read_file", lambda path: points_df.copy())
    monkeypatch.setattr(
        "wwtp.utils.ensure_output_dir_for_file",
        lambda path: captured.setdefault("ensured", path),
    )

    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, filename=None, driver=None, index=None, **kwargs):
        captured["write"] = {
            "filename": filename,
            "driver": driver,
            "index": index,
            "rows": len(self),
            "category_names": self["category_name"].tolist(),
        }

    try:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)
        merge_annotations.main()
    finally:
        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)

    assert captured["ensured"] == cfg["paths"]["annotated_all_filepath"]
    assert captured["write"]["filename"] == cfg["paths"]["annotated_all_filepath"]
    assert captured["write"]["driver"] == "GPKG"
    assert captured["write"]["index"] is False
    assert captured["write"]["rows"] == 3
    assert captured["write"]["category_names"][0] == "Residential"
    assert captured["write"]["category_names"][1] == "Industrial"


def test_merge_annotations_main_requires_idx_column_in_points(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "annotated_images_output_dir": str(tmp_path / "images"),
            "annotations_results_filepath": str(tmp_path / "annotations.csv"),
            "corrected_all_filepath": str(tmp_path / "corrected_all.gpkg"),
        }
    }

    annotations_df = pd.DataFrame(
        {
            "image": ["tile_1.png"],
            "gen_text": ["Decision: 1. Residential\nJustification: lagoon"],
        }
    )
    points_df = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(merge_annotations.os, "chdir", lambda path: None)
    monkeypatch.setattr(merge_annotations, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(merge_annotations, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(merge_annotations.pd, "read_csv", lambda path: annotations_df.copy())
    monkeypatch.setattr(merge_annotations.gpd, "read_file", lambda path: points_df.copy())

    with pytest.raises(KeyError, match="required 'idx' column"):
        merge_annotations.main()
