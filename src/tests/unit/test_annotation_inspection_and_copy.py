from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from src.annotation_scripts import annotations_inspection, copy_falsy_images


pytestmark = pytest.mark.unit


def test_sanitize_folder_name_replaces_forbidden_chars_and_handles_empty():
    assert annotations_inspection.sanitize_folder_name('A<B>:C|D?*"') == "A_B__C_D___"
    assert annotations_inspection.sanitize_folder_name("   ") == "Uncategorized"


def test_get_stratified_sample_balances_categories_and_backfills_shortfall():
    df = pd.DataFrame(
        {
            "category_name": ["A", "A", "B", "B", None],
            "image": ["1.png", "2.png", "3.png", "4.png", "5.png"],
        }
    )

    sampled = annotations_inspection.get_stratified_sample(df, "category_name", total_n=4, seed=7)

    assert len(sampled) == 4
    counts = sampled["category_name"].fillna("NaN").value_counts().to_dict()
    assert counts["A"] >= 1
    assert counts["B"] >= 1


def test_organize_files_by_category_copies_existing_files(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    image_path = source_dir / "img1.png"
    image_path.write_text("x", encoding="utf-8")

    df = pd.DataFrame(
        {
            "filepath": [str(image_path), str(source_dir / "missing.png")],
            "category_name": ["A/B", None],
        }
    )

    annotations_inspection.organize_files_by_category(df, "filepath", "category_name", str(tmp_path / "out"))

    copied = tmp_path / "out" / "A_B" / "img1.png"
    assert copied.exists()


def test_plot_category_distribution_writes_file(tmp_path):
    df = pd.DataFrame({"category_name": ["A", "A", "B", None]})
    outpath = tmp_path / "hist.png"

    annotations_inspection.plot_category_distribution(df, save_path=str(outpath), show=False)

    assert outpath.exists()


def test_copy_falsy_images_main_copies_expected_subset(monkeypatch, tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    present_img = input_dir / "keep.png"
    present_img.write_text("stub", encoding="utf-8")

    out_dir = tmp_path / "verification"
    corrected_path = str(tmp_path / "corrected_all.gpkg")
    cfg = {
        "paths": {
            "annotated_all_filepath": corrected_path,
            "annotated_images_output_dir": str(input_dir),
            "annotations_verf_image_outpath_dir": str(out_dir / "x"),
        }
    }

    gdf = gpd.GeoDataFrame(
        {
            "category_number": [pd.NA, 8, 5],
            "image": ["keep.png", "keep.png", "skip.png"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(copy_falsy_images.os, "chdir", lambda path: None)
    monkeypatch.setattr(copy_falsy_images, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(copy_falsy_images, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(copy_falsy_images.gpd, "read_file", lambda path: gdf.copy())

    copy_falsy_images.main()

    falsy_dir = out_dir / "falsy_images"
    assert (falsy_dir / "keep.png").exists()
    assert not (falsy_dir / "skip.png").exists()


def test_copy_falsy_images_main_handles_missing_source_file(monkeypatch, tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()

    cfg = {
        "paths": {
            "annotated_all_filepath": str(tmp_path / "corrected_all.gpkg"),
            "annotated_images_output_dir": str(input_dir),
            "annotations_verf_image_outpath_dir": str(tmp_path / "verification" / "x"),
        }
    }
    gdf = gpd.GeoDataFrame(
        {
            "category_number": [8],
            "image": ["missing.png"],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(copy_falsy_images.os, "chdir", lambda path: None)
    monkeypatch.setattr(copy_falsy_images, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(copy_falsy_images, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(copy_falsy_images.gpd, "read_file", lambda path: gdf.copy())

    copy_falsy_images.main()

    falsy_dir = tmp_path / "verification" / "falsy_images"
    assert not (falsy_dir / "missing.png").exists()


def test_organize_files_by_category_counts_copy_errors(monkeypatch, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    image_path = src / "img.png"
    image_path.write_text("stub", encoding="utf-8")
    df = pd.DataFrame({"filepath": [str(image_path)], "category_name": ["A"]})

    monkeypatch.setattr(annotations_inspection.shutil, "copy2", lambda src_path, dst_path: (_ for _ in ()).throw(RuntimeError("copy fail")))

    annotations_inspection.organize_files_by_category(df, "filepath", "category_name", str(tmp_path / "out"))

    assert not (tmp_path / "out" / "A" / "img.png").exists()


def test_annotations_inspection_main_raises_when_required_columns_missing(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "annotated_images_output_dir": str(tmp_path / "images"),
            "annotations_verf_image_outpath_dir": str(tmp_path / "out"),
            "annotations_results_filepath": str(tmp_path / "annotations.csv"),
        },
        "annotations": {"n_sample_size": 5, "random_seed": 1},
    }

    monkeypatch.setattr(annotations_inspection.os, "chdir", lambda path: None)
    monkeypatch.setattr(annotations_inspection, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(annotations_inspection, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(annotations_inspection.pd, "read_csv", lambda path: pd.DataFrame({"image": ["1.png"]}))

    with pytest.raises(KeyError, match="Missing expected annotation columns"):
        annotations_inspection.main()


def test_annotations_inspection_main_runs_sampling_and_exports(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "annotated_images_output_dir": str(tmp_path / "images"),
            "annotations_verf_image_outpath_dir": str(tmp_path / "out"),
            "annotations_results_filepath": str(tmp_path / "annotations.csv"),
        },
        "annotations": {"n_sample_size": 2, "random_seed": 11},
    }
    captured = {}
    df = pd.DataFrame(
        {
            "image": ["1.png", "2.png"],
            "gen_text": ["1.Residential: note", "3.Industrial: note"],
        }
    )

    monkeypatch.setattr(annotations_inspection.os, "chdir", lambda path: None)
    monkeypatch.setattr(annotations_inspection, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(annotations_inspection, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(annotations_inspection.pd, "read_csv", lambda path: df.copy())
    monkeypatch.setattr(
        annotations_inspection,
        "plot_category_distribution",
        lambda frame, column, save_path=None, show=False: captured.setdefault("hist", save_path),
    )
    monkeypatch.setattr(
        annotations_inspection,
        "organize_files_by_category",
        lambda frame, source_col, category_col, base_dir: captured.setdefault("organized", (len(frame), source_col, category_col, base_dir)),
    )

    original_to_csv = pd.DataFrame.to_csv

    def fake_to_csv(self, path, index=False, **kwargs):
        captured["csv"] = {"path": path, "index": index, "rows": len(self)}

    try:
        monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
        annotations_inspection.main()
    finally:
        monkeypatch.setattr(pd.DataFrame, "to_csv", original_to_csv)

    assert captured["csv"]["index"] is False
    assert captured["csv"]["rows"] == 2
    assert captured["organized"][0] == 2


def test_annotations_inspection_main_rejects_non_positive_sample_size(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "annotated_images_output_dir": str(tmp_path / "images"),
            "annotations_verf_image_outpath_dir": str(tmp_path / "out"),
            "annotations_results_filepath": str(tmp_path / "annotations.csv"),
        },
        "annotations": {"n_sample_size": 0, "random_seed": 11},
    }
    df = pd.DataFrame(
        {
            "image": ["1.png"],
            "gen_text": ["1.Residential: note"],
        }
    )

    monkeypatch.setattr(annotations_inspection.os, "chdir", lambda path: None)
    monkeypatch.setattr(annotations_inspection, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(annotations_inspection, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(annotations_inspection.pd, "read_csv", lambda path: df.copy())

    with pytest.raises(ValueError, match="n_sample_size"):
        annotations_inspection.main()


def test_copy_falsy_images_main_requires_expected_columns(monkeypatch, tmp_path):
    cfg = {
        "paths": {
            "annotated_all_filepath": str(tmp_path / "corrected_all.gpkg"),
            "annotated_images_output_dir": str(tmp_path / "images"),
            "annotations_verf_image_outpath_dir": str(tmp_path / "verification" / "x"),
        }
    }
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(copy_falsy_images.os, "chdir", lambda path: None)
    monkeypatch.setattr(copy_falsy_images, "parse_config_overrides", lambda *a, **k: {})
    monkeypatch.setattr(copy_falsy_images, "load_config", lambda **overrides: cfg)
    monkeypatch.setattr(copy_falsy_images.gpd, "read_file", lambda path: gdf.copy())

    with pytest.raises(KeyError, match="Missing expected columns"):
        copy_falsy_images.main()
