"""Sampling annotated images per class: decoding, per-class caps, reproducible draws."""
import json
import sys

import pandas as pd
import pytest

from src.annotation_scripts import sample_annotations_by_class as sac

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("text, expected", [("... Decision: 4. Industrial", "4"), ("no decision", None), (None, None)])
def test_decode_category_number(text, expected):
    assert sac._decode_category_number(text) == expected


def test_main_caps_each_class_and_is_reproducible(tmp_path, monkeypatch):
    results = tmp_path / "res.csv"
    rows = [{"gen_text": f"Decision: {c}.", "image": f"img_{c}_{i}.png"} for c, n in ((1, 12), (2, 3), (10, 5)) for i in range(n)]
    rows.append({"gen_text": "garbled", "image": "x.png"})
    pd.DataFrame(rows).to_csv(results, index=False)
    out = tmp_path / "sampled.json"
    cfg = {"sampling": {"n_per_class": 4}, "annotations": {"random_seed": 7},
           "paths": {"annotations_results_filepath": str(results), "sampled_images_output_filepath": str(out)}}
    monkeypatch.setattr(sac, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(sys, "argv", ["sample_annotations_by_class"])
    sac.main()
    first = json.load(open(out))
    assert [len(g) for g in first] == [4, 3, 4]                         # classes 1, 2, 10 in numeric order
    assert all(name.startswith("img_1_") for name in first[0])
    sac.main()
    assert json.load(open(out)) == first                                 # same seed, same sample


def test_main_needs_the_annotation_columns(tmp_path, monkeypatch):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["Decision: 1."]}).to_csv(bad, index=False)
    cfg = {"sampling": {"n_per_class": 1}, "annotations": {"random_seed": 1},
           "paths": {"annotations_results_filepath": str(bad), "sampled_images_output_filepath": str(tmp_path / "o.json")}}
    monkeypatch.setattr(sac, "load_config", lambda **kwargs: cfg)
    monkeypatch.setattr(sys, "argv", ["sample_annotations_by_class"])
    with pytest.raises(KeyError, match="gen_text"):
        sac.main()
