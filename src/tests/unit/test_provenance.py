"""Run manifests record config, code version, packages and input fingerprints."""
import hashlib

import pytest
import yaml

from src import provenance

pytestmark = [pytest.mark.unit, pytest.mark.run_manifest]


def test_manifest_names_the_run(tmp_path, monkeypatch):
    small = tmp_path / "plants.csv"
    small.write_text("a,b\n1,2\n")
    monkeypatch.setattr(provenance, "load_config", lambda script_name: {
        "hash_max_mb": 1, "paths": {"manifest_dir": str(tmp_path / "runs")}})
    cfg = {"level": "6", "paths": {"plants": str(small), "missing": str(tmp_path / "nope.gpkg"),
                                   "out_dir": str(tmp_path)}, "distance_fn": len}
    out = provenance.write_run_manifest(cfg, "create_voronoi", argv=["create_voronoi", "--approach", "1"])
    m = yaml.safe_load(open(out))
    assert m["stage"] == "create_voronoi" and m["command"][-1] == "1"
    assert m["config"]["level"] == "6" and m["config"]["distance_fn"].startswith("<builtins.len")
    assert set(m["inputs_found_at_start"]) == {"plants"}  # existing files only
    assert m["inputs_found_at_start"]["plants"]["sha256"] == hashlib.sha256(small.read_bytes()).hexdigest()
    assert "numpy" in m["packages"] and "commit" in m["code"]


def test_large_inputs_get_size_and_time_only(tmp_path):
    big = tmp_path / "raster.tif"
    big.write_bytes(b"0" * 2048)
    fp = provenance.file_fingerprint(str(big), hash_max_mb=0.001)
    assert fp["bytes"] == 2048 and "sha256" not in fp
