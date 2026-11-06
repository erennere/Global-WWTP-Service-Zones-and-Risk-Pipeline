from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box


SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _isolate_argv(monkeypatch):
    """Hide pytest's own argv from entry points that call ``parse_args()``.

    Every ``main()`` now builds its own argparse parser, so without this the
    parser would see pytest's command line and exit. Tests that care about
    specific flags set ``sys.argv`` themselves after this fixture runs.
    """
    monkeypatch.setattr(sys, "argv", ["pipeline-entry-point"])


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict):
            current = target[key]
            if not isinstance(current, dict):
                target[key] = deepcopy(value)
                continue
            _deep_update(current, value)
            continue
        target[key] = value


def _prepare_config_dict(base_config):
    cfg = deepcopy(base_config)
    cfg["correct_locations_w_OSM"]["paths"]["data_dir"] = "./data"

    for section in cfg.values():
        if not isinstance(section, dict) or "paths" not in section:
            continue
        paths = section["paths"]
        if paths is None or not isinstance(paths, dict):
            continue
        for key, value in list(paths.items()):
            if value is None or not isinstance(value, str) or "://" in value:
                continue
            if Path(value).is_absolute():
                suffix = Path(value).suffix
                paths[key] = f"./external/{key}{suffix}"

    return cfg


def _write_config_file(config_root, cfg):
    (config_root / "data").mkdir(parents=True, exist_ok=True)
    (config_root / "external").mkdir(parents=True, exist_ok=True)
    config_path = config_root / "config.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    return config_path


@pytest.fixture(scope="session")
def base_config():
    config_path = SRC_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def tiny_points_gdf():
    return gpd.GeoDataFrame(
        {
            "WASTE_ID": [1, 2, 3, 4, 5, 6],
            "HYBAS_ID": [101, 101, 101, 202, 202, 202],
            "ISO_2": ["DE", "DE", "DE", "FR", "FR", "FR"],
            "pop_served": [120, 250, 800, 100, 400, 900],
            "wwtp_area_rect": ["[10 5]", "[1.5]", "[0]", "[4 2]", "[6]", "[0]"],
            "diameters": ["[2, 4]", "[3]", "[]", "[4]", "[2, 2]", "[1]"],
            "num_detection_rect": [3, 0, 1, 2, 1, 0],
            "num_detection_circle": [1, 0, 0, 1, 2, 1],
            "basin_area": [1500.0, 1500.0, 1500.0, 2400.0, 2400.0, 2400.0],
            "mean_2_nnd": [3500.0, 2500.0, 4000.0, 7000.0, 6500.0, 7500.0],
            "category_number": ["1", "mix-industrial", "2", "3", "mix", "4"],
            "geometry": [
                Point(-0.02, -0.01),
                Point(0.00, 0.00),
                Point(0.02, 0.01),
                Point(1.00, -0.02),
                Point(1.02, 0.00),
                Point(1.04, 0.02),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_sites_gdf(tiny_points_gdf):
    return tiny_points_gdf.copy().reset_index(drop=True)


@pytest.fixture
def tiny_watershed_gdf():
    return gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 202],
            "country": ["DE", "FR"],
            "geometry": [
                box(-0.08, -0.08, 0.08, 0.08),
                box(0.92, -0.08, 1.10, 0.08),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


@pytest.fixture
def tiny_country_gdf(tiny_watershed_gdf):
    return gpd.GeoDataFrame(
        {
            "country": tiny_watershed_gdf["country"].tolist(),
            "geometry": tiny_watershed_gdf["geometry"].tolist(),
        },
        geometry="geometry",
        crs=tiny_watershed_gdf.crs,
    )


@pytest.fixture
def tiny_population_array():
    return {
        "array": np.array(
            [
                [0.0, 2.0, 4.0, 0.0],
                [1.0, 3.0, 5.0, 1.0],
                [0.0, 2.0, 6.0, 0.0],
                [0.0, 1.0, 2.0, 0.0],
            ],
            dtype="float32",
        ),
        "transform": from_origin(-0.10, 0.10, 0.05, 0.05),
        "crs": "EPSG:4326",
    }


@pytest.fixture
def mock_rivershed_gdf():
    return gpd.GeoDataFrame(
        {
            "river_id": [1, 2],
            "HYBAS_ID": [101, 202],
            "discharge_cms": [0.0, 5.0],
            "geometry": [
                LineString([(-0.08, 0.0), (0.08, 0.0)]),
                LineString([(0.92, 0.0), (1.10, 0.0)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


@pytest.fixture(scope="session")
def _mock_cfg_session(base_config, tmp_path_factory):
    from wwtp.starter import load_config

    config_root = tmp_path_factory.mktemp("mock_cfg")
    cfg_dict = _prepare_config_dict(base_config)
    config_path = _write_config_file(config_root, cfg_dict)
    cfg = load_config(script_name="create_voronoi", config=str(config_path))

    for path_value in cfg["paths"].values():
        if not isinstance(path_value, str) or "://" in path_value:
            continue
        path_obj = Path(path_value)
        if path_obj.suffix:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        else:
            path_obj.mkdir(parents=True, exist_ok=True)

    return cfg


@pytest.fixture
def mock_cfg(_mock_cfg_session):
    return deepcopy(_mock_cfg_session)


@pytest.fixture
def write_test_config(tmp_path, base_config):
    def _write(updates=None):
        cfg = _prepare_config_dict(base_config)

        if updates:
            _deep_update(cfg, updates)

        return _write_config_file(tmp_path, cfg)

    return _write

@pytest.fixture(autouse=True)
def _no_run_manifests(request, monkeypatch):
    """Stage mains called inside tests must not write manifests into ../data.

    Tests marked ``run_manifest`` get the real writer (and point it at tmp_path).
    """
    if request.node.get_closest_marker("run_manifest"):
        return
    from wwtp import provenance

    monkeypatch.setattr(provenance, "write_run_manifest", lambda cfg, stage, argv=None: None)


@pytest.fixture(autouse=True)
def _no_config_overlays(monkeypatch):
    """Tests see the repository config only: no overlay files from the environment or the machine."""
    from wwtp import starter

    monkeypatch.delenv(starter.OVERLAYS_ENV_VAR, raising=False)
    monkeypatch.setattr(starter, "LOCAL_OVERLAY_FILENAME", "__tests_ignore_config.local.yaml")


@pytest.fixture(autouse=True)
def _direct_writes_for_fake_writers(request, monkeypatch):
    """Unit tests often replace ``to_file``/``to_parquet`` with fakes that never create a file.

    ``utils.atomic_output`` would then fail to rename the missing temporary file, so
    unit tests get a pass-through that yields the target path itself. The rename is
    tested in test_utils (marker ``real_atomic_output``) and by the mock-world chain,
    which runs every stage in a subprocess with the real helper.
    """
    if not request.node.get_closest_marker("unit") or request.node.get_closest_marker("real_atomic_output"):
        return
    from contextlib import contextmanager

    from wwtp import utils

    @contextmanager
    def passthrough(filepath):
        utils.ensure_output_dir_for_file(filepath)
        yield filepath

    real = utils.atomic_output
    for name, module in list(sys.modules.items()):  # modules that imported it by name, then utils for later imports
        if name.startswith("wwtp.") and name != "wwtp.utils" and getattr(module, "atomic_output", None) is real:
            monkeypatch.setattr(module, "atomic_output", passthrough)
    monkeypatch.setattr(utils, "atomic_output", passthrough)
