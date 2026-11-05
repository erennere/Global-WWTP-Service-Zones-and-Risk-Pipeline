from pathlib import Path

import pytest

from src import pipelines, starter


pytestmark = pytest.mark.integration


def test_load_config_and_output_path_builders_stay_aligned(write_test_config):
    config_path = write_test_config(
        {
            "combine_watersheds": {
                "level": "4",
            },
            "correct_locations_w_OSM": {
                "version": "3",
            },
            "create_voronoi": {
                "buffer": 1800,
                "weight_method": "linear",
                "weight_func": "mult",
                "dynamic_buffering": False,
            },
        }
    )

    voronoi_cfg = starter.load_config(script_name="create_voronoi", config=str(config_path))
    raster_cfg = starter.load_config(script_name="create_rasters", config=str(config_path))

    output_paths = pipelines.create_output_paths(voronoi_cfg)
    pop_output_paths = pipelines.create_pop_output_paths(raster_cfg)

    assert Path(output_paths["voronoi"]["0"]).parent == Path(voronoi_cfg["paths"]["voronoi_dir"])
    assert Path(output_paths["voronoi"]["0"]).name == "appr_0_v3_lvl4_bf1800_li_mult.gpkg"
    assert Path(pop_output_paths["voronoi"]["0"]).parent == Path(raster_cfg["paths"]["pop_output_dir"])
    assert Path(pop_output_paths["voronoi"]["0"]).name == "pop_added_appr_0_v3_lvl4_bf1800_li_mult.gpkg"