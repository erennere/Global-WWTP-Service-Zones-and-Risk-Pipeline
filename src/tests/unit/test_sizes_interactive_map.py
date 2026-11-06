"""Interactive sizes map: GeoJSON with the right population column and an HTML that fetches it."""
import json

import geopandas as gpd
import pytest
from shapely.geometry import box

from src.figures_scripts import sizes_interactive_map as sim

pytestmark = pytest.mark.unit


def test_population_column_prefers_the_latest_zonal_sum():
    gdf = gpd.GeoDataFrame({"2020_zonal_sum": [1], "2024_zonal_sum": [2], "POP_SERVED": [3]}, geometry=[box(0, 0, 1, 1)])
    assert sim._resolve_population_column(gdf) == "2024_zonal_sum"
    assert sim._resolve_population_column(gdf.drop(columns=["2020_zonal_sum", "2024_zonal_sum"])) == "POP_SERVED"
    assert sim._resolve_population_column(gdf[["geometry"]]) is None


def test_geojson_and_html(tmp_path, monkeypatch):
    src = tmp_path / "pop.gpkg"
    gpd.GeoDataFrame({"total_area": [10.0, 20.0], "round_area": [5.0, 0.0], "2024_zonal_sum": [100.0, None]},
                     geometry=[box(8, 50, 8.1, 50.1), box(8.2, 50, 8.3, 50.1)], crs=4326).to_crs(3035).to_file(src)
    monkeypatch.setattr(sim, "create_pop_output_paths", lambda cfg: {"voronoi": {"1": str(src)}})
    template = tmp_path / "template.html"
    template.write_text('<script>fetch("old.geojson").then(r => r.json())</script>')
    cfg = {"figures": {"approach": 1}, "paths": {
        "sizes_interactive_geojson_filepath": str(tmp_path / "out" / "sizes.geojson"),
        "sizes_interactive_template_filepath": str(template),
        "sizes_interactive_html_filepath": str(tmp_path / "out" / "map" / "sizes.html")}}
    out = sim._build_geojson(cfg)
    features = json.load(open(out))["features"]
    assert [f["properties"]["population"] for f in features] == [100.0, 0.0]
    assert features[0]["properties"]["bbox_minx"] == pytest.approx(8.0)  # reprojected to EPSG:4326
    html = open(sim._build_html(cfg, out)).read()
    assert 'fetch("../sizes.geojson")' in html and "old.geojson" not in html
