#!/usr/bin/env python
"""Interactive unconnected industrial map with vector tiles and dynamic view clustering."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, cast

import geopandas as gpd
import mapbox_vector_tile
import mercantile
import numpy as np
import pandas as pd
from shapely.errors import GEOSException
from shapely.geometry import box, mapping

from ..starter import add_standard_override_arguments, load_config, parse_config_overrides
from ..utils import configure_logging, ensure_output_dir_for_file
from ..geo_utils import repair_geometry


def _load_unconnected_industrial_input(cfg: dict) -> gpd.GeoDataFrame:
    """Load configured industrial unconnected input with no fallback/demo mode."""
    input_path = cfg["paths"]["industrial_unconnected_input"]
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"industrial_unconnected_input not found: {input_path}")

    lower = input_path.lower()
    if lower.endswith(".parquet"):
        gdf = gpd.read_parquet(input_path)
    else:
        gdf = gpd.read_file(input_path)

    if gdf.empty:
        raise ValueError(f"industrial_unconnected_input is empty: {input_path}")
    if "geometry" not in gdf.columns:
        raise KeyError("industrial_unconnected_input must contain a geometry column")

    gdf = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf.set_crs("EPSG:4326")

    repaired = 0
    dropped = 0
    for idx, geom in gdf.geometry.items():
        if geom is None or geom.is_empty:
            gdf.at[idx, "geometry"] = None
            dropped += 1
            continue
        if geom.is_valid:
            continue
        candidate = repair_geometry(geom)
        if candidate is None:
            gdf.at[idx, "geometry"] = None
            dropped += 1
            continue
        gdf.at[idx, "geometry"] = cast(Any, candidate)
        repaired += 1

    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if repaired or dropped:
        print(f"Input geometry sanitization: repaired={repaired}, dropped={dropped}")

    if gdf.empty:
        raise ValueError("No valid geometries remain after input sanitization")

    if "source_id" not in gdf.columns:
        gdf["source_id"] = list(range(len(gdf)))
    if "name" not in gdf.columns:
        gdf["name"] = [f"Industrial Area {idx + 1}" for idx in range(len(gdf))]

    gdf["area_proxy_m2"] = gdf.to_crs(3857).geometry.area.astype(float)
    return gdf[["source_id", "name", "geometry", "area_proxy_m2"]].copy()


def _assign_polygon_min_zoom(gdf: gpd.GeoDataFrame, min_zoom: int, max_zoom: int) -> gpd.GeoDataFrame:
    """Assign each polygon a start zoom in [min_zoom, max_zoom] from area rank."""
    work = gdf.copy()
    area = pd.to_numeric(work["area_proxy_m2"], errors="coerce")
    finite = np.isfinite(area.to_numpy())
    if not finite.any():
        work["tile_min_zoom"] = min_zoom
        return work

    safe_area = area.where(finite, other=0.0).clip(lower=0.0)
    log_area = safe_area.apply(lambda v: math.log1p(float(v)))

    lo = float(log_area.min())
    hi = float(log_area.max())
    if hi <= lo:
        work["tile_min_zoom"] = min_zoom
        return work

    # Large polygons appear earlier (lower zoom); small polygons appear later.
    norm = (log_area - lo) / (hi - lo)
    assigned = max_zoom - norm * float(max_zoom - min_zoom)
    work["tile_min_zoom"] = assigned.fillna(float(min_zoom)).round().clip(lower=min_zoom, upper=max_zoom).astype(int)
    return work


def _build_source_points(gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
    """Create centroid points used for dynamic in-view clustering."""
    centroid_gdf = gdf[["source_id", "name", "geometry", "area_proxy_m2"]].copy()
    centroid_merc = centroid_gdf.to_crs(3857)
    centroid_merc["geometry"] = centroid_merc.geometry.centroid
    centroid_gdf = centroid_merc.to_crs(4326)
    centroid_gdf["lon"] = centroid_gdf.geometry.x
    centroid_gdf["lat"] = centroid_gdf.geometry.y

    points: List[Dict[str, Any]] = []
    for row in centroid_gdf.itertuples(index=False):
        source_id = int(cast(Any, row.source_id))
        points.append(
            {
                "source_id": source_id,
                "name": str(cast(Any, row.name)),
                "lon": float(cast(Any, row.lon)),
                "lat": float(cast(Any, row.lat)),
                "area_proxy_m2": float(cast(Any, row.area_proxy_m2)),
                "polygon_count": 1,
            }
        )
    return points


def _polygon_parts(geom):
    """Return polygonal parts only, flattening collections."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type in {"Polygon", "MultiPolygon"}:
        return [geom]
    if geom.geom_type == "GeometryCollection" or hasattr(geom, "geoms"):
        parts = []
        for part in geom.geoms:
            parts.extend(_polygon_parts(part))
        return parts
    return []


def _build_vector_tiles(gdf: gpd.GeoDataFrame, cfg: dict, tiles_dir: str) -> int:
    """Generate only tiles that contain geometries, with per-polygon start zoom."""
    tile_cfg = cfg["vector_tiles"]
    min_zoom = int(tile_cfg["min_zoom"])
    max_zoom = int(tile_cfg["max_zoom"])
    extent = int(tile_cfg["extent"])
    layer_name = str(tile_cfg["layer_name"])
    simplify_tolerance_deg = float(tile_cfg["simplify_tolerance_deg"])

    if gdf.crs is None:
        work = gdf.set_crs("EPSG:4326")
    else:
        work = gdf.to_crs("EPSG:4326")

    work = _assign_polygon_min_zoom(work, min_zoom=min_zoom, max_zoom=max_zoom)

    tile_features: Dict[tuple[int, int, int], List[Dict[str, Any]]] = {}
    repaired_geometries = 0
    skipped_geometries = 0

    for row in work[["source_id", "name", "geometry", "tile_min_zoom"]].itertuples(index=False):
        source_id = int(cast(Any, row.source_id))
        name = str(cast(Any, row.name))
        start_zoom = int(cast(Any, row.tile_min_zoom))
        original_geom = cast(Any, row.geometry)
        geom = repair_geometry(original_geom)
        if geom is None:
            skipped_geometries += 1
            continue

        if simplify_tolerance_deg > 0:
            geom = geom.simplify(simplify_tolerance_deg, preserve_topology=True)
            geom = repair_geometry(geom)
            if geom is None:
                skipped_geometries += 1
                continue

        if original_geom is not None and not original_geom.is_valid:
            repaired_geometries += 1

        min_x, min_y, max_x, max_y = geom.bounds
        for zoom in range(start_zoom, max_zoom + 1):
            for tile in mercantile.tiles(min_x, min_y, max_x, max_y, zooms=[zoom]):
                bounds = mercantile.bounds(tile)
                tile_poly = box(bounds.west, bounds.south, bounds.east, bounds.north)
                try:
                    if not geom.intersects(tile_poly):
                        continue
                    clipped = geom.intersection(tile_poly)
                except GEOSException:
                    clipped = repair_geometry(geom)
                    if clipped is None:
                        skipped_geometries += 1
                        break
                    try:
                        if not clipped.intersects(tile_poly):
                            continue
                        clipped = clipped.intersection(tile_poly)
                        repaired_geometries += 1
                    except GEOSException:
                        continue

                if clipped.is_empty:
                    continue
                clipped = repair_geometry(clipped)
                if clipped is None:
                    continue

                parts = _polygon_parts(clipped)
                if not parts:
                    continue

                key = (int(tile.z), int(tile.x), int(tile.y))
                bucket = tile_features.setdefault(key, [])
                for part in parts:
                    bucket.append(
                        {
                            "id": source_id,
                            "geometry": mapping(part),
                            "properties": {
                                "source_id": source_id,
                                "name": name,
                            },
                        }
                    )

    if repaired_geometries or skipped_geometries:
        print(f"Tile geometry repair summary: repaired={repaired_geometries}, skipped={skipped_geometries}")

    tile_count = 0
    for (z, x, y), features in tile_features.items():
        bounds = mercantile.bounds(x, y, z)
        payload = mapbox_vector_tile.encode(
            {"name": layer_name, "features": features},
            default_options={
                "quantize_bounds": (bounds.west, bounds.south, bounds.east, bounds.north),
                "extents": extent,
            },
        )
        out_path = Path(tiles_dir) / str(z) / str(x) / f"{y}.pbf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)
        tile_count += 1

    return tile_count


def _zip_vector_tiles(tiles_dir: str, zip_out: str) -> int:
    """Zip generated z/x/y.pbf files and return archived tile count."""
    ensure_output_dir_for_file(zip_out)
    count = 0
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pbf in Path(tiles_dir).rglob("*.pbf"):
            zf.write(pbf, arcname=pbf.relative_to(tiles_dir).as_posix())
            count += 1
    return count


def _build_html(cfg: dict, source_points: List[Dict[str, Any]], vector_tiles_rel_template: str) -> str:
    """Build standalone Leaflet HTML with dynamic view-only aggregation."""
    center_lon = 0.0
    center_lat = 15.0
    min_render_zoom = int(cfg["vector_tiles"]["min_render_zoom"])
    max_native_zoom = int(cfg["vector_tiles"]["max_zoom"])
    source_points_json = json.dumps(source_points)

    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Unconnected Industrial Areas</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
  <style>
    html, body, #map { height: 100%; width: 100%; margin: 0; }
    .legend {
      background: rgba(255,255,255,0.95);
      padding: 10px 12px;
      border-radius: 10px;
      box-shadow: 0 1px 8px rgba(0,0,0,0.22);
      font: 12px/1.3 sans-serif;
      max-width: 320px;
    }
    .legend h4 { margin: 0; font-size: 13px; }
    .legend .legend-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .legend .legend-toggle {
      border: 1px solid rgba(0,0,0,0.25);
      background: rgba(255,255,255,0.9);
      border-radius: 4px;
      width: 24px;
      height: 22px;
      cursor: pointer;
      font-weight: 700;
      line-height: 1;
      padding: 0;
    }
    .legend .legend-title { cursor: pointer; }
    .legend .bubble-wrap { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
    .legend .bubble { border-radius: 50%; border: 1px solid rgba(0,0,0,0.28); background: rgba(78, 179, 211, 0.45); }
    .legend .gradient-bar { height: 12px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.2); margin: 6px 0 4px; }
    .bubble-marker {
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      border: 1px solid rgba(20, 28, 38, 0.85);
      box-shadow: 0 2px 7px rgba(0,0,0,0.24);
      color: #fff;
      font: 700 12px/1 sans-serif;
      text-shadow: 0 1px 2px rgba(0,0,0,0.55);
      user-select: none;
      overflow: hidden;
    }
    .bubble-marker .count { padding: 0 4px; }
    .map-note {
      position: absolute;
      z-index: 999;
      top: 12px;
      left: 58px;
      background: rgba(255,255,255,0.95);
      border-radius: 6px;
      padding: 6px 8px;
      font: 12px sans-serif;
      box-shadow: 0 1px 7px rgba(0,0,0,0.2);
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="map-note">Polygons render only between zoom __MIN_RENDER_ZOOM__ and __MAX_NATIVE_ZOOM__. Bubble clusters and legend are computed only from in-view polygons.</div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <script src="https://unpkg.com/leaflet.vectorgrid@1.3.0/dist/Leaflet.VectorGrid.bundled.js" crossorigin=""></script>
  <script>
    const sourcePoints = __SOURCE_POINTS_JSON__;
    const vectorTileTemplate = __VECTOR_TILE_TEMPLATE_JSON__;
    const vectorTileLayerName = __VECTOR_TILE_LAYER_JSON__;
    const minRenderZoom = __MIN_RENDER_ZOOM__;
    const maxNativeZoom = __MAX_NATIVE_ZOOM__;

    const map = L.map('map', { zoomControl: true, minZoom: 0, maxZoom: 22 }).setView([__CENTER_LAT__, __CENTER_LON__], __INITIAL_ZOOM__);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 22,
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    const polygonLayer = L.vectorGrid.protobuf(vectorTileTemplate, {
      vectorTileLayerStyles: {
        [vectorTileLayerName]: {
          color: '#2f3d4a',
          weight: 1.8,
          fillColor: '#2f3d4a',
          fillOpacity: 0.0,
          opacity: 1.0
        }
      },
      maxNativeZoom: maxNativeZoom,
      maxZoom: 22,
      interactive: false,
    });

    const circlesLayer = L.layerGroup().addTo(map);
    let legendExpanded = true;

    function clamp01(v) { return Math.max(0, Math.min(1, v)); }
    function colorForCount(count, minCount, maxCount) {
      const t = (maxCount > minCount) ? clamp01((count - minCount) / (maxCount - minCount)) : 0.5;
      const hue = 215 - 205 * t;
      const light = 78 - 30 * t;
      return `hsl(${hue.toFixed(1)}, 78%, ${light.toFixed(1)}%)`;
    }
    function computeRadiusPx(areaRounded, minArea, maxArea, radiusScale) {
      const viewportCap = Math.max(18, Math.min(window.innerWidth, window.innerHeight) * 0.11);
      const lo = Math.sqrt(Math.max(minArea, 0));
      const hi = Math.sqrt(Math.max(maxArea, 0));
      const v = Math.sqrt(Math.max(areaRounded, 0));
      const norm = (hi > lo) ? ((v - lo) / (hi - lo)) : 0.5;
      const raw = (8 + 18 * norm) * radiusScale;
      return Math.max(8, Math.min(raw, viewportCap));
    }
    function bubbleStyle(radius, color) {
      const size = Math.max(18, radius * 2);
      return [
        `width:${size}px`,
        `height:${size}px`,
        `line-height:${size}px`,
        `background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.94) 0%, ${color} 42%, rgba(18, 24, 34, 0.96) 100%)`
      ].join(';');
    }

    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function() {
      this._div = L.DomUtil.create('div', 'legend');
      L.DomEvent.disableClickPropagation(this._div);
      L.DomEvent.disableScrollPropagation(this._div);
      this._div.innerHTML = `
        <div class="legend-top">
          <h4 class="legend-title">Current View Aggregation</h4>
          <button type="button" class="legend-toggle" title="Toggle legend">−</button>
        </div>
        <div class="legend-body"></div>
      `;

      const toggleLegend = (event) => {
        event.preventDefault();
        event.stopPropagation();
        legendExpanded = !legendExpanded;
        const body = this._div.querySelector('.legend-body');
        const button = this._div.querySelector('.legend-toggle');
        if (body) body.style.display = legendExpanded ? 'block' : 'none';
        if (button) button.textContent = legendExpanded ? '−' : '+';
      };

      const btn = this._div.querySelector('.legend-toggle');
      const title = this._div.querySelector('.legend-title');
      if (btn) btn.addEventListener('click', toggleLegend);
      if (title) title.addEventListener('click', toggleLegend);
      return this._div;
    };
    legend.addTo(map);

    function formatArea(v) {
      if (v >= 1e9) return (v / 1e9).toFixed(2) + ' bn m²';
      if (v >= 1e6) return (v / 1e6).toFixed(2) + ' mn m²';
      if (v >= 1e3) return (v / 1e3).toFixed(1) + 'k m²';
      return v.toFixed(0) + ' m²';
    }

    function updateLegend(minCount, maxCount, minArea, maxArea) {
      const samples = [minArea, (minArea + maxArea) / 2, maxArea];
      const labels = ['Low density', 'Typical view', 'High density'];
      const bubbles = samples.map((a, i) => {
        const r = computeRadiusPx(a, minArea, maxArea, 1.0);
        return `<div class="bubble-wrap"><div class="bubble" style="width:${r*2}px;height:${r*2}px"></div><span>${labels[i]}: ${formatArea(a)}</span></div>`;
      }).join('');
      const grad = `linear-gradient(90deg, ${colorForCount(minCount, minCount, maxCount)}, ${colorForCount((minCount + maxCount)/2, minCount, maxCount)}, ${colorForCount(maxCount, minCount, maxCount)})`;

      const body = legend._div.querySelector('.legend-body');
      const toggleBtn = legend._div.querySelector('.legend-toggle');
      if (body) {
        body.style.display = legendExpanded ? 'block' : 'none';
        body.innerHTML = `
          <div><b>Color:</b> polygon count gradient</div>
          <div class="gradient-bar" style="background: ${grad}"></div>
          <div style="display:flex;justify-content:space-between;gap:8px"><span>Low count</span><span>High count</span></div>
          <hr />
          <div><b>Size:</b> view-based area proxy</div>
          ${bubbles}
        `;
      }
      if (toggleBtn) toggleBtn.textContent = legendExpanded ? '−' : '+';
    }

    function clusterPoints(points, cellPx) {
      const bounds = map.getBounds();
      const bins = new Map();

      for (const p of points) {
        if (!bounds.contains([p.lat, p.lon])) continue;
        const screen = map.latLngToContainerPoint([p.lat, p.lon]);
        const bx = Math.floor(screen.x / cellPx);
        const by = Math.floor(screen.y / cellPx);
        const key = `${bx}|${by}`;
        if (!bins.has(key)) {
          bins.set(key, { lonSum: 0, latSum: 0, areaSum: 0, polygon_count: 0 });
        }
        const bucket = bins.get(key);
        bucket.lonSum += p.lon;
        bucket.latSum += p.lat;
        bucket.areaSum += p.area_proxy_m2;
        bucket.polygon_count += 1;
      }

      return Array.from(bins.values()).map(bucket => ({
        lon: bucket.lonSum / Math.max(1, bucket.polygon_count),
        lat: bucket.latSum / Math.max(1, bucket.polygon_count),
        area_proxy_m2_rounded: bucket.areaSum,
        polygon_count: bucket.polygon_count,
      }));
    }

    function currentCellPx() {
      const minDim = Math.max(320, Math.min(window.innerWidth, window.innerHeight));
      return Math.max(24, Math.min(84, minDim / 9));
    }

    function renderBubble(point, minArea, maxArea, minCount, maxCount) {
      const radius = computeRadiusPx(point.area_proxy_m2_rounded, minArea, maxArea, 1.0);
      const color = colorForCount(point.polygon_count, minCount, maxCount);
      const icon = L.divIcon({
        className: '',
        html: `<div class="bubble-marker" style="${bubbleStyle(radius, color)}"><span class="count">${point.polygon_count}</span></div>`,
        iconSize: [radius * 2, radius * 2],
        iconAnchor: [radius, radius],
      });

      L.marker([point.lat, point.lon], {
        icon: icon,
        interactive: true,
        riseOnHover: true,
      }).bindPopup(
        `<b>${point.polygon_count === 1 ? 'Industrial polygon' : 'Current view cluster'}</b><br/>` +
        `Polygons: ${point.polygon_count}<br/>` +
        `Area proxy (rounded): ${formatArea(point.area_proxy_m2_rounded)}`
      ).addTo(circlesLayer);
    }

    function syncVectorLayerVisibility() {
      const zoom = map.getZoom();
      const shouldRenderPolygons = zoom >= minRenderZoom && zoom <= maxNativeZoom;
      const hasLayer = map.hasLayer(polygonLayer);
      if (shouldRenderPolygons && !hasLayer) {
        polygonLayer.addTo(map);
      } else if (!shouldRenderPolygons && hasLayer) {
        map.removeLayer(polygonLayer);
      }
    }

    function drawForZoom() {
      syncVectorLayerVisibility();
      circlesLayer.clearLayers();

      const displayPoints = clusterPoints(sourcePoints, currentCellPx());
      if (!displayPoints || displayPoints.length === 0) {
        legend._div.innerHTML = '<h4>No in-view data</h4>';
        return;
      }

      const areaVals = displayPoints.map(p => p.area_proxy_m2_rounded);
      const countVals = displayPoints.map(p => p.polygon_count);
      const minArea = Math.min(...areaVals);
      const maxArea = Math.max(...areaVals);
      const minCount = Math.min(...countVals);
      const maxCount = Math.max(...countVals);

      for (const point of displayPoints) {
        renderBubble(point, minArea, maxArea, minCount, maxCount);
      }

      updateLegend(minCount, maxCount, minArea, maxArea);
    }

    map.on('zoomend', drawForZoom);
    map.on('moveend', drawForZoom);
    drawForZoom();
  </script>
</body>
</html>
"""

    return (
        html.replace("__SOURCE_POINTS_JSON__", source_points_json)
        .replace("__VECTOR_TILE_TEMPLATE_JSON__", json.dumps(vector_tiles_rel_template))
        .replace("__VECTOR_TILE_LAYER_JSON__", json.dumps(str(cfg["vector_tiles"]["layer_name"])))
        .replace("__MIN_RENDER_ZOOM__", str(min_render_zoom))
        .replace("__MAX_NATIVE_ZOOM__", str(max_native_zoom))
        .replace("__CENTER_LAT__", str(center_lat))
        .replace("__CENTER_LON__", str(center_lon))
        .replace("__INITIAL_ZOOM__", str(int(cfg["initial_zoom"])))
    )


def _write_prepared_output(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """Write prepared features to configured path using extension-driven format."""
    ensure_output_dir_for_file(out_path)
    lower = out_path.lower()
    if lower.endswith(".parquet"):
        gdf.to_parquet(out_path, index=False)
        return
    if lower.endswith(".gpkg"):
        gdf.to_file(out_path, driver="GPKG", index=False)
        return
    gdf.to_file(out_path, index=False)


def parse_args():
    """Parse the standardized named config-override flags."""
    parser = argparse.ArgumentParser(description="Run interactive_unconnected_industrial_map.")
    add_standard_override_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Load industrial unconnected input and build vector-tile interactive map."""
    overrides = parse_config_overrides(args=parse_args())
    cfg = load_config(script_name="interactive_unconnected_industrial_map", **overrides)

    if not bool(cfg["vector_tiles"]["enabled"]):
        raise ValueError("interactive_unconnected_industrial_map requires vector_tiles.enabled=true")

    gdf = _load_unconnected_industrial_input(cfg)
    source_points = _build_source_points(gdf)

    prepared_out = cfg["paths"]["industrial_unconnected_output"]
    html_out = cfg["paths"]["industrial_unconnected_html_filepath"]
    tiles_dir = cfg["paths"]["industrial_unconnected_tiles_dir"]
    tiles_zip_out = cfg["paths"]["industrial_unconnected_tiles_zip_filepath"]

    ensure_output_dir_for_file(html_out)

    if os.path.isdir(tiles_dir):
        shutil.rmtree(tiles_dir)
    os.makedirs(tiles_dir, exist_ok=True)

    _write_prepared_output(gdf[["source_id", "name", "geometry", "area_proxy_m2"]], prepared_out)

    tile_count = _build_vector_tiles(gdf, cfg, tiles_dir)

    archived_count = 0
    if bool(cfg["vector_tiles"]["zip_tiles"]):
        archived_count = _zip_vector_tiles(tiles_dir, tiles_zip_out)

    vector_tiles_abs_template = os.path.join(tiles_dir, "{z}", "{x}", "{y}.pbf")
    vector_tiles_rel_template = os.path.relpath(vector_tiles_abs_template, os.path.dirname(html_out)).replace("\\", "/")
    if not vector_tiles_rel_template.startswith("."):
        vector_tiles_rel_template = f"./{vector_tiles_rel_template}"

    html = _build_html(cfg, source_points, vector_tiles_rel_template)
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Prepared output written: {prepared_out}")
    print(f"Vector tiles written: {tile_count} (.pbf files) in {tiles_dir}")
    if bool(cfg["vector_tiles"]["zip_tiles"]):
        print(f"Vector tile archive written: {tiles_zip_out} ({archived_count} tiles)")
    print(f"HTML map written: {html_out}")


if __name__ == "__main__":
    configure_logging()
    main()
