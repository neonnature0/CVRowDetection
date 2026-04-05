#!/usr/bin/env python3
"""
Standalone aerial annotation tool for vineyard feature training data.

Opens a browser-based map (Leaflet + ESRI/LINZ tiles) where you can
draw block boundaries and row lines. Annotations save as local GeoJSON.

Usage:
    python map_annotator.py                                 # Open at default (Marlborough NZ)
    python map_annotator.py --lat -41.52 --lng 173.95       # Specific location
    python map_annotator.py --load dataset/standalone/x.geojson  # Resume session
    python map_annotator.py --port 8765                     # Custom port
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_DIR = Path("dataset")
STANDALONE_DIR = DATASET_DIR / "standalone"


def _find_python() -> str:
    """Find the venv Python executable."""
    for candidate in [
        Path(__file__).parent / "venv" / "Scripts" / "python.exe",
        Path(__file__).parent / "venv" / "bin" / "python",
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _build_html(state: dict) -> str:
    """Build the complete HTML page with embedded state."""
    ann = state["annotations"]
    blocks_json = json.dumps({"blocks": ann.get("blocks", []), "rows": ann.get("rows", [])})
    linz_key = os.environ.get("LINZ_API_KEY", "")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vineyard Map Annotator</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; display: flex; height: 100vh; }}
  #sidebar {{
    width: 280px; background: #1a1a2e; color: #e0e0e0; padding: 16px;
    display: flex; flex-direction: column; gap: 12px; overflow-y: auto; flex-shrink: 0;
  }}
  #sidebar h2 {{ color: #fff; font-size: 16px; margin-bottom: 4px; }}
  #sidebar h3 {{ color: #aaa; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px; }}
  #map {{ flex: 1; }}
  .btn {{
    display: block; width: 100%; padding: 10px; border: none; border-radius: 6px;
    font-size: 13px; font-weight: 600; cursor: pointer; text-align: center;
  }}
  .btn-primary {{ background: #4a9eff; color: #fff; }}
  .btn-primary:hover {{ background: #3a8eef; }}
  .btn-success {{ background: #2ecc71; color: #fff; }}
  .btn-success:hover {{ background: #27ae60; }}
  .btn-warning {{ background: #f39c12; color: #fff; }}
  .btn-warning:hover {{ background: #e67e22; }}
  .btn-danger {{ background: #e74c3c; color: #fff; }}
  .btn-danger:hover {{ background: #c0392b; }}
  .btn-secondary {{ background: #444; color: #ddd; }}
  .btn-secondary:hover {{ background: #555; }}
  #status {{ font-size: 11px; color: #888; line-height: 1.5; }}
  #block-list {{ font-size: 12px; max-height: 200px; overflow-y: auto; }}
  .block-item {{
    padding: 6px 8px; margin: 2px 0; background: #2a2a4a; border-radius: 4px;
    cursor: pointer; display: flex; justify-content: space-between; align-items: center;
  }}
  .block-item:hover {{ background: #3a3a5a; }}
  .del-btn {{
    background: none; border: none; color: #e74c3c; cursor: pointer; font-size: 14px; padding: 0 4px;
  }}
  #log {{
    font-family: monospace; font-size: 10px; color: #888; background: #111;
    padding: 8px; border-radius: 4px; min-height: 50px; max-height: 100px;
    overflow-y: auto; white-space: pre-wrap; margin-top: auto;
  }}
  .coord-input {{ display: flex; gap: 4px; align-items: center; }}
  .coord-input input {{
    width: 100px; padding: 6px; border: 1px solid #444; border-radius: 4px;
    background: #2a2a4a; color: #e0e0e0; font-size: 12px;
  }}
  .coord-input .btn {{ width: auto; padding: 6px 12px; }}
</style>
</head>
<body>
<div id="sidebar">
  <h2>Vineyard Annotator</h2>

  <div class="coord-input">
    <input type="text" id="lat-input" placeholder="Lat" value="{state['lat']}">
    <input type="text" id="lng-input" placeholder="Lng" value="{state['lng']}">
    <button class="btn btn-secondary" onclick="goToCoords()">Go</button>
  </div>

  <h3>Drawing</h3>
  <button class="btn btn-primary" onclick="startDrawBlock()">Draw Block (manual)</button>
  <button class="btn btn-secondary" onclick="startDrawRow()">Draw Row</button>

  <h3>AI Detect (click on block)</h3>
  <button class="btn btn-danger" id="sam-btn" onclick="toggleSamMode()">SAM Click Mode: OFF</button>
  <div style="font-size:11px; color:#888; margin-top:4px;">Click inside a vineyard block to auto-detect its boundary</div>

  <h3>Blocks</h3>
  <div id="block-list"></div>

  <h3>Actions</h3>
  <button class="btn btn-success" onclick="saveAnnotations()">Save</button>
  <button class="btn btn-warning" onclick="runGenerateData()">Generate Training Data</button>
  <button class="btn btn-warning" onclick="runTrain()">Train Model</button>

  <div id="status">
    Zoom: <span id="zoom-display">{state['zoom']}</span> |
    Blocks: <span id="block-count">{len(ann.get('blocks', []))}</span> |
    Rows: <span id="row-count">{len(ann.get('rows', []))}</span>
  </div>

  <div id="log">Ready.</div>
</div>
<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script>
// --- State ---
var annotations = {blocks_json};
var savePath = {json.dumps(state['save_path'])};
var nextBlockId = {state['next_block_id']};
var nextRowId = {state['next_row_id']};
var blockLayers = {{}};
var rowLayers = {{}};

// --- Map ---
var map = L.map('map').setView([{state['lat']}, {state['lng']}], {state['zoom']});

L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  maxZoom: 20, attribution: 'Esri'
}}).addTo(map);

var linzKey = {json.dumps(linz_key)};
if (linzKey) {{
  L.tileLayer('https://basemaps.linz.govt.nz/v1/tiles/aerial/WebMercatorQuad/{{z}}/{{x}}/{{y}}.webp?api=' + linzKey, {{
    maxZoom: 22, attribution: 'LINZ'
  }}).addTo(map);
}}

var drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

var drawControl = new L.Control.Draw({{
  draw: {{ polygon: false, polyline: false, rectangle: false, circle: false, marker: false, circlemarker: false }},
  edit: {{ featureGroup: drawnItems, edit: true, remove: true }}
}});
map.addControl(drawControl);

var activeDrawer = null;

function startDrawBlock() {{
  if (activeDrawer) activeDrawer.disable();
  activeDrawer = new L.Draw.Polygon(map, {{
    shapeOptions: {{ color: '#00ffff', weight: 2, fillOpacity: 0.15 }},
    allowIntersection: false
  }});
  activeDrawer.enable();
  logMsg('Click to place vertices. Click first point to close.');
}}

function startDrawRow() {{
  if (activeDrawer) activeDrawer.disable();
  activeDrawer = new L.Draw.Polyline(map, {{
    shapeOptions: {{ color: '#ff00ff', weight: 2 }}
  }});
  activeDrawer.enable();
  logMsg('Click to place row points. Double-click to finish.');
}}

map.on(L.Draw.Event.CREATED, function(e) {{
  var layer = e.layer;
  if (e.layerType === 'polygon') {{
    var latlngs = layer.getLatLngs()[0];
    var coords = latlngs.map(function(ll) {{ return [ll.lng, ll.lat]; }});
    var block = {{ id: nextBlockId++, label: 'Block ' + (nextBlockId - 1), polygon_lnglat: coords }};
    annotations.blocks.push(block);
    addBlockToMap(block);
    updateBlockList();
    logMsg('Added ' + block.label);
  }} else if (e.layerType === 'polyline') {{
    var latlngs = layer.getLatLngs();
    var coords = latlngs.map(function(ll) {{ return [ll.lng, ll.lat]; }});
    var row = {{ id: nextRowId++, block_id: 0, control_points_lnglat: coords }};
    annotations.rows.push(row);
    addRowToMap(row);
    logMsg('Added row ' + row.id);
  }}
  updateCounts();
  activeDrawer = null;
}});

map.on(L.Draw.Event.EDITED, function(e) {{
  e.layers.eachLayer(function(layer) {{
    for (var id in blockLayers) {{
      if (blockLayers[id] === layer) {{
        var latlngs = layer.getLatLngs()[0];
        var block = annotations.blocks.find(function(b) {{ return b.id == id; }});
        if (block) block.polygon_lnglat = latlngs.map(function(ll) {{ return [ll.lng, ll.lat]; }});
        return;
      }}
    }}
    for (var id in rowLayers) {{
      if (rowLayers[id] === layer) {{
        var latlngs = layer.getLatLngs();
        var row = annotations.rows.find(function(r) {{ return r.id == id; }});
        if (row) row.control_points_lnglat = latlngs.map(function(ll) {{ return [ll.lng, ll.lat]; }});
        return;
      }}
    }}
  }});
  logMsg('Annotations edited');
}});

map.on(L.Draw.Event.DELETED, function(e) {{
  e.layers.eachLayer(function(layer) {{
    for (var id in blockLayers) {{
      if (blockLayers[id] === layer) {{
        annotations.blocks = annotations.blocks.filter(function(b) {{ return b.id != id; }});
        delete blockLayers[id];
        updateBlockList();
        return;
      }}
    }}
    for (var id in rowLayers) {{
      if (rowLayers[id] === layer) {{
        annotations.rows = annotations.rows.filter(function(r) {{ return r.id != id; }});
        delete rowLayers[id];
        return;
      }}
    }}
  }});
  updateCounts();
  logMsg('Features deleted');
}});

function addBlockToMap(block) {{
  var latlngs = block.polygon_lnglat.map(function(c) {{ return [c[1], c[0]]; }});
  var layer = L.polygon(latlngs, {{ color: '#00ffff', weight: 2, fillOpacity: 0.15 }});
  layer.bindTooltip(block.label, {{ permanent: false, direction: 'center' }});
  drawnItems.addLayer(layer);
  blockLayers[block.id] = layer;
}}

function addRowToMap(row) {{
  var latlngs = row.control_points_lnglat.map(function(c) {{ return [c[1], c[0]]; }});
  var layer = L.polyline(latlngs, {{ color: '#ff00ff', weight: 2 }});
  drawnItems.addLayer(layer);
  rowLayers[row.id] = layer;
}}

function updateBlockList() {{
  var list = document.getElementById('block-list');
  while (list.firstChild) list.removeChild(list.firstChild);
  annotations.blocks.forEach(function(block) {{
    var item = document.createElement('div');
    item.className = 'block-item';
    var span = document.createElement('span');
    span.textContent = block.label;
    span.onclick = function() {{ zoomToBlock(block.id); }};
    var btn = document.createElement('button');
    btn.className = 'del-btn';
    btn.textContent = 'x';
    btn.onclick = function() {{ deleteBlock(block.id); }};
    item.appendChild(span);
    item.appendChild(btn);
    list.appendChild(item);
  }});
}}

function deleteBlock(id) {{
  if (blockLayers[id]) {{ drawnItems.removeLayer(blockLayers[id]); delete blockLayers[id]; }}
  annotations.blocks = annotations.blocks.filter(function(b) {{ return b.id !== id; }});
  updateBlockList();
  updateCounts();
  logMsg('Deleted block ' + id);
}}

function zoomToBlock(id) {{
  if (blockLayers[id]) map.fitBounds(blockLayers[id].getBounds().pad(0.2));
}}

function updateCounts() {{
  document.getElementById('block-count').textContent = annotations.blocks.length;
  document.getElementById('row-count').textContent = annotations.rows.length;
}}

function goToCoords() {{
  var lat = parseFloat(document.getElementById('lat-input').value);
  var lng = parseFloat(document.getElementById('lng-input').value);
  if (!isNaN(lat) && !isNaN(lng)) map.setView([lat, lng], 17);
}}

function saveAnnotations() {{
  var geojson = annotationsToGeoJSON();
  fetch('/api/save', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ path: savePath, data: geojson }})
  }}).then(function(r) {{ return r.json(); }}).then(function(r) {{
    if (r.ok) logMsg('Saved ' + annotations.blocks.length + ' blocks + ' + annotations.rows.length + ' rows');
    else logMsg('Save failed: ' + (r.error || 'unknown'));
  }}).catch(function(e) {{ logMsg('Save error: ' + e); }});
}}

function annotationsToGeoJSON() {{
  var features = [];
  annotations.blocks.forEach(function(b) {{
    var coords = b.polygon_lnglat.map(function(c) {{ return [c[0], c[1]]; }});
    if (coords.length > 0) {{
      var first = coords[0], last = coords[coords.length - 1];
      if (first[0] !== last[0] || first[1] !== last[1]) coords.push([first[0], first[1]]);
    }}
    features.push({{
      type: 'Feature',
      geometry: {{ type: 'Polygon', coordinates: [coords] }},
      properties: {{ feature_type: 'block', name: b.label, id: b.id, vineyard_name: b.vineyard_name || '', region: b.region || '' }}
    }});
  }});
  annotations.rows.forEach(function(r) {{
    features.push({{
      type: 'Feature',
      geometry: {{ type: 'LineString', coordinates: r.control_points_lnglat.map(function(c) {{ return [c[0], c[1]]; }}) }},
      properties: {{ feature_type: 'row', block_id: r.block_id, id: r.id }}
    }});
  }});
  return {{
    type: 'FeatureCollection',
    metadata: {{ name: savePath, modified_at: new Date().toISOString(), source: 'map_annotator' }},
    features: features
  }};
}}

function runCommand(endpoint, label) {{
  logMsg(label + '...');
  fetch('/api/' + endpoint, {{ method: 'POST' }})
    .then(function(r) {{ return r.json(); }})
    .then(function(r) {{ logMsg(r.ok ? label + ' started. Check terminal.' : label + ': ' + (r.error||'failed')); }})
    .catch(function(e) {{ logMsg(label + ' error: ' + e); }});
}}

function runGenerateData() {{ saveAnnotations(); setTimeout(function() {{ runCommand('generate', 'Generate Data'); }}, 500); }}
function runTrain() {{ runCommand('train', 'Train Model'); }}
var detectRunning = false;
function runDetection() {{
  if (detectRunning) {{ logMsg('Detection already running, please wait...'); return; }}
  var bounds = map.getBounds();
  detectRunning = true;
  logMsg('Detection running... this takes 30-60s. Please wait.');
  fetch('/api/detect', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{
      south: bounds.getSouth(), north: bounds.getNorth(),
      west: bounds.getWest(), east: bounds.getEast(),
      zoom: map.getZoom()
    }})
  }}).then(function(r) {{ return r.json(); }}).then(function(r) {{
    detectRunning = false;
    if (r.ok && r.blocks) {{
      r.blocks.forEach(function(b) {{
        var block = {{ id: nextBlockId++, label: 'Detected ' + (nextBlockId-1), polygon_lnglat: b.polygon_lnglat }};
        annotations.blocks.push(block);
        addBlockToMap(block);
      }});
      updateBlockList(); updateCounts();
      logMsg('Found ' + r.blocks.length + ' blocks!');
    }} else {{
      logMsg('Detection: ' + (r.error || 'no blocks found'));
    }}
  }}).catch(function(e) {{ detectRunning = false; logMsg('Detection error: ' + e); }});
}}

function logMsg(msg) {{
  var el = document.getElementById('log');
  el.textContent = msg + '\\n' + el.textContent.substring(0, 500);
}}

// --- SAM click-to-segment ---
var samMode = false;
var samRunning = false;

function toggleSamMode() {{
  samMode = !samMode;
  var btn = document.getElementById('sam-btn');
  btn.textContent = samMode ? 'SAM Click Mode: ON' : 'SAM Click Mode: OFF';
  btn.style.background = samMode ? '#e74c3c' : '#444';
  if (samMode) {{
    map.getContainer().style.cursor = 'crosshair';
    logMsg('SAM mode ON. Click inside a vineyard block.');
  }} else {{
    map.getContainer().style.cursor = '';
    logMsg('SAM mode OFF.');
  }}
}}

map.on('click', function(e) {{
  if (!samMode || samRunning) return;
  samRunning = true;
  var lat = e.latlng.lat, lng = e.latlng.lng;
  logMsg('SAM segmenting at ' + lat.toFixed(5) + ', ' + lng.toFixed(5) + '... (~5-10s)');

  fetch('/api/sam-segment', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ lat: lat, lng: lng, zoom: map.getZoom() }})
  }}).then(function(r) {{ return r.json(); }}).then(function(r) {{
    samRunning = false;
    if (r.ok && r.polygon_lnglat && r.polygon_lnglat.length >= 3) {{
      var block = {{ id: nextBlockId++, label: 'SAM ' + (nextBlockId-1), polygon_lnglat: r.polygon_lnglat }};
      annotations.blocks.push(block);
      addBlockToMap(block);
      updateBlockList(); updateCounts();
      logMsg('SAM detected block with ' + r.polygon_lnglat.length + ' vertices. Confidence: ' + (r.confidence || 'n/a'));
    }} else {{
      logMsg('SAM: ' + (r.error || 'no segment found at this point'));
    }}
  }}).catch(function(e) {{ samRunning = false; logMsg('SAM error: ' + e); }});
}});

// --- Zoom display ---
map.on('zoomend', function() {{
  document.getElementById('zoom-display').textContent = map.getZoom();
}});

// --- Init ---
annotations.blocks.forEach(addBlockToMap);
annotations.rows.forEach(addRowToMap);
updateBlockList();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class AnnotatorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = _build_html(self.server.state)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""

        if self.path == "/api/save":
            self._handle_save(body)
        elif self.path == "/api/generate":
            self._handle_run([_find_python(), "prepare_block_dataset.py", "--all"])
        elif self.path == "/api/train":
            self._handle_run([_find_python(), "-m", "detection.train_blocks", "--epochs", "50", "--batch-size", "4", "--patience", "15"])
        elif self.path == "/api/detect":
            self._handle_detect(body)
        elif self.path == "/api/sam-segment":
            self._handle_sam_segment(body)
        else:
            self.send_error(404)

    def _handle_save(self, body):
        try:
            payload = json.loads(body)
            path = Path(payload["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload["data"], f, indent=2)
            logger.info("Saved to %s", path)
            self._json({"ok": True})
        except Exception as e:
            logger.error("Save failed: %s", e)
            self._json({"ok": False, "error": str(e)})

    def _handle_detect(self, body):
        """Run block detection on the viewport area."""
        encoder_path = Path(__file__).parent / "block_detection" / "checkpoints" / "encoder.pth"
        head_path = Path(__file__).parent / "block_detection" / "checkpoints" / "block_head.pth"

        if not encoder_path.exists() or not head_path.exists():
            self._json({"ok": False, "error": "No trained model found. Click Train Model first."})
            return

        try:
            params = json.loads(body) if body else {}
            south = params.get("south", -41.54)
            north = params.get("north", -41.52)
            west = params.get("west", 173.94)
            east = params.get("east", 173.96)

            # Build a bbox polygon for tile fetching
            bbox_ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]

            base_dir = Path(__file__).parent
            sys.path.insert(0, str(base_dir))
            from vinerow.acquisition.tile_fetcher import fetch_and_stitch, TILE_SOURCES, auto_select_source, default_zoom
            from vinerow.acquisition.geo_utils import pixel_to_lnglat

            centroid_lng = (west + east) / 2
            source_name = auto_select_source(centroid_lng)
            source = TILE_SOURCES[source_name]
            # Use browser zoom clamped to a reasonable range for detection
            browser_zoom = params.get("zoom", default_zoom(source_name))
            zoom = max(17, min(int(browser_zoom), source.max_zoom))

            # Limit detection area — cap at zoom 18 to avoid fetching thousands of tiles
            zoom = min(zoom, 18)

            logger.info("Detection: bbox [%.4f,%.4f]-[%.4f,%.4f] zoom=%d source=%s", west, south, east, north, zoom, source_name)

            from vinerow.acquisition.geo_utils import tiles_covering_bbox, polygon_bbox as geo_bbox
            bbox = (west, south, east, north)
            tiles_needed = tiles_covering_bbox(bbox, zoom)
            logger.info("Detection: %d tiles needed", len(tiles_needed))

            if len(tiles_needed) > 200:
                self._json({"ok": False, "error": f"Viewport too large ({len(tiles_needed)} tiles). Zoom in closer."})
                return

            image, mask, tile_origin = fetch_and_stitch(source, bbox_ring, zoom, source_name, "output/.tile_cache")
            logger.info("Detection: image %dx%d", image.shape[1], image.shape[0])

            # Run model
            from block_detection.predict_blocks import predict_blocks
            result = predict_blocks(image, str(encoder_path), str(head_path),
                                    tile_origin=tile_origin, zoom=zoom, tile_size=source.tile_size)

            # Convert to response format
            blocks_out = []
            for block in result.blocks:
                if block.polygon_lnglat:
                    blocks_out.append({"polygon_lnglat": list(block.polygon_lnglat), "confidence": block.confidence})

            logger.info("Detection found %d blocks in %.1fs", len(blocks_out), result.processing_time_s)
            self._json({"ok": True, "blocks": blocks_out})

        except Exception as e:
            logger.error("Detection failed: %s", e, exc_info=True)
            self._json({"ok": False, "error": str(e)})

    _sam_model = None  # lazy-loaded MobileSAM

    def _handle_sam_segment(self, body):
        """Run MobileSAM on a small tile area around the clicked point."""
        try:
            params = json.loads(body) if body else {}
            click_lat = params["lat"]
            click_lng = params["lng"]
            browser_zoom = params.get("zoom", 18)

            base_dir = Path(__file__).parent
            sys.path.insert(0, str(base_dir))
            from vinerow.acquisition.tile_fetcher import fetch_and_stitch, TILE_SOURCES, auto_select_source
            from vinerow.acquisition.geo_utils import pixel_to_lnglat, _lng_lat_to_tile_float

            source_name = auto_select_source(click_lng)
            source = TILE_SOURCES[source_name]
            # Use zoom 19 for good resolution — SAM needs to see the block clearly
            zoom = min(19, source.max_zoom)
            ts = source.tile_size

            # Fetch a small area around the click point (~500m radius)
            # At zoom 19, each tile covers ~76m (at lat -41). We want ~6x6 tiles.
            pad_deg = 0.003  # ~300m in each direction
            bbox_ring = [
                [click_lng - pad_deg, click_lat - pad_deg],
                [click_lng + pad_deg, click_lat - pad_deg],
                [click_lng + pad_deg, click_lat + pad_deg],
                [click_lng - pad_deg, click_lat + pad_deg],
                [click_lng - pad_deg, click_lat - pad_deg],
            ]

            logger.info("SAM: fetching tiles around (%.5f, %.5f) at zoom %d", click_lat, click_lng, zoom)
            image, mask, tile_origin = fetch_and_stitch(source, bbox_ring, zoom, source_name, "output/.tile_cache")
            logger.info("SAM: image %dx%d", image.shape[1], image.shape[0])

            # Convert click point to pixel coordinates in the stitched image
            tx_f, ty_f = _lng_lat_to_tile_float(click_lng, click_lat, zoom)
            click_px = (tx_f - tile_origin[0]) * ts
            click_py = (ty_f - tile_origin[1]) * ts
            logger.info("SAM: click pixel (%.0f, %.0f)", click_px, click_py)

            # Save temp image for SAM
            import cv2
            temp_path = str(base_dir / "output" / "_sam_input.png")
            Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(temp_path, image)

            # Lazy-load MobileSAM
            if AnnotatorHandler._sam_model is None:
                logger.info("SAM: loading MobileSAM model (first time)...")
                from ultralytics import SAM
                AnnotatorHandler._sam_model = SAM("mobile_sam.pt")
                logger.info("SAM: model loaded")

            # Run SAM with point prompt
            import time
            t0 = time.time()
            results = AnnotatorHandler._sam_model.predict(
                temp_path,
                points=[[click_px, click_py]],
                labels=[1],  # 1 = foreground point
            )
            elapsed = time.time() - t0
            logger.info("SAM: inference took %.1fs", elapsed)

            r = results[0]
            if r.masks is None or len(r.masks.data) == 0:
                self._json({"ok": False, "error": "SAM found no segment at this point"})
                return

            # Take the mask, convert to polygon
            import numpy as np
            mask_np = r.masks.data[0].cpu().numpy().astype(np.uint8)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self._json({"ok": False, "error": "No contours found in SAM mask"})
                return

            # Take largest contour
            largest = max(contours, key=cv2.contourArea)
            if len(largest) < 4:
                self._json({"ok": False, "error": "Contour too small"})
                return

            # Simplify polygon
            epsilon = 0.001 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            if len(approx) < 4:
                approx = largest

            # Convert pixel polygon to geographic coordinates
            polygon_lnglat = []
            for pt in approx:
                px, py = float(pt[0][0]), float(pt[0][1])
                lng, lat = pixel_to_lnglat(px, py, tile_origin, zoom, ts)
                polygon_lnglat.append([lng, lat])

            # Confidence from mask area
            mask_area = float(mask_np.sum())
            image_area = float(mask_np.shape[0] * mask_np.shape[1])
            confidence = min(mask_area / image_area, 1.0)

            logger.info("SAM: found polygon with %d vertices, area=%.0f px (%.1f%% of image)", len(polygon_lnglat), mask_area, confidence * 100)
            self._json({"ok": True, "polygon_lnglat": polygon_lnglat, "confidence": round(confidence, 3)})

        except Exception as e:
            logger.error("SAM segment failed: %s", e, exc_info=True)
            self._json({"ok": False, "error": str(e)})

    def _handle_run(self, cmd):
        try:
            subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
            self._json({"ok": True})
        except Exception as e:
            self._json({"ok": False, "error": str(e)})

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, fmt, *args):
        pass  # suppress noisy HTTP logs


# ---------------------------------------------------------------------------
# Load annotations
# ---------------------------------------------------------------------------

def load_geojson_annotations(path: Path) -> tuple[dict, int, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    blocks, rows = [], []
    max_bid, max_rid = 0, 0
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        if props.get("feature_type") == "block" and geom.get("type") == "Polygon":
            ring = geom["coordinates"][0]
            if len(ring) > 1 and ring[0] == ring[-1]:
                ring = ring[:-1]
            bid = props.get("id", max_bid + 1)
            blocks.append({"id": bid, "label": props.get("name", f"Block {bid}"), "polygon_lnglat": ring,
                           "vineyard_name": props.get("vineyard_name", ""), "region": props.get("region", "")})
            max_bid = max(max_bid, bid)
        elif props.get("feature_type") == "row" and geom.get("type") == "LineString":
            rid = props.get("id", max_rid + 1)
            rows.append({"id": rid, "block_id": props.get("block_id", 0), "control_points_lnglat": geom["coordinates"]})
            max_rid = max(max_rid, rid)
    return {"blocks": blocks, "rows": rows}, max_bid + 1, max_rid + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Browser-based vineyard annotation tool")
    parser.add_argument("--lat", type=float, default=-41.52)
    parser.add_argument("--lng", type=float, default=173.95)
    parser.add_argument("--zoom", type=int, default=15)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--name", type=str, default="untitled")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    annotations = {"blocks": [], "rows": []}
    next_block_id, next_row_id = 1, 1
    save_path = str(STANDALONE_DIR / f"{args.name}.geojson")

    if args.load:
        p = Path(args.load)
        if p.exists():
            annotations, next_block_id, next_row_id = load_geojson_annotations(p)
            save_path = str(p)
            logger.info("Loaded %d blocks + %d rows from %s", len(annotations["blocks"]), len(annotations["rows"]), p)
            if annotations["blocks"]:
                coords = annotations["blocks"][0]["polygon_lnglat"]
                if coords:
                    args.lng = sum(c[0] for c in coords) / len(coords)
                    args.lat = sum(c[1] for c in coords) / len(coords)

    STANDALONE_DIR.mkdir(parents=True, exist_ok=True)

    server = HTTPServer(("127.0.0.1", args.port), AnnotatorHandler)
    server.state = {
        "annotations": annotations, "save_path": save_path,
        "lat": args.lat, "lng": args.lng, "zoom": args.zoom,
        "next_block_id": next_block_id, "next_row_id": next_row_id,
    }

    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  Vineyard Map Annotator")
    print(f"  {url}")
    print(f"  Save path: {save_path}")
    print(f"  Press Ctrl+C to stop\n")

    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
