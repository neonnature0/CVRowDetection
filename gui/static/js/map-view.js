/**
 * Map View — MapLibre GL + polygon drawing for adding blocks.
 *
 * Initialised when the #map view becomes active. Uses mapbox-gl-draw
 * (compatible with MapLibre) for polygon creation.
 */

let _map = null;
let _draw = null;
const BLOCK_SOURCE = 'existing-blocks';
const BLOCK_FILL_LAYER = 'blocks-fill';
const BLOCK_LINE_LAYER = 'blocks-outline';

// Stage → color mapping
const STAGE_COLORS = {
  draft:     'rgba(150, 150, 150, 0.25)',
  detected:  'rgba(96, 165, 250, 0.25)',
  annotated: 'rgba(74, 222, 128, 0.25)',
  training_ready: 'rgba(251, 191, 36, 0.25)',
  verified:  'rgba(251, 191, 36, 0.35)',
};
const STAGE_LINE_COLORS = {
  draft:     '#888',
  detected:  '#60a5fa',
  annotated: '#4ade80',
  training_ready: '#fbbf24',
  verified:  '#fbbf24',
};

function initMap() {
  if (_map) return; // already initialised

  _map = new maplibregl.Map({
    container: 'map-container',
    style: {
      version: 8,
      sources: {
        'linz-aerial': {
          type: 'raster',
          tiles: ['/api/tiles/linz/{z}/{x}/{y}'],
          tileSize: 256,
          maxzoom: 22,
          attribution: '&copy; LINZ CC-BY 4.0',
        },
      },
      layers: [{
        id: 'linz-tiles',
        type: 'raster',
        source: 'linz-aerial',
      }],
    },
    center: [173.95, -41.51],  // Marlborough
    zoom: 14,
  });

  // Drawing tool
  _draw = new MapboxDraw({
    displayControlsDefault: false,
    controls: { polygon: true, trash: true },
    defaultMode: 'simple_select',
  });
  _map.addControl(_draw, 'top-left');
  _map.addControl(new maplibregl.NavigationControl(), 'top-left');

  // Enable save button when a polygon is drawn
  _map.on('draw.create', updateSaveButton);
  _map.on('draw.delete', updateSaveButton);
  _map.on('draw.update', updateSaveButton);

  // Load existing blocks once map is ready
  _map.on('load', () => {
    addBlockLayers();
    refreshBlockOverlays();
  });

  // Wire up save button
  document.getElementById('save-block-btn').addEventListener('click', saveDrawnBlock);
}

function updateSaveButton() {
  const features = _draw.getAll().features;
  const hasPolygon = features.some(f => f.geometry.type === 'Polygon');
  document.getElementById('save-block-btn').disabled = !hasPolygon;
}

async function saveDrawnBlock() {
  const features = _draw.getAll().features;
  const polygon = features.find(f => f.geometry.type === 'Polygon');
  if (!polygon) return;

  const btn = document.getElementById('save-block-btn');
  btn.disabled = true;
  btn.textContent = 'Saving...';

  try {
    await API.post('/api/blocks', { boundary: polygon.geometry });
    // Clear drawn polygon
    _draw.deleteAll();
    // Refresh
    await Alpine.store('app').refreshBlocks();
    refreshBlockOverlays();
    renderSidebarList();
    btn.textContent = 'Save Block';
  } catch (e) {
    console.error('Save failed:', e);
    btn.textContent = 'Save Block';
    btn.disabled = false;
    alert('Failed to save block: ' + e.message);
  }
}

function addBlockLayers() {
  // GeoJSON source for existing blocks
  _map.addSource(BLOCK_SOURCE, {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [] },
  });

  // Fill layer (transparent, stage-colored)
  _map.addLayer({
    id: BLOCK_FILL_LAYER,
    type: 'fill',
    source: BLOCK_SOURCE,
    paint: {
      'fill-color': ['match', ['get', 'stage'],
        'draft',     STAGE_COLORS.draft,
        'detected',  STAGE_COLORS.detected,
        'annotated', STAGE_COLORS.annotated,
        'verified',  STAGE_COLORS.verified,
        'rgba(150,150,150,0.2)',
      ],
    },
  });

  // Outline layer
  _map.addLayer({
    id: BLOCK_LINE_LAYER,
    type: 'line',
    source: BLOCK_SOURCE,
    paint: {
      'line-color': ['match', ['get', 'stage'],
        'draft',     STAGE_LINE_COLORS.draft,
        'detected',  STAGE_LINE_COLORS.detected,
        'annotated', STAGE_LINE_COLORS.annotated,
        'verified',  STAGE_LINE_COLORS.verified,
        '#888',
      ],
      'line-width': 2,
    },
  });

  // Popup on click
  _map.on('click', BLOCK_FILL_LAYER, (e) => {
    const props = e.features[0].properties;
    const popup = new maplibregl.Popup({ closeButton: true, maxWidth: '200px' })
      .setLngLat(e.lngLat);
    const el = document.createElement('div');
    el.style.color = '#333';
    const nameEl = document.createElement('b');
    nameEl.textContent = props.name;
    el.appendChild(nameEl);
    el.appendChild(document.createElement('br'));
    el.appendChild(document.createTextNode('Stage: ' + (props.stage || 'draft')));
    popup.setDOMContent(el).addTo(_map);
  });
  _map.on('mouseenter', BLOCK_FILL_LAYER, () => { _map.getCanvas().style.cursor = 'pointer'; });
  _map.on('mouseleave', BLOCK_FILL_LAYER, () => { _map.getCanvas().style.cursor = ''; });
}

function refreshBlockOverlays() {
  const blocks = Alpine.store('app').blocks;
  const features = blocks
    .filter(b => b.boundary)
    .map(b => ({
      type: 'Feature',
      geometry: b.boundary,
      properties: { name: b.name, stage: b.stage || 'draft' },
    }));

  const source = _map.getSource(BLOCK_SOURCE);
  if (source) {
    source.setData({ type: 'FeatureCollection', features });
  }
}

function renderSidebarList() {
  const container = document.getElementById('sidebar-block-list');
  if (!container) return;
  // Clear existing content
  container.replaceChildren();

  const blocks = Alpine.store('app').blocks;
  blocks.forEach(b => {
    const item = document.createElement('div');
    item.className = 'block-item';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'name';
    nameSpan.textContent = b.name;

    const stageSpan = document.createElement('span');
    stageSpan.className = 'stage stage-' + (b.stage || 'draft');
    stageSpan.textContent = b.stage || 'draft';

    const delBtn = document.createElement('button');
    delBtn.title = 'Delete';
    delBtn.textContent = '\u2715';
    Object.assign(delBtn.style, {
      background: 'none', border: 'none', color: 'var(--danger)',
      cursor: 'pointer', fontSize: '14px', padding: '0 4px',
    });
    delBtn.addEventListener('click', () => deleteBlock(b.name));

    item.appendChild(nameSpan);
    item.appendChild(stageSpan);
    item.appendChild(delBtn);
    container.appendChild(item);
  });
}

async function deleteBlock(name) {
  if (!confirm('Delete block ' + name + '?')) return;
  try {
    await API.del('/api/blocks/' + name);
    await Alpine.store('app').refreshBlocks();
    refreshBlockOverlays();
    renderSidebarList();
  } catch (e) {
    alert('Delete failed: ' + e.message);
  }
}

// ── Lifecycle: init map when view becomes active ──

// Use a MutationObserver to detect when #map-container appears in the DOM
const _observer = new MutationObserver(() => {
  const el = document.getElementById('map-container');
  if (el && !_map) {
    // Small delay to ensure the container has dimensions
    setTimeout(() => {
      initMap();
      renderSidebarList();
    }, 50);
  }
});
_observer.observe(document.body, { childList: true, subtree: true });
