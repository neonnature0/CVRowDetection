/**
 * Map View — MapLibre GL + Terra Draw for adding block polygons.
 */

let _map = null;
let _drawControl = null;
const BLOCK_SOURCE = 'existing-blocks';
const BLOCK_FILL_LAYER = 'blocks-fill';
const BLOCK_LINE_LAYER = 'blocks-outline';

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
  if (_map) return;

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
    center: [173.95, -41.51],
    zoom: 14,
  });

  // Terra Draw control for polygon drawing
  _drawControl = new MaplibreTerradrawControl.MaplibreTerradrawControl({
    modes: ['polygon', 'select', 'delete-selection'],
    open: true,
  });
  _map.addControl(_drawControl, 'top-left');
  _map.addControl(new maplibregl.NavigationControl(), 'top-left');

  _map.on('load', () => {
    addBlockLayers();
    refreshBlockOverlays();

    // Poll for drawn features to enable save button
    // Terra Draw doesn't fire events the same way as MapboxDraw,
    // so we check periodically when the draw control exists
    setInterval(updateSaveButton, 1000);
  });

  document.getElementById('save-block-btn').addEventListener('click', saveDrawnBlock);
}

function updateSaveButton() {
  if (!_drawControl) return;
  try {
    const features = _drawControl.getFeatures();
    const hasPolygon = features && features.some(f => f.geometry && f.geometry.type === 'Polygon');
    document.getElementById('save-block-btn').disabled = !hasPolygon;
  } catch (e) {
    // Draw control may not be ready yet
  }
}

async function saveDrawnBlock() {
  if (!_drawControl) return;

  const features = _drawControl.getFeatures();
  const polygon = features.find(f => f.geometry && f.geometry.type === 'Polygon');
  if (!polygon) return;

  const btn = document.getElementById('save-block-btn');
  btn.disabled = true;
  btn.textContent = 'Saving...';

  try {
    await API.post('/api/blocks', { boundary: polygon.geometry });
    // Clear drawn features
    const ids = features.map(f => f.id).filter(Boolean);
    ids.forEach(id => {
      try { _drawControl.removeFeatures([id]); } catch (_) {}
    });
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
  _map.addSource(BLOCK_SOURCE, {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [] },
  });

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

// ── Lifecycle ──

document.addEventListener('alpine:init', () => {
  Alpine.effect(() => {
    const view = Alpine.store('app').view;
    if (view === 'map' && _map) {
      setTimeout(() => _map.resize(), 50);
    }
  });
});

document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    initMap();
    renderSidebarList();
  }, 200);
});
